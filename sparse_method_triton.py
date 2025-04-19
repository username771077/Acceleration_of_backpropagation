import torch
import torch.nn as nn
import triton
import triton.language as tl
import traceback
from torch.nn.parameter import Parameter
import torch.autograd

SPARSITY_THRESHOLD = 1e-6

def check_tensors_gpu_ready(*tensors):
    """Ensure tensors are contiguous and on CUDA for Triton kernel calls."""
    for t in tensors:
        assert t is not None and t.is_cuda and t.is_contiguous(), \
               f"Tensor check failed: CUDA={t.is_cuda if t is not None else 'None'}, Contiguous={t.is_contiguous() if t is not None else 'None'}"

@triton.autotune(
    configs=[ 
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],

    key=['sparse_rows_to_keep', 'N', 'K'],
)
@triton.jit
def fixed_slice_matmul_kernel(
    A_ptr, B_ptr, C_ptr,          # Pointers: A=dY, B=W, C=dX
    M_full, N, K,                 # Full dimensions (M_full needed for stride A)
    stride_a_m, stride_a_k,       # Strides for A (dY)
    stride_b_k, stride_b_n,       # Strides for B (W)
    stride_c_m, stride_c_n,       # Strides for C (dX)
    sparse_rows_to_keep: tl.constexpr, # <<< Number of top rows to process >>>
    # --- Autotuner parameters ---
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    num_stages: tl.constexpr, num_warps: tl.constexpr,
):   
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(sparse_rows_to_keep, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n; group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M; group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m); pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_b_k + offs_bn[None, :] * stride_b_n)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_am[:, None] < sparse_rows_to_keep) & (offs_k[None, :] + k * BLOCK_K < K)
        b_mask = (offs_k[:, None] + k * BLOCK_K < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    c = accumulator.to(C_ptr.dtype.element_ty)

    offs_cm = offs_am
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_c_m * offs_cm[:, None] + stride_c_n * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < sparse_rows_to_keep) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# --- Python Wrapper for Sparse Method using Triton ---
def sparse_row_matmul_sparse_method_triton(
    dY: torch.Tensor, W: torch.Tensor, sparse_rows_to_keep: int, layer_idx: str = None
):
    
    dY = dY.contiguous(); W = W.contiguous() 
    check_tensors_gpu_ready(dY, W)
    dY_shape = dY.shape; dtype = dY.dtype; device = dY.device

    if dY.ndim > 2: M_total = dY.numel() // dY_shape[-1]; K = dY_shape[-1]; dY_2d = dY.reshape(M_total, K)
    elif dY.ndim == 2: M_total, K = dY.shape; dY_2d = dY
    else: raise ValueError(f"dY must have at least 2 dimensions, got {dY.ndim}")

    K_W, N = W.shape; assert K == K_W
    M = M_total

    N_dense = min(sparse_rows_to_keep, M) 
    if N_dense <= 0: return torch.zeros((M, N), device=device, dtype=dtype).reshape(dY_shape[:-1] + (N,))

    dX = torch.zeros((M, N), device=device, dtype=dtype)
    output_shape_final = dY_shape[:-1] + (N,)
    grid = lambda META: (triton.cdiv(N_dense, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    try:
        fixed_slice_matmul_kernel[grid](
            dY_2d, W, dX, M, N, K,
            dY_2d.stride(0), dY_2d.stride(1), W.stride(0), W.stride(1), dX.stride(0), dX.stride(1),
            sparse_rows_to_keep=N_dense 
        )
    except Exception as e:
        print(f"ERROR DURING Sparse Method Triton Kernel LAUNCH: {e}"); traceback.print_exc()
        return torch.zeros(output_shape_final, device=device, dtype=dtype)

    # Return the dX which was partially filled by the kernel
    try: return dX.view(output_shape_final)
    except RuntimeError: return dX.reshape(output_shape_final)


# --- AUTOGRAD FUNCTION (Calls Sparse Method Triton) ---
class SparseMethodLinearFunction(torch.autograd.Function):
    _sparse_rows_to_keep = 50

    @staticmethod
    def forward(ctx, input, weight, bias, layer_idx, layer_type):
        output = input @ weight.T
        if bias is not None: output += bias
        ctx.save_for_backward(weight, bias) 
        ctx.layer_idx = layer_idx; ctx.layer_type_str = 'intermediate' if layer_type == 0 else 'output'
        ctx.input_shape = input.shape 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx; layer_type_str = ctx.layer_type_str
        grad_input = grad_weight = grad_bias = None; layer_key = f"{layer_idx}-{layer_type_str}"

        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1] # Corresponds to weight in forward inputs
        needs_bias_grad = ctx.needs_input_grad[2]   # Corresponds to bias in forward inputs

        if bias is not None and needs_bias_grad:
            if grad_output is not None:
                 grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        if needs_weight_grad:
             grad_weight = None

        if needs_input_grad:
            if grad_output is not None:
                grad_input = sparse_row_matmul_sparse_method_triton(
                    grad_output, weight, 
                    sparse_rows_to_keep=SparseMethodLinearFunction._sparse_rows_to_keep,
                    layer_idx=layer_key
                )
            else:
                grad_input = torch.zeros(ctx.input_shape, device=weight.device, dtype=weight.dtype)

        return grad_input, grad_weight, grad_bias, None, None


# --- nn.Module wrappers (Pass layer_idx/type to forward) ---
class SparseMethodLinearIntermediate(nn.Module):
    """nn.Linear replacement using SparseMethodLinearFunction for intermediate layers."""
    def __init__(self, in_features, out_features, layer_idx, bias=True, device=None, dtype=None):
        super().__init__(); factory_kwargs={'device': device, 'dtype': dtype}
        self.in_features=in_features; self.out_features=out_features; self.layer_idx = layer_idx
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias: self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else: self.register_parameter('bias', None)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5);
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def from_linear(self, linear: nn.Linear):
        """Copies weights and bias from an existing nn.Linear layer."""
        if not isinstance(linear, nn.Linear): raise TypeError("Input 'linear' must be nn.Linear")
        if self.weight.shape != linear.weight.shape: raise ValueError(f"Weight shape mismatch: expected {self.weight.shape}, got {linear.weight.shape}")
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            if self.bias is None: self.bias = Parameter(torch.empty(self.out_features, device=linear.bias.device, dtype=linear.bias.dtype))
            if self.bias.shape != linear.bias.shape: raise ValueError(f"Bias shape mismatch: expected {self.bias.shape}, got {linear.bias.shape}")
            self.bias = nn.Parameter(linear.bias.data.clone())
        elif self.bias is not None: self.register_parameter('bias', None)
        return self

    def forward(self, x):
        return SparseMethodLinearFunction.apply(x, self.weight, self.bias, self.layer_idx, 0)

class SparseMethodLinearOutput(nn.Module):
    """nn.Linear replacement using SparseMethodLinearFunction for output layers."""
    def __init__(self, in_features, out_features, layer_idx, bias=True, device=None, dtype=None):
        super().__init__(); factory_kwargs={'device': device, 'dtype': dtype}
        self.in_features=in_features; self.out_features=out_features; self.layer_idx = layer_idx
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias: self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else: self.register_parameter('bias', None)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5);
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def from_linear(self, linear: nn.Linear):
        """Copies weights and bias from an existing nn.Linear layer."""
        if not isinstance(linear, nn.Linear): raise TypeError("Input 'linear' must be nn.Linear")
        if self.weight.shape != linear.weight.shape: raise ValueError(f"Weight shape mismatch: expected {self.weight.shape}, got {linear.weight.shape}")
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            if self.bias is None: self.bias = Parameter(torch.empty(self.out_features, device=linear.bias.device, dtype=linear.bias.dtype))
            if self.bias.shape != linear.bias.shape: raise ValueError(f"Bias shape mismatch: expected {self.bias.shape}, got {linear.bias.shape}")
            self.bias = nn.Parameter(linear.bias.data.clone())
        elif self.bias is not None: self.register_parameter('bias', None)
        return self

    def forward(self, x):
        return SparseMethodLinearFunction.apply(x, self.weight, self.bias, self.layer_idx, 1)

print("Sparse method Triton definitions complete.")
print("-" * 40)