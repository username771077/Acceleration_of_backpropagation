import torch
import torch.nn as nn
import os
import traceback
from collections import defaultdict
import statistics
from torch.nn.parameter import Parameter
import torch.autograd
import gc
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, AutoConfig,
    PreTrainedModel, get_scheduler
)
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.roberta.modeling_roberta import RobertaLayer
from torch.optim import AdamW
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
import types
import torch.profiler
import math
import matplotlib.pyplot as plt
from itertools import chain
import copy
import evaluate
import random

try:
    from sparse_method_triton import sparse_row_matmul_sparse_method_triton
    SPARSE_METHOD_TRITON_AVAILABLE = True
    print("Imported sparse_row_matmul_sparse_method_triton from sparse_method_triton.py")
except ImportError as e:
    print(f"WARNING: Could not import sparse_row_matmul_sparse_method_triton from sparse_method_triton.py: {e}")
    print("         The 'Sparse' method comparison will likely fail.")
    def sparse_row_matmul_sparse_method_triton(*args, **kwargs):
        raise NotImplementedError("sparse_row_matmul_sparse_method_triton could not be imported.")
    SPARSE_METHOD_TRITON_AVAILABLE = False


try:
    from dropbp.layer import DropBP
    from dropbp.handler import DropBPHandler
    DROPBP_AVAILABLE = True
    print("DropBP library imported successfully.")
except ImportError:
    print("WARNING: DropBP library not found. Install it to include DropBP comparison.")
    DROPBP_AVAILABLE = False
    class DropBP:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    class DropBPHandler:
        def __init__(self, model, drop_rate=0.0, **kwargs): pass
        def set_initial_drop_rate(self, rate): pass
        def set_dropped_layers(self): pass
        def detact_non_grad(self): return False
    def add_dropbp_to_model(model, seq_len): print("DropBP not available, skipping attachment."); return model

print("Imports complete.")
print("-" * 40)


SPARSITY_THRESHOLD = 1e-6

class LinearFunctionAccelarate(torch.autograd.Function):
    _N_dense_rows_to_keep = 50
    @staticmethod
    def forward(ctx, input, weight, bias, layer_idx, layer_type):
        output = input @ weight.T;
        if bias is not None: output += bias
        ctx.save_for_backward(weight, bias); ctx.layer_idx = layer_idx; ctx.layer_type_str = 'intermediate' if layer_type == 0 else 'output'; ctx.input_shape = input.shape; return output
    @staticmethod
    def backward(ctx, grad_output):
        weight, bias = ctx.saved_tensors; layer_idx = ctx.layer_idx; layer_type_str = ctx.layer_type_str; grad_input = grad_weight = grad_bias = None; layer_key = f"{layer_idx}-{layer_type_str}";
        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        needs_bias_grad = ctx.needs_input_grad[2]

        if bias is not None and needs_bias_grad:
            if grad_output is not None: grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
            else: grad_bias = torch.zeros_like(bias)

        if needs_weight_grad:
            grad_weight = None

        if needs_input_grad:
            if grad_output is not None:
                grad_input = sparse_row_matmul_sparse_method_triton(
                    grad_output, weight,
                    N_dense_rows_to_keep=LinearFunctionAccelarate._N_dense_rows_to_keep,
                    layer_idx=layer_key
                )
            else: grad_input = torch.zeros(ctx.input_shape, device=weight.device, dtype=weight.dtype)

        return grad_input, grad_weight, grad_bias, None, None

class TritonSparseLinearIntermediate(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, bias=True, device=None, dtype=None):
        super().__init__(); factory_kwargs={'device': device, 'dtype': dtype}; self.in_features=in_features; self.out_features=out_features; self.layer_idx = layer_idx; self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs));
        if bias: self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else: self.register_parameter('bias', None)
        self._init_weights()
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5));
        if self.bias is not None: fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight); bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0; nn.init.uniform_(self.bias, -bound, bound)
    def from_linear(self, linear: nn.Linear):
        if not isinstance(linear, nn.Linear): raise TypeError("Input 'linear' must be nn.Linear"); assert self.weight.shape == linear.weight.shape, f"Weight shape mismatch: {self.weight.shape} vs {linear.weight.shape}"; self.weight = nn.Parameter(linear.weight.data.clone());
        if linear.bias is not None:
            if self.bias is None: self.bias = Parameter(torch.empty(self.out_features, device=linear.bias.device, dtype=linear.bias.dtype));
            assert self.bias.shape == linear.bias.shape, f"Bias shape mismatch: {self.bias.shape} vs {linear.bias.shape}"; self.bias = nn.Parameter(linear.bias.data.clone());
        elif self.bias is not None: self.register_parameter('bias', None);
        return self
    def forward(self, x): return LinearFunctionAccelarate.apply(x, self.weight, self.bias, self.layer_idx, 0)

class TritonSparseLinearOutput(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, bias=True, device=None, dtype=None):
        super().__init__(); factory_kwargs={'device': device, 'dtype': dtype}; self.in_features=in_features; self.out_features=out_features; self.layer_idx = layer_idx; self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs));
        if bias: self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else: self.register_parameter('bias', None)
        self._init_weights()
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5));
        if self.bias is not None: fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight); bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0; nn.init.uniform_(self.bias, -bound, bound)
    def from_linear(self, linear: nn.Linear):
        if not isinstance(linear, nn.Linear): raise TypeError("Input 'linear' must be nn.Linear"); assert self.weight.shape == linear.weight.shape, f"Weight shape mismatch: {self.weight.shape} vs {linear.weight.shape}"; self.weight = nn.Parameter(linear.weight.data.clone());
        if linear.bias is not None:
            if self.bias is None: self.bias = Parameter(torch.empty(self.out_features, device=linear.bias.device, dtype=linear.bias.dtype));
            assert self.bias.shape == linear.bias.shape, f"Bias shape mismatch: {self.bias.shape} vs {linear.bias.shape}"; self.bias = nn.Parameter(linear.bias.data.clone());
        elif self.bias is not None: self.register_parameter('bias', None);
        return self
    def forward(self, x): return LinearFunctionAccelarate.apply(x, self.weight, self.bias, self.layer_idx, 1)

def replace_layers_with_triton_sparse(model: PreTrainedModel, layers_to_modify: list = None):
    model_type = model.config.model_type.lower()
    action = "Replacing ALL" if layers_to_modify is None else f"Replacing layers {layers_to_modify} in"
    print(f"! {action} {model_type.upper()} FFN layers with TRITON SPARSE backward !");
    device = next(model.parameters()).device; dtype = next(model.parameters()).dtype; encoder = None;

    if model_type == 'bert':
        if hasattr(model, "bert") and hasattr(model.bert, "encoder"): encoder = model.bert.encoder
    elif model_type == 'roberta':
         if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"): encoder = model.roberta.encoder
    if encoder is None and hasattr(model, "encoder"):
        encoder = model.encoder
        print("Warning: Using generic 'encoder' attribute.")
    if encoder is None:
        print(f"Warning: Could not find standard {model_type.upper()} encoder structure."); return model

    layers_replaced_count = 0;
    if not hasattr(encoder, "layer"): print("Warning: Encoder does not have 'layer' attribute."); return model
    modify_set = set(layers_to_modify) if layers_to_modify is not None else None;

    for i, layer in enumerate(encoder.layer):
        should_modify = (modify_set is None or i in modify_set)
        intermediate_module = getattr(layer, "intermediate", None);
        if intermediate_module:
            dense_layer = getattr(intermediate_module, "dense", None);
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; h, t = orig.weight.shape
                    new = TritonSparseLinearIntermediate(t, h, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig);
                    layer.intermediate.dense = new; layers_replaced_count +=1;
        output_module = getattr(layer, "output", None);
        if output_module:
            dense_layer_out = getattr(output_module, "dense", None);
            if dense_layer_out and isinstance(dense_layer_out, nn.Linear):
                if should_modify:
                    orig = dense_layer_out; t_out, h_out = orig.weight.shape
                    new = TritonSparseLinearOutput(h_out, t_out, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig);
                    layer.output.dense = new; layers_replaced_count += 1;

    if layers_replaced_count > 0: print(f"! Triton Sparse Layer replacement complete ({layers_replaced_count} Linear layers modified) !")
    elif layers_to_modify is not None: print(f"! No layers were replaced for specified indices {layers_to_modify} (Triton Sparse). Check indices/structure. !")
    else: print("! No layers were replaced (Triton Sparse). Check model structure or replacement logic. !")
    return model

print("Custom (Sparse Method using imported Triton) definitions complete.")
print("-" * 40)


def add_dropbp_to_bert_layer(layer, config, seq_length):
    if not DROPBP_AVAILABLE: return
    if not isinstance(layer, BertLayer): print(f"Warning: Expected BertLayer, got {type(layer)}. Skipping DropBP."); return
    if hasattr(layer, 'dropbp_attn'): return

    hidden_size = config.hidden_size; attn_flops = 8 * hidden_size ** 2 + 4 * hidden_size * seq_length; mlp_flops = 16 * hidden_size ** 2
    layer.dropbp_attn = DropBP(flops=attn_flops); layer.dropbp_mlp = DropBP(flops=mlp_flops)
    if not hasattr(layer, '_original_forward'): layer._original_forward = layer.forward

    def custom_forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                       encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attention_outputs = self.attention.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output_raw = self_attention_outputs[0]; attention_output_dropped = self.dropbp_attn(attention_output_raw); attention_output_final = self.attention.output(attention_output_dropped, hidden_states)
        intermediate_output = self.intermediate(attention_output_final); ffn_output_dropped = self.dropbp_mlp(intermediate_output); layer_output = self.output(ffn_output_dropped, attention_output_final)
        outputs = (layer_output,) + self_attention_outputs[1:]; return outputs
    layer.forward = custom_forward.__get__(layer, BertLayer)

def add_dropbp_to_roberta_layer(layer, config, seq_length):
    if not DROPBP_AVAILABLE: return
    if not isinstance(layer, RobertaLayer): print(f"Warning: Expected RobertaLayer, got {type(layer)}. Skipping DropBP."); return
    if hasattr(layer, 'dropbp_attn'): return

    hidden_size = config.hidden_size; attn_flops = 8 * hidden_size ** 2 + 4 * hidden_size * seq_length; mlp_flops = 16 * hidden_size ** 2
    layer.dropbp_attn = DropBP(flops=attn_flops); layer.dropbp_mlp = DropBP(flops=mlp_flops)
    if not hasattr(layer, '_original_forward'): layer._original_forward = layer.forward

    def custom_forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                       encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attention_outputs = self.attention.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output_raw = self_attention_outputs[0]; attention_output_dropped = self.dropbp_attn(attention_output_raw); attention_output_final = self.attention.output(attention_output_dropped, hidden_states)
        intermediate_output = self.intermediate(attention_output_final); ffn_output_dropped = self.dropbp_mlp(intermediate_output); layer_output = self.output(ffn_output_dropped, attention_output_final)
        outputs = (layer_output,) + self_attention_outputs[1:]; return outputs
    layer.forward = custom_forward.__get__(layer, RobertaLayer)

def add_dropbp_to_model(model, seq_length):
    if not DROPBP_AVAILABLE: return model
    print(f"Attaching DropBP modules to {model.config.model_type} model...")
    config = model.config; encoder = None; layer_type = None; add_func = None

    if hasattr(model, "bert") and hasattr(model.bert, "encoder"): encoder = model.bert.encoder; layer_type = BertLayer; add_func = add_dropbp_to_bert_layer; print(" -> Detected BERT structure")
    elif hasattr(model, "roberta") and hasattr(model.roberta, "encoder"): encoder = model.roberta.encoder; layer_type = RobertaLayer; add_func = add_dropbp_to_roberta_layer; print(" -> Detected RoBERTa structure")
    else: print("Warning: Unsupported model structure for DropBP attachment."); return model
    if not hasattr(encoder, "layer"): print("Warning: Encoder does not have 'layer' attribute."); return model

    layers_modified = 0
    for i, layer in enumerate(encoder.layer):
        if isinstance(layer, layer_type) and add_func is not None: add_func(layer, config, seq_length); layers_modified += 1
    print(f"DropBP potentially attached to {layers_modified} layers.")
    return model

print("DropBP functions defined.")
print("-" * 40)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def prepare_glue_data(task_name: str, tokenizer: AutoTokenizer, batch_size: int, seq_length: int, split_name: str):
    is_stsb = (task_name == "stsb")
    task_to_keys = {"cola": ("sentence", None), "mnli": ("premise", "hypothesis"), "mrpc": ("sentence1", "sentence2"), "qnli": ("question", "sentence"), "qqp": ("question1", "question2"), "rte": ("sentence1", "sentence2"), "sst2": ("sentence", None), "wnli": ("sentence1", "sentence2"), "ax": ("premise", "hypothesis"),}
    try:
        dataset = load_dataset("glue", task_name, trust_remote_code=True, cache_dir="./hf_cache")
        validation_split_name = "validation_matched" if task_name == "mnli" else "validation"
        if task_name == "mnli" and validation_split_name not in dataset: validation_split_name = "validation_mismatched"
        if task_name == 'ax': validation_split_name = 'test'

        if split_name == 'validation': current_split_name = validation_split_name
        elif split_name == 'train': current_split_name = 'train'
        else: current_split_name = split_name

        if current_split_name not in dataset: print(f"Error: Split '{current_split_name}' not found for GLUE task {task_name}."); return None
        ds_split = dataset[current_split_name]
        if split_name == 'train': ds_split = ds_split.shuffle(seed=random.randint(1,10000))
    except Exception as e: print(f"Failed load GLUE task {task_name}: {e}"); return None

    s1k, s2k = task_to_keys.get(task_name, ("sentence1", "sentence2"))
    if s1k not in ds_split.column_names or (s2k is not None and s2k not in ds_split.column_names):
        if not (task_name in ['cola', 'sst2'] and s1k in ds_split.column_names and s2k is None): print(f"Error: Keys '{s1k}'/'{s2k}' not found in {current_split_name} columns: {ds_split.column_names}. Task: {task_name}"); return None
        else: s2k = None
    def tok_fn(ex):
        args = (ex[s1k],) if s2k is None else (ex[s1k], ex[s2k])
        args = tuple(list(map(lambda x: "" if x is None else str(x), a)) if isinstance(a, list) else ("" if a is None else str(a)) for a in args)
        return tokenizer(*args, padding="max_length", truncation=True, max_length=seq_length)
    try:
        cols_to_keep = ['input_ids', 'attention_mask', 'token_type_ids', 'label', 'labels', 'idx']
        cols_rem = [c for c in ds_split.column_names if c not in cols_to_keep]
        tok_ds = ds_split.map(tok_fn, batched=True, remove_columns=cols_rem, load_from_cache_file=True, desc=f"Tokenizing {task_name} {split_name}")
    except Exception as e: print(f"Tokenize error: {e}"); traceback.print_exc(); return None
    label_col = 'label'; final_label_col = 'labels'
    if label_col not in tok_ds.column_names and final_label_col not in tok_ds.column_names: tok_ds = tok_ds.add_column(final_label_col, [0] * len(tok_ds))
    elif label_col in tok_ds.column_names and final_label_col not in tok_ds.column_names: tok_ds = tok_ds.rename_column(label_col, final_label_col)

    if is_stsb:
        def convert_label_to_float(example): example[final_label_col] = float(example[final_label_col]); return example
        tok_ds = tok_ds.map(convert_label_to_float, load_from_cache_file=True, desc="Convert labels to float")
    else:
        def convert_label_to_long(example):
            try: example[final_label_col] = int(example[final_label_col])
            except (ValueError, TypeError): example[final_label_col] = 0
            return example
        tok_ds = tok_ds.map(convert_label_to_long, load_from_cache_file=True, desc="Convert labels to long")

    req_cols = ['input_ids', 'attention_mask']; m_name = tokenizer.name_or_path.lower()
    if 'bert' in m_name and 'token_type_ids' in tok_ds.column_names: req_cols.append('token_type_ids')
    final_cols = [c for c in req_cols + [final_label_col] if c in tok_ds.column_names]
    tok_ds.set_format("torch", columns=final_cols)
    dl = DataLoader(tok_ds, batch_size=batch_size, shuffle=(split_name == 'train'))
    return dl

def evaluate_accuracy(model, eval_dataloader, device, task_name):
    metric = evaluate.load('accuracy', cache_dir="./eval_cache")
    model.eval()
    all_preds = []; all_labels = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad(): outputs = model(**batch)
        logits = outputs.logits; predictions = torch.argmax(logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy()); all_labels.extend(batch["labels"].cpu().numpy())
    model.train()
    try: eval_metric = metric.compute(predictions=all_preds, references=all_labels); accuracy = eval_metric['accuracy']; return accuracy
    except Exception as e: print(f"Error computing metric for {task_name}: {e}"); return 0.0

def measure_latency(model: nn.Module, input_batch: dict, num_repeats: int = 10, num_warmup: int = 3, task_name: str = None):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for latency measurement.")
    device = next(model.parameters()).device
    is_regression = (task_name == 'stsb')

    try:
        input_batch_gpu = {k: v.to(device) for k, v in input_batch.items() if isinstance(v, torch.Tensor)}
        if 'labels' not in input_batch_gpu:
            print("Warning: 'labels' not found in batch for latency measurement backward pass.")
            input_batch_gpu['labels'] = torch.zeros(input_batch_gpu['input_ids'].shape[0], dtype=torch.long if not is_regression else torch.float, device=device)
    except Exception as e:
        print(f"Error moving batch to GPU for latency measurement: {e}")
        return float('nan'), float('nan')

    forward_times = []; backward_times = []
    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    model.eval()
    for _ in range(num_warmup):
        try:
            with torch.no_grad(): _ = model(**input_batch_gpu)
            model.train()
            outputs = model(**input_batch_gpu)
            logits = outputs.logits; labels = input_batch_gpu['labels']
            if is_regression: loss = loss_fn(logits.squeeze(-1), labels.float())
            else: loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1).long())
            loss.backward()
            model.zero_grad(set_to_none=True); model.eval()
        except Exception as e:
            print(f"Latency Warmup Exception: {e}"); traceback.print_exc()
            return float('nan'), float('nan')

    torch.cuda.synchronize()

    for i in range(num_repeats):
        try:
            model.eval()
            fwd_start = torch.cuda.Event(enable_timing=True); fwd_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()
            with torch.no_grad(): outputs = model(**input_batch_gpu)
            fwd_end.record(); torch.cuda.synchronize()
            forward_times.append(fwd_start.elapsed_time(fwd_end))

            model.train()
            outputs = model(**input_batch_gpu)
            logits = outputs.logits; labels = input_batch_gpu['labels']
            if is_regression: loss = loss_fn(logits.squeeze(-1), labels.float())
            else: loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1).long())
            torch.cuda.synchronize()

            bwd_start = torch.cuda.Event(enable_timing=True); bwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start.record()
            loss.backward()
            bwd_end.record(); torch.cuda.synchronize()
            backward_times.append(bwd_start.elapsed_time(bwd_end))

            model.zero_grad(set_to_none=True); model.eval()

        except Exception as e:
            print(f"Latency Measurement Exception (Repeat {i+1}): {e}"); traceback.print_exc()
            forward_times.append(float('nan')); backward_times.append(float('nan'))

    avg_fwd = np.nanmean(forward_times) if forward_times else 0.0
    avg_bwd = np.nanmean(backward_times) if backward_times else 0.0
    return avg_fwd, avg_bwd

def plot_accuracy_comparison(task_name, results, output_dir, model_name, n_dense, drop_rate):
    plt.figure(figsize=(10, 6))
    colors = {"Original": "blue", "Sparse": "green", "DropBP": "red"}
    markers = {"Original": "o", "Sparse": "s", "DropBP": "^"}
    linestyles = {"Original": "-", "Sparse": "--", "DropBP": ":"}

    for method, data in results.items():
        if data.get('steps') and data.get('mean_accuracies'):
            steps, means, stds = data['steps'], data['mean_accuracies'], data['std_accuracies']
            plot_data_steps = steps; plot_data_means = means; plot_data_stds = stds
            if steps and steps[0] == 0:
                plot_data_steps = steps[1:]; plot_data_means = means[1:]; plot_data_stds = stds[1:]

            if plot_data_steps:
                final_plot_steps = np.array([0] + plot_data_steps)
                final_plot_means = np.array([0.0] + plot_data_means)
                final_plot_stds = np.array([0.0] + plot_data_stds)
            else:
                 final_plot_steps = np.array([0] + (steps if steps else []))
                 final_plot_means = np.array([0.0] + (means if means else []))
                 final_plot_stds = np.array([0.0] + (stds if stds else []))

            plt.errorbar(final_plot_steps, final_plot_means, yerr=final_plot_stds, label=method,
                         color=colors.get(method, "black"), marker=markers.get(method, '.'),
                         linestyle=linestyles.get(method, '-'), markersize=5, capsize=3, elinewidth=1, markeredgewidth=1)

    plt.xlabel("Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Accuracy: {task_name.upper()}")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tight_layout()

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True); safe_model_name = model_name.replace('/', '_')
            plot_filename = f"{safe_model_name}_{task_name}_SparseN{n_dense}_DropBP{drop_rate}_accuracy.png"
            plot_path = os.path.join(output_dir, plot_filename); plt.savefig(plot_path); print(f"Plot saved to: {plot_path}")
        except Exception as e: print(f"Error saving plot for {task_name}: {e}")
    plt.show()
    plt.close()

print("Helper functions defined.")
print("-" * 40)


def train_and_evaluate_accuracy(run_num, model_type, model, tokenizer, task_name, device, config, dropbp_handler=None):
    current_seed = config.random_seed_base + run_num
    set_seed(current_seed)

    current_seq_length = config.seq_length
    train_dataloader = prepare_glue_data(task_name, tokenizer, config.batch_size, current_seq_length, split_name='train')
    eval_dataloader = prepare_glue_data(task_name, tokenizer, config.batch_size, current_seq_length, split_name='validation')
    if train_dataloader is None or eval_dataloader is None: return [], []

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps_actual = config.num_train_steps
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps_actual)

    model.to(device)

    eval_steps = []; eval_accuracies = []
    global_step = 0; start_time = time.time()

    initial_accuracy = evaluate_accuracy(model, eval_dataloader, device, task_name)
    eval_steps.append(0); eval_accuracies.append(initial_accuracy)
    print(f"  Run {run_num+1} Initial Accuracy: {initial_accuracy:.4f}")

    model.train()
    train_iterator = iter(train_dataloader)
    while global_step < num_training_steps_actual:
        try: batch = next(train_iterator)
        except StopIteration:
            print(f"  Re-initializing training data iterator at step {global_step} (Run {run_num+1})")
            train_iterator = iter(train_dataloader)
            try: batch = next(train_iterator)
            except StopIteration: print("  Error: Training data loader is empty even after re-initialization. Stopping run."); break

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        if dropbp_handler: dropbp_handler.set_dropped_layers()

        try:
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                logits = outputs.logits; labels = batch['labels']; loss_fn = nn.CrossEntropyLoss();
                is_regression = (task_name == 'stsb')
                if is_regression: loss = loss_fn(logits.squeeze(-1), labels.float())
                else: loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1).long())

            non_grad = False
            if dropbp_handler: non_grad = dropbp_handler.detact_non_grad()

            if not non_grad:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else: optimizer.zero_grad()

        except Exception as e:
            print(f"Error during training step {global_step + 1} (Run {run_num+1}): {e}"); traceback.print_exc()
            optimizer.zero_grad()

        global_step += 1

        if global_step % config.eval_every == 0 or global_step >= num_training_steps_actual:
            if global_step > 0:
                accuracy = evaluate_accuracy(model, eval_dataloader, device, task_name)
                eval_steps.append(global_step); eval_accuracies.append(accuracy)
                print(f"  Run {run_num+1} Step {global_step}: Accuracy={accuracy:.4f}")
            model.train()
            if global_step >= num_training_steps_actual: break

    print(f"  Run {run_num+1} finished. Final Accuracy: {eval_accuracies[-1]:.4f} at step {eval_steps[-1]}")
    return eval_steps, eval_accuracies


def run_accuracy_comparison(config):
    if not torch.cuda.is_available(): print("CUDA required."); return
    device = torch.device("cuda"); print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Model: {config.model_name}"); print(f"Tasks: {config.tasks}")
    print(f"GLUE SL: {config.seq_length}")
    print(f"BS: {config.batch_size}, Steps: {config.num_train_steps}, Eval Every: {config.eval_every}, LR: {config.learning_rate}")
    print(f"Number of Runs per Method (Accuracy): {config.num_runs}")
    print(f"Latency Measurement Repeats: {config.latency_repeats}, Warmup: {config.latency_warmup}")


    LinearFunctionAccelarate._N_dense_rows_to_keep = config.N_dense_rows_to_keep
    print(f"Using Sparse Method (Triton): N_dense = {config.N_dense_rows_to_keep}")
    if DROPBP_AVAILABLE: print(f"Using DropBP Method (Initial Rate: {config.dropbp_initial_drop_rate}).")
    else: print("DropBP library not found, skipping DropBP comparison.")

    try: tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except Exception as e: print(f"Load Tokenizer Error: {e}"); traceback.print_exc(); return

    latency_results = []

    for task in config.tasks:
        print(f"\n{'='*25} Task: {task.upper()} {'='*25}")

        print(f"\n--- Measuring Latency for Task: {task} ---")
        task_latency = {"Task": task}
        latency_dataloader = prepare_glue_data(task, tokenizer, config.batch_size, config.seq_length, split_name='validation')
        if latency_dataloader is None:
            print(f"Could not load data for latency measurement of task {task}. Skipping latency.")
            sample_batch = None
        else:
            try: sample_batch = next(iter(latency_dataloader))
            except StopIteration:
                print(f"Validation dataloader for {task} is empty. Skipping latency.")
                sample_batch = None

        if sample_batch:
            methods_for_latency = ["Original", "Sparse"]
            if DROPBP_AVAILABLE: methods_for_latency.append("DropBP")

            for method_name in methods_for_latency:
                clear_gpu_cache(); model_latency = None; model_base_latency = None
                print(f"  Measuring latency for: {method_name}")
                try:
                    model_config_latency = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
                    task_labels = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2, "ax": 3}
                    num_labels = 1 if task == 'stsb' else task_labels.get(task, 2)
                    model_config_latency.num_labels = num_labels
                    if getattr(model_config_latency, "id2label", None) is None: model_config_latency.id2label = {i: f"LABEL_{i}" for i in range(model_config_latency.num_labels)}; model_config_latency.label2id = {v: k for k, v in model_config_latency.id2label.items()}

                    model_latency = AutoModelForSequenceClassification.from_pretrained(config.model_name, config=model_config_latency, trust_remote_code=True)

                    if method_name == "Sparse":
                        if not SPARSE_METHOD_TRITON_AVAILABLE:
                             print(f"Skipping Sparse latency as Triton wrapper import failed.")
                             task_latency[f"{method_name} Forward (ms)"] = float('nan')
                             task_latency[f"{method_name} Backward (ms)"] = float('nan')
                             continue
                        model_latency = replace_layers_with_triton_sparse(model_latency, layers_to_modify=config.layers_to_modify);
                    elif method_name == "DropBP":
                        model_latency = add_dropbp_to_model(model_latency, config.seq_length)

                    model_latency.to(device).eval()

                    fwd_ms, bwd_ms = measure_latency(
                        model_latency, sample_batch, config.latency_repeats, config.latency_warmup, task_name=task
                    )
                    task_latency[f"{method_name} Forward (ms)"] = fwd_ms
                    task_latency[f"{method_name} Backward (ms)"] = bwd_ms
                    print(f"    {method_name} - Fwd: {fwd_ms:.3f} ms, Bwd: {bwd_ms:.3f} ms")

                except Exception as e:
                    print(f"Error during latency measurement for {method_name} on {task}: {e}")
                    traceback.print_exc()
                    task_latency[f"{method_name} Forward (ms)"] = float('nan')
                    task_latency[f"{method_name} Backward (ms)"] = float('nan')
                finally:
                     del model_latency; del model_base_latency; clear_gpu_cache()

            latency_results.append(task_latency)


        task_accuracy_run_results = {
            "Original": {'steps': None, 'all_accuracies': [], 'mean_accuracies': [], 'std_accuracies': []},
            "Sparse": {'steps': None, 'all_accuracies': [], 'mean_accuracies': [], 'std_accuracies': []}
        }
        methods_for_accuracy = ["Original", "Sparse"]
        if DROPBP_AVAILABLE:
            task_accuracy_run_results["DropBP"] = {'steps': None, 'all_accuracies': [], 'mean_accuracies': [], 'std_accuracies': []}
            methods_for_accuracy.append("DropBP")

        for method_name in methods_for_accuracy:
            print(f"\n--- Running Accuracy Method: {method_name} for {config.num_runs} runs ---")
            method_accuracies_across_runs = []
            common_eval_steps = None

            if method_name == "Sparse" and not SPARSE_METHOD_TRITON_AVAILABLE:
                 print("Skipping Sparse accuracy run as Triton wrapper import failed.")
                 task_accuracy_run_results[method_name]['steps'] = []; task_accuracy_run_results[method_name]['mean_accuracies'] = []; task_accuracy_run_results[method_name]['std_accuracies'] = []
                 continue

            for run_i in range(config.num_runs):
                print(f"  Starting Accuracy Run {run_i+1}/{config.num_runs}...")
                clear_gpu_cache()
                model_instance = None; model_base_state_dict = None
                current_dropbp_handler = None

                try:
                    model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
                    task_labels = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2, "ax": 3}
                    num_labels = 1 if task == 'stsb' else task_labels.get(task, 2)
                    model_config.num_labels = num_labels
                    if getattr(model_config, "id2label", None) is None: model_config.id2label = {i: f"LABEL_{i}" for i in range(model_config.num_labels)}; model_config.label2id = {v: k for k, v in model_config.id2label.items()}

                    model_base_state_dict = AutoModelForSequenceClassification.from_pretrained(config.model_name, config=copy.deepcopy(model_config), trust_remote_code=True).state_dict()
                    model_instance = AutoModelForSequenceClassification.from_pretrained(config.model_name, config=copy.deepcopy(model_config), trust_remote_code=True)
                    model_instance.load_state_dict(model_base_state_dict)

                    if method_name == "Original": pass
                    elif method_name == "Sparse":
                        model_instance = replace_layers_with_triton_sparse(model_instance, layers_to_modify=config.layers_to_modify);
                    elif method_name == "DropBP":
                        model_instance = add_dropbp_to_model(model_instance, config.seq_length)
                        if DROPBP_AVAILABLE:
                            current_dropbp_handler = DropBPHandler(model_instance, config.dropbp_initial_drop_rate)

                    steps, accs = train_and_evaluate_accuracy(
                        run_i, method_name, model_instance, tokenizer, task, device, config,
                        dropbp_handler=current_dropbp_handler
                    )

                    if steps and accs:
                        method_accuracies_across_runs.append(accs)
                        if common_eval_steps is None: common_eval_steps = steps
                        elif common_eval_steps != steps:
                            print(f"Warning: Mismatched eval steps! R{run_i+1}:{steps} vs Expected:{common_eval_steps}. Aligning...")
                            min_len = min(len(common_eval_steps), len(steps))
                            common_eval_steps = common_eval_steps[:min_len]
                            method_accuracies_across_runs = [run_accs[:min_len] for run_accs in method_accuracies_across_runs]

                except Exception as e: print(f"Error during {method_name} Accuracy R{run_i+1} T{task}: {e}"); traceback.print_exc()
                finally: del model_instance; del model_base_state_dict; clear_gpu_cache()

            if method_accuracies_across_runs and common_eval_steps is not None and len(method_accuracies_across_runs) > 0:
                min_len = min(len(acc_list) for acc_list in method_accuracies_across_runs)
                if min_len < len(common_eval_steps): common_eval_steps = common_eval_steps[:min_len]
                aligned_accuracies = [acc_list[:min_len] for acc_list in method_accuracies_across_runs]

                mean_accs = np.mean(aligned_accuracies, axis=0).tolist()
                std_accs = np.std(aligned_accuracies, axis=0).tolist()
                task_accuracy_run_results[method_name]['steps'] = common_eval_steps
                task_accuracy_run_results[method_name]['mean_accuracies'] = mean_accs
                task_accuracy_run_results[method_name]['std_accuracies'] = std_accs
                if mean_accs:
                     print(f"  Aggregated {method_name} Accuracy. Final Mean Acc: {mean_accs[-1]:.4f} +/- {std_accs[-1]:.4f}")
                else: print(f"  Aggregated {method_name} Accuracy. No data points.")
            else:
              print(f"  No successful accuracy runs aggregated for {method_name} on task {task}.")
              task_accuracy_run_results[method_name]['steps'] = []; task_accuracy_run_results[method_name]['mean_accuracies'] = []; task_accuracy_run_results[method_name]['std_accuracies'] = []

        print(f"\nGenerating accuracy plot with error bars for {task}...")
        plot_accuracy_comparison(
            task, task_accuracy_run_results, config.output_dir, config.model_name,
            config.N_dense_rows_to_keep, config.dropbp_initial_drop_rate
        )

    if latency_results:
        print("\n" + "="*60)
        print("--- Performance Comparison (Latency) ---")
        print("="*60)
        perf_df = pd.DataFrame(latency_results)

        methods = ["Sparse"]
        if DROPBP_AVAILABLE: methods.append("DropBP")

        for method in methods:
            fwd_col = f"{method} Forward (ms)"
            bwd_col = f"{method} Backward (ms)"
            if fwd_col in perf_df.columns and "Original Forward (ms)" in perf_df.columns:
                perf_df[f"{method} Fwd Speedup"] = pd.to_numeric(perf_df["Original Forward (ms)"], errors='coerce') / pd.to_numeric(perf_df[fwd_col], errors='coerce')
            else: perf_df[f"{method} Fwd Speedup"] = np.nan
            if bwd_col in perf_df.columns and "Original Backward (ms)" in perf_df.columns:
                perf_df[f"{method} Bwd Speedup"] = pd.to_numeric(perf_df["Original Backward (ms)"], errors='coerce') / pd.to_numeric(perf_df[bwd_col], errors='coerce')
            else: perf_df[f"{method} Bwd Speedup"] = np.nan

        float_cols = [col for col in perf_df.columns if '(ms)' in col]
        speedup_cols = [col for col in perf_df.columns if 'Speedup' in col]

        for col in float_cols: perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notnull(x) else 'NaN')
        for col in speedup_cols: perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce').apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else '-')

        col_order = ["Task", "Original Forward (ms)", "Original Backward (ms)"]
        if "Sparse Forward (ms)" in perf_df.columns:
            col_order.extend(["Sparse Forward (ms)", "Sparse Fwd Speedup", "Sparse Backward (ms)", "Sparse Bwd Speedup"])
        if DROPBP_AVAILABLE and "DropBP Forward (ms)" in perf_df.columns:
            col_order.extend(["DropBP Forward (ms)", "DropBP Fwd Speedup", "DropBP Backward (ms)", "DropBP Bwd Speedup"])
        col_order = [col for col in col_order if col in perf_df.columns]
        perf_df = perf_df[col_order]

        print(perf_df.to_markdown(index=False))

        if config.output_dir:
            try:
                os.makedirs(config.output_dir, exist_ok=True); safe_model_name = config.model_name.replace('/', '_')
                latency_filename = f"{safe_model_name}_SparseN{config.N_dense_rows_to_keep}_DropBP{config.dropbp_initial_drop_rate}_latency.csv"
                latency_path = os.path.join(config.output_dir, latency_filename)
                perf_df_numeric = pd.DataFrame(latency_results)
                for method in methods:
                     if f"{method} Forward (ms)" in perf_df_numeric.columns and "Original Forward (ms)" in perf_df_numeric.columns:
                         perf_df_numeric[f"{method} Fwd Speedup"] = pd.to_numeric(perf_df_numeric["Original Forward (ms)"], errors='coerce') / pd.to_numeric(perf_df_numeric[f"{method} Forward (ms)"], errors='coerce')
                     if f"{method} Backward (ms)" in perf_df_numeric.columns and "Original Backward (ms)" in perf_df_numeric.columns:
                         perf_df_numeric[f"{method} Bwd Speedup"] = pd.to_numeric(perf_df_numeric["Original Backward (ms)"], errors='coerce') / pd.to_numeric(perf_df_numeric[f"{method} Backward (ms)"], errors='coerce')
                perf_df_numeric = perf_df_numeric[col_order]
                perf_df_numeric.to_csv(latency_path, index=False, float_format='%.4f')
                print(f"\nLatency results saved to {latency_path}")
            except Exception as e: print(f"Error saving latency results: {e}")
    else:
        print("\nNo latency results were generated.")


    print("\n--- Accuracy & Speed Comparison Run Finished ---")


if __name__ == "__main__":

    config = types.SimpleNamespace()
    config.model_name = "bert-base-uncased"
    config.tasks = ["sst2", "mrpc", "rte", "cola", "qnli", "qqp", "mnli"]
    config.batch_size = 16
    config.seq_length = 128
    config.output_dir = f"./results_accuracy_speed_{config.model_name.split('/')[-1]}"
    config.num_train_steps = 200
    config.learning_rate = 2e-5
    config.eval_every = 40
    config.num_runs = 3
    config.random_seed_base = 42
    config.latency_repeats = 10
    config.latency_warmup = 3
    config.N_dense_rows_to_keep = 50
    config.layers_to_modify = None
    config.dropbp_initial_drop_rate = 0.15

    DO_COMPARISON_RUN = True

    if DO_COMPARISON_RUN:
        print("\n === Starting Accuracy & Speed Comparison Run (Original vs Sparse vs DropBP) ===\n")
        print("="*60)
        print(f" >> Comparing: Original (Blue), Sparse (Green), DropBP (Red)")
        print(f" >> Model: {config.model_name}")
        print(f" >> Tasks: {config.tasks}")
        print(f" >> Steps: {config.num_train_steps}, Eval Freq: {config.eval_every}")
        print(f" >> Number of Runs (Accuracy): {config.num_runs}")
        runtime_multiplier = config.num_runs * (1 + (1 if DROPBP_AVAILABLE and SPARSE_METHOD_TRITON_AVAILABLE else 0))
        print("="*60)
        time.sleep(5)
        clear_gpu_cache()
        try: run_accuracy_comparison(config)
        except Exception as e: print(f"\nAn unexpected error occurred during the run: {e}"); traceback.print_exc()
        finally: clear_gpu_cache(); print("\n === Run Finished ===\n")
    else: print("DO_COMPARISON_RUN is False. Set to True to run.")