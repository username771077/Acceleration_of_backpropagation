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
    AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, PreTrainedModel
)
from datasets import load_dataset
import time
import pandas as pd
import numpy as np
import types
import torch.profiler
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from itertools import chain

print("Imports complete.")
print("-" * 40)


print("Defining PyTorch-based sparse functions and analysis logic...")


CURRENT_TASK_SPARSITY_RESULTS = defaultdict(lambda: {'element': [], 'row': []})
CURRENT_GRAD_OUTPUT_SAMPLE = None
CURRENT_PLOT_LAYER_KEY = None
SPARSITY_THRESHOLD = 1e-6


def check_tensors_gpu_ready(*tensors):
    for t in tensors: assert t is not None and t.is_cuda

def sparse_row_matmul_pytorch(
    dY: torch.Tensor, W: torch.Tensor, threshold: float = 1e-6, layer_idx: str = None
):
    check_tensors_gpu_ready(dY, W)
    dY_shape = dY.shape; dtype = dY.dtype; device = dY.device
    if dY.ndim > 2: M_total = dY.numel() // dY_shape[-1]; K = dY_shape[-1]; dY_2d = dY.reshape(M_total, K)
    elif dY.ndim == 2: M_total, K = dY.shape; dY_2d = dY
    else: raise ValueError(f"dY must have at least 2 dimensions, got {dY.ndim}")
    K_W, N = W.shape; assert K == K_W
    M = M_total
    dX = torch.zeros((M, N), device=device, dtype=dtype); output_shape_final = dY_shape[:-1] + (N,)
    row_abs_sums = torch.sum(torch.abs(dY_2d), dim=1)
    dense_row_indices = torch.nonzero(row_abs_sums > threshold).squeeze(1)
    num_dense_rows = dense_row_indices.shape[0]
    if num_dense_rows > 0:
        dY_compacted = dY_2d[dense_row_indices]
        W_cont = W.contiguous()
        dX_compacted = dY_compacted @ W_cont
        dX[dense_row_indices] = dX_compacted
    try: return dX.view(output_shape_final)
    except RuntimeError: return dX.reshape(output_shape_final)


def reset_current_task_sparsity_results():
    global CURRENT_TASK_SPARSITY_RESULTS; CURRENT_TASK_SPARSITY_RESULTS.clear()


def reset_current_grad_sample():
    global CURRENT_GRAD_OUTPUT_SAMPLE, CURRENT_PLOT_LAYER_KEY
    CURRENT_GRAD_OUTPUT_SAMPLE = None
    CURRENT_PLOT_LAYER_KEY = None

def analyze_sparsity(tensor: torch.Tensor, threshold: float):
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0: return 0.0, 0.0
    total_elements = tensor.numel()
    zero_elements = torch.sum(torch.isclose(tensor, torch.zeros_like(tensor), atol=threshold)).item()
    element_sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
    if tensor.ndim > 1:
        rows_tensor = tensor.reshape(-1, tensor.shape[-1])
        total_rows = rows_tensor.shape[0]
        zero_rows = torch.sum(torch.all(torch.isclose(rows_tensor, torch.zeros_like(rows_tensor), atol=threshold), dim=1)).item()
        row_sparsity = zero_rows / total_rows if total_rows > 0 else 0.0
    elif total_elements > 0: row_sparsity = element_sparsity
    else: row_sparsity = 0.0
    return element_sparsity, row_sparsity


class LinearFunctionAccelarate(torch.autograd.Function):
    _plot_target_layer_idx = -1

    @staticmethod
    def forward(ctx, input, weight, bias, layer_idx, layer_type):
        output = input @ weight.T
        if bias is not None: output += bias
        ctx.save_for_backward(input, weight, bias)
        ctx.layer_idx = layer_idx; ctx.layer_type_str = 'intermediate' if layer_type == 0 else 'output'
        return output

    @staticmethod
    def backward(ctx, grad_output):
        global CURRENT_GRAD_OUTPUT_SAMPLE, CURRENT_PLOT_LAYER_KEY, CURRENT_TASK_SPARSITY_RESULTS
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx; layer_type_str = ctx.layer_type_str
        grad_input = grad_weight = grad_bias = None; layer_key = f"{layer_idx}-{layer_type_str}"

        if grad_output is not None:
            elem_sparsity, row_sparsity = analyze_sparsity(grad_output, SPARSITY_THRESHOLD)
            try:
                CURRENT_TASK_SPARSITY_RESULTS[layer_key]['element'].append(elem_sparsity)
                CURRENT_TASK_SPARSITY_RESULTS[layer_key]['row'].append(row_sparsity)
            except Exception as e: print(f"Warning: Could not store sparsity: {e}")

            if layer_idx == LinearFunctionAccelarate._plot_target_layer_idx:
                CURRENT_GRAD_OUTPUT_SAMPLE = grad_output.detach().cpu().clone()
                CURRENT_PLOT_LAYER_KEY = layer_key

        if bias is not None and ctx.needs_input_grad[2]: grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        if ctx.needs_input_grad[1]:
            input_2d = input.reshape(-1, input.shape[-1]); grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_weight = grad_output_2d.T @ input_2d
        if ctx.needs_input_grad[0]:
            grad_input = sparse_row_matmul_pytorch(grad_output, weight, threshold=SPARSITY_THRESHOLD, layer_idx=layer_key)

        return grad_input, grad_weight, grad_bias, None, None


class LinearIntermediateAccelarate(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, bias=True, device=None, dtype=None):
        super().__init__(); factory_kwargs={'device': device, 'dtype': dtype}
        self.in_features=in_features; self.out_features=out_features; self.layer_idx = layer_idx
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias: self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else: self.register_parameter('bias', None)
        self._init_weights()
    def _init_weights(self): nn.init.kaiming_uniform_(self.weight, a=5**0.5);
    def from_linear(self, linear: nn.Linear):
        if not isinstance(linear, nn.Linear): raise TypeError("Input 'linear' must be nn.Linear")
        if self.weight.shape != linear.weight.shape: raise ValueError(f"Weight shape mismatch: expected {self.weight.shape}, got {linear.weight.shape}")
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            if self.bias is None: self.bias = Parameter(torch.empty(self.out_features, device=linear.bias.device, dtype=linear.bias.dtype))
            self.bias = nn.Parameter(linear.bias.data.clone())
        elif self.bias is not None: self.register_parameter('bias', None)
        return self
    def forward(self, x): return LinearFunctionAccelarate.apply(x, self.weight, self.bias, self.layer_idx, 0)

class LinearOutputAccelarate(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, bias=True, device=None, dtype=None):
        super().__init__(); factory_kwargs={'device': device, 'dtype': dtype}
        self.in_features=in_features; self.out_features=out_features; self.layer_idx = layer_idx
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias: self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else: self.register_parameter('bias', None)
        self._init_weights()
    def _init_weights(self): nn.init.kaiming_uniform_(self.weight, a=5**0.5);
    def from_linear(self, linear: nn.Linear):
        if not isinstance(linear, nn.Linear): raise TypeError("Input 'linear' must be nn.Linear")
        if self.weight.shape != linear.weight.shape: raise ValueError(f"Weight shape mismatch: expected {self.weight.shape}, got {linear.weight.shape}")
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            if self.bias is None: self.bias = Parameter(torch.empty(self.out_features, device=linear.bias.device, dtype=linear.bias.dtype))
            self.bias = nn.Parameter(linear.bias.data.clone())
        elif self.bias is not None: self.register_parameter('bias', None)
        return self
    def forward(self, x): return LinearFunctionAccelarate.apply(x, self.weight, self.bias, self.layer_idx, 1)


def replace_bert_layers(model: PreTrainedModel, layers_to_modify: list = None):
    action = "Replacing ALL" if layers_to_modify is None else f"Replacing layers {layers_to_modify} in"
    print(f"! {action} BERT FFN layers !");
    device = next(model.parameters()).device; dtype = next(model.parameters()).dtype
    encoder = None
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"): encoder = model.bert.encoder
    elif hasattr(model, "encoder"): encoder = model.encoder
    else: print("Warning: Could not find standard BERT encoder structure."); return model
    layers_replaced_count = 0
    if not hasattr(encoder, "layer"): print("Warning: Encoder does not have 'layer' attribute."); return model
    modify_set = set(layers_to_modify) if layers_to_modify is not None else None
    for i, layer in enumerate(encoder.layer):
        should_modify = (modify_set is None or i in modify_set)
        intermediate_module = getattr(layer, "intermediate", None)
        if intermediate_module:
            dense_layer = getattr(intermediate_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; h, t = orig.weight.shape
                    new = LinearIntermediateAccelarate(t, h, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.intermediate.dense = new; layers_replaced_count +=1
        output_module = getattr(layer, "output", None)
        if output_module:
            dense_layer = getattr(output_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; t_out, h_out = orig.weight.shape
                    new = LinearOutputAccelarate(h_out, t_out, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.output.dense = new; layers_replaced_count += 1
    if layers_replaced_count > 0: print(f"! Layer replacement complete ({layers_replaced_count} Linear layers modified) !")
    elif layers_to_modify is not None: print(f"! No layers were replaced for specified indices {layers_to_modify}. Check indices/structure. !")
    else: print("! No layers were replaced. Check model structure or replacement logic. !")
    return model

def replace_roberta_layers(model: PreTrainedModel, layers_to_modify: list = None):
    action = "Replacing ALL" if layers_to_modify is None else f"Replacing layers {layers_to_modify} in"
    print(f"! {action} RoBERTa FFN layers !");
    device = next(model.parameters()).device; dtype = next(model.parameters()).dtype
    encoder = None
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"): encoder = model.roberta.encoder
    elif hasattr(model, "encoder"): encoder = model.encoder
    else: print("Warning: Could not find standard RoBERTa encoder structure."); return model
    layers_replaced_count = 0
    if not hasattr(encoder, "layer"): print("Warning: Encoder does not have 'layer' attribute."); return model
    modify_set = set(layers_to_modify) if layers_to_modify is not None else None
    for i, layer in enumerate(encoder.layer):
        should_modify = (modify_set is None or i in modify_set)
        intermediate_module = getattr(layer, "intermediate", None)
        if intermediate_module:
            dense_layer = getattr(intermediate_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; h, t = orig.weight.shape
                    new = LinearIntermediateAccelarate(t, h, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.intermediate.dense = new; layers_replaced_count += 1
        output_module = getattr(layer, "output", None)
        if output_module:
            dense_layer = getattr(output_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; t_out, h_out = orig.weight.shape
                    new = LinearOutputAccelarate(h_out, t_out, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.output.dense = new; layers_replaced_count += 1
    if layers_replaced_count > 0: print(f"! Layer replacement complete ({layers_replaced_count} Linear layers modified) !")
    elif layers_to_modify is not None: print(f"! No layers were replaced for specified indices {layers_to_modify}. Check indices/structure. !")
    else: print("! No layers were replaced. Check model structure or replacement logic. !")
    return model

print("Custom definitions complete.")
print("-" * 40)


print("Defining data and plotting helper functions...")

def get_gpu_memory_usage():
    if torch.cuda.is_available(): return torch.cuda.memory_allocated() / (1024**3)
    return 0

def clear_gpu_cache():
    if torch.cuda.is_available(): gc.collect(); torch.cuda.empty_cache()

def prepare_data(task_name: str, tokenizer: AutoTokenizer, batch_size: int, seq_length: int, num_batches: int):
    print(f"Loading dataset: {task_name}...")
    is_wikitext = task_name.lower().startswith('wikitext')
    is_stsb = (task_name == "stsb")

    if is_wikitext:
        try:
            raw_datasets = load_dataset('wikitext', 'wikitext-103-v1', trust_remote_code=True)
            split = 'validation' if 'validation' in raw_datasets else 'test'
            if split not in raw_datasets: split = 'train'
            column_name = "text"
            dataset = raw_datasets[split]
            dataset = dataset.filter(lambda x: len(x[column_name].strip()) > 0)
        except Exception as e: print(f"Failed load/process wikitext: {e}"); return None

        print(f"Tokenizing {task_name} (split: {split})...")
        def tokenize_function(examples):
            return tokenizer(examples[column_name], truncation=False)

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=[column_name])

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length < seq_length: return {'input_ids': [], 'labels': [], 'attention_mask': []}
            total_length = (total_length // seq_length) * seq_length
            result = {
                k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
                for k, t in concatenated_examples.items()
            }
            num_chunks = len(result["input_ids"])
            result["labels"] = [0] * num_chunks
            result["attention_mask"] = [torch.ones(seq_length, dtype=torch.long).tolist() for _ in range(num_chunks)]
            return result

        print(f"Grouping texts into chunks of {seq_length} for {task_name}...")
        lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000)
        lm_datasets = lm_datasets.filter(lambda example: len(example['input_ids']) > 0)

        req_cols = ['input_ids', 'attention_mask', 'labels']
        if 'bert' in tokenizer.name_or_path.lower() and 'token_type_ids' in lm_datasets.column_names:
             req_cols.append('token_type_ids')
        final_cols = [c for c in req_cols if c in lm_datasets.column_names]
        lm_datasets.set_format("torch", columns=final_cols)
        processed_dataset = lm_datasets

    else:
        try: dataset = load_dataset("glue", task_name, trust_remote_code=True)
        except Exception as e: print(f"Failed load GLUE task {task_name}: {e}"); return None
        task_to_keys = {"cola": ("sentence", None), "mnli": ("premise", "hypothesis"), "mrpc": ("sentence1", "sentence2"), "qnli": ("question", "sentence"), "qqp": ("question1", "question2"), "rte": ("sentence1", "sentence2"), "sst2": ("sentence", None), "stsb": ("sentence1", "sentence2"), "wnli": ("sentence1", "sentence2"), "ax": ("premise", "hypothesis"),}
        s1k, s2k = task_to_keys.get(task_name, ("sentence1", "sentence2"))
        split = "validation_matched" if task_name == "mnli" else ("test" if task_name == "ax" else "validation")
        if task_name == "mnli" and split not in dataset: split = "validation_mismatched"
        if split not in dataset: split = "train"
        if split not in dataset: print(f"Split not found for {task_name}."); return None
        print(f"Tokenizing {task_name} (split: {split})...")
        ds_split = dataset[split]
        if s1k not in ds_split.column_names or (s2k is not None and s2k not in ds_split.column_names): print(f"Keys '{s1k}'/'{s2k}' not found in {split} columns: {ds_split.column_names}."); return None

        def tok_fn(ex):
            args = (ex[s1k],) if s2k is None else (ex[s1k], ex[s2k])
            args = tuple(list(map(lambda x: "" if x is None else x, a)) if isinstance(a, list) else ("" if a is None else a) for a in args)
            return tokenizer(*args, padding="max_length", truncation=True, max_length=seq_length)
        try:
            cols_to_keep = ['input_ids', 'attention_mask', 'token_type_ids', 'label', 'labels', 'idx']
            cols_rem = [c for c in ds_split.column_names if c not in cols_to_keep]
            tok_ds = ds_split.map(tok_fn, batched=True, remove_columns=cols_rem)
        except Exception as e: print(f"Tokenize error: {e}"); traceback.print_exc(); return None
        label_col = 'label'; final_label_col = 'labels'
        if label_col not in tok_ds.column_names:
            print(f"Warning: '{label_col}' column not found. Adding dummy labels.")
            num_rows = len(tok_ds); dummy_label = 0.0 if is_stsb else 0
            tok_ds = tok_ds.add_column(final_label_col, [dummy_label] * num_rows)
        elif is_stsb:
            print("Converting STSB labels to float.")
            tok_ds = tok_ds.map(lambda x: {final_label_col: float(x[label_col])})
            if label_col != final_label_col and label_col in tok_ds.column_names: tok_ds = tok_ds.remove_columns([label_col])
        elif label_col != final_label_col: tok_ds = tok_ds.rename_column(label_col, final_label_col)
        req_cols = ['input_ids', 'attention_mask']; m_name = tokenizer.name_or_path.lower()
        if 'bert' in m_name: req_cols.append('token_type_ids')
        final_cols = [c for c in req_cols + [final_label_col] if c in tok_ds.column_names]
        tok_ds.set_format("torch", columns=final_cols)
        processed_dataset = tok_ds

    dl = torch.utils.data.DataLoader(processed_dataset, batch_size=batch_size, drop_last=True)
    batches = []
    print(f"Preparing batches (size {batch_size})...")
    for i, batch in enumerate(dl):
        if i >= num_batches: break
        m_name = tokenizer.name_or_path.lower()
        if 'bert' in m_name and 'token_type_ids' not in batch: batch['token_type_ids'] = torch.zeros_like(batch['input_ids'])
        elif 'roberta' in m_name and 'token_type_ids' in batch:
             if 'token_type_ids' in batch: del batch['token_type_ids']
        batches.append(batch)

    if not batches: print(f"No batches prepared for {task_name}."); return None
    print(f"Data prepared ({len(batches)} batches).")
    return batches


def summarize_sparsity(sparsity_results: dict):
    agg_data = defaultdict(lambda: {'element': [], 'row': []})
    if not sparsity_results: print("No sparsity data collected."); return pd.DataFrame(columns=["Layer", "Type", "Avg Element Sparsity (%)", "Avg Row Sparsity (%)"])
    for key, data in sparsity_results.items():
        try: idx_str, type_str = key.split('-'); idx = int(idx_str)
        except Exception as e: print(f"Warning: Could not parse sparsity key '{key}': {e}"); continue
        agg_data[(idx, type_str)]['element'].extend(data['element'])
        agg_data[(idx, type_str)]['row'].extend(data['row'])
    summary_list = []; sorted_keys = sorted(agg_data.keys())
    for key in sorted_keys:
        idx, type_str = key; layer_data = agg_data[key]
        el_sp = statistics.mean(layer_data['element']) * 100 if layer_data['element'] else 0.0
        row_sp = statistics.mean(layer_data['row']) * 100 if layer_data['row'] else 0.0
        summary_list.append({"Layer": idx, "Type": type_str, "Avg Element Sparsity (%)": f"{el_sp:.2f}", "Avg Row Sparsity (%)": f"{row_sp:.2f}"})
    df = pd.DataFrame(summary_list);
    if df.empty: df = pd.DataFrame(columns=["Layer", "Type", "Avg Element Sparsity (%)", "Avg Row Sparsity (%)"])
    return df


def summarize_sparsity_per_task(task_sparsity_results: dict):
    summary_rows = []
    metric_types = ['AVG', 'MAX', 'MIN']
    layer_types = ['intermediate', 'output']
    tasks = list(task_sparsity_results.keys())
    if not tasks: print("No per-task sparsity data collected."); return pd.DataFrame(columns=["Metric", "Layer Type"] + tasks)

    agg_metrics = {task: {m: {lt: 0.0 for lt in layer_types} for m in metric_types} for task in tasks}

    for task, layer_results in task_sparsity_results.items():
        intermediate_sparsities = []
        output_sparsities = []
        if not layer_results: print(f"Warning: No layer results found for task {task} in per-task summary."); continue

        for layer_key, data in layer_results.items():
            try:
                _ , type_str = layer_key.split('-')
                valid_rows = [x for x in data.get('row', []) if isinstance(x, (int, float))]
                if not valid_rows: continue
                avg_row_sparsity = statistics.mean(valid_rows)
                if type_str == 'intermediate': intermediate_sparsities.append(avg_row_sparsity)
                elif type_str == 'output': output_sparsities.append(avg_row_sparsity)
            except Exception as e: print(f"Warning: Error processing sparsity key '{layer_key}' for task '{task}': {e}")

        if intermediate_sparsities:
            agg_metrics[task]['AVG']['intermediate'] = statistics.mean(intermediate_sparsities)
            agg_metrics[task]['MAX']['intermediate'] = max(intermediate_sparsities)
            agg_metrics[task]['MIN']['intermediate'] = min(intermediate_sparsities)

        if output_sparsities:
            agg_metrics[task]['AVG']['output'] = statistics.mean(output_sparsities)
            agg_metrics[task]['MAX']['output'] = max(output_sparsities)
            agg_metrics[task]['MIN']['output'] = min(output_sparsities)

    header_row = ["Metric", "Layer Type"] + tasks
    data_rows = []
    for metric in metric_types:
        for lt in layer_types:
            row_values = []
            for task in tasks:
                val = agg_metrics[task][metric][lt]
                if isinstance(val, (int, float)):
                    row_values.append(f"{val:.3f}")
                else:
                    row_values.append('N/A')
            row = [metric, lt] + row_values
            data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=header_row)
    return df


def plot_gradient_heatmap(tensor_sample, layer_key, threshold, task_name="", model_name="", slice_dims=None):
    if tensor_sample is None: print(f"No gradient sample captured for plotting (Layer: {layer_key}, Task: {task_name})."); return
    if not plt: print("Matplotlib not imported, cannot plot."); return
    try:
        data = tensor_sample.cpu().numpy()
        plot_title = f"Gradient Output (dY) Sample\nModel: {model_name}, Task: {task_name}, Layer: {layer_key}"

        if data.ndim > 2: M, K = data.shape[0]*data.shape[1], data.shape[2]; data_2d = data.reshape(M, K); xlabel = "Hidden Dim"; ylabel = f"Batch*SeqPos ({data.shape[0]}x{data.shape[1]})"
        elif data.ndim == 2: data_2d = data; M, K = data_2d.shape; xlabel = "Feature Dim"; ylabel = "Row Index"
        else: print(f"Cannot plot tensor with {data.ndim} dimensions."); return

        if slice_dims is not None and len(slice_dims) == 2:
             rows, cols = slice_dims
             data_to_plot = data_2d[:min(rows, M), :min(cols, K)]
             plot_title += f" (Slice: {data_to_plot.shape[0]}x{data_to_plot.shape[1]})"
             ylabel = f"Row Index (Top {data_to_plot.shape[0]})"
             xlabel = f"Hidden Dim (First {data_to_plot.shape[1]})"
        else:
             data_to_plot = data_2d

        plt.figure(figsize=(10, 8)); cmap = 'jet'
        non_zeros = data_to_plot[np.abs(data_to_plot) > threshold]
        if non_zeros.size > 0: vmax = np.percentile(np.abs(non_zeros), 99) if len(non_zeros) > 1 else np.abs(non_zeros[0]); vmin = -vmax
        else: vmin, vmax = -threshold*10, threshold*10
        if vmin == vmax or vmax == 0: vmin = min(vmin, -0.001); vmax = max(vmax, 0.001)

        im = plt.imshow(data_to_plot, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Gradient Value"); plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(plot_title)
        plt.tight_layout();
        safe_layer_key = layer_key.replace('-', '_') if layer_key else "unknown"
        slice_suffix = f"_slice{slice_dims[0]}x{slice_dims[1]}" if slice_dims else ""
        plot_filename = f"./grad_heatmap_{model_name.replace('/','_')}_task_{task_name}_layer_{safe_layer_key}{slice_suffix}.png"
        plt.savefig(plot_filename); print(f"Saved gradient heatmap to {plot_filename}")
        plt.show(); plt.close()
    except Exception as e: print(f"Error during plotting: {e}"); traceback.print_exc()

def plot_sparsity_vs_layer(sparsity_df, model_name=""):
    if sparsity_df is None or sparsity_df.empty: print(f"Sparsity DataFrame for {model_name} is empty, skipping plot."); return
    if not plt: print("Matplotlib not imported, cannot plot."); return
    try:
        sparsity_df["Layer"] = pd.to_numeric(sparsity_df["Layer"], errors='coerce')
        sparsity_df["Avg Row Sparsity (%)"] = pd.to_numeric(sparsity_df["Avg Row Sparsity (%)"], errors='coerce')
        sparsity_df.dropna(subset=["Layer","Avg Row Sparsity (%)"], inplace=True)
        sparsity_df["Layer"] = sparsity_df["Layer"].astype(int)
        if sparsity_df.empty: print(f"No valid numeric sparsity data to plot for {model_name}."); return

        pivot_df = sparsity_df.pivot(index='Layer', columns='Type', values='Avg Row Sparsity (%)')
        max_layer = pivot_df.index.max();
        pivot_df = pivot_df.reindex(range(max_layer + 1))

        ax = pivot_df.plot(kind='bar', figsize=(14, 7), width=0.8)
        plt.title(f'Average Row Sparsity vs. Layer Index ({model_name})')
        plt.xlabel('Layer Index'); plt.ylabel('Average Row Sparsity (%)'); plt.xticks(range(max_layer + 1), rotation=0)
        plt.ylim(0, 105); plt.grid(axis='y', linestyle='--'); plt.legend(title='FFN Layer Type')
        plt.tight_layout(); plot_filename = f"./sparsity_vs_layer_{model_name.replace('/','_')}.png"
        plt.savefig(plot_filename); print(f"Saved sparsity vs layer plot to {plot_filename}")
        plt.show(); plt.close()
    except Exception as e: print(f"Error during sparsity plot generation for {model_name}: {e}"); traceback.print_exc()

def plot_layer_sparsity_histogram(sparsity_df, model_name=""):
    if sparsity_df is None or sparsity_df.empty: print(f"Sparsity DataFrame for {model_name} is empty, skipping histogram."); return
    if not plt: print("Matplotlib not imported, cannot plot."); return
    try:
        sparsity_values = pd.to_numeric(sparsity_df["Avg Row Sparsity (%)"], errors='coerce').dropna()
        if sparsity_values.empty: print(f"No valid numeric row sparsity data to plot histogram for {model_name}."); return

        plt.figure(figsize=(10, 6))
        plt.hist(sparsity_values, bins=20, range=(0, 100))
        plt.xlabel('Average Row Sparsity (%)')
        plt.ylabel('Number of Layer Types (Intermediate/Output)')
        plt.title(f'Histogram of Average Row Sparsity Across Layers ({model_name})')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plot_filename = f"./layer_sparsity_histogram_{model_name.replace('/','_')}.png"
        plt.savefig(plot_filename); print(f"Saved layer sparsity histogram to {plot_filename}")
        plt.show(); plt.close()

    except Exception as e: print(f"Error during sparsity histogram generation for {model_name}: {e}"); traceback.print_exc()


print("Data and plotting functions defined.")
print("-" * 40)


def profile_original_model_backward(config, task):
    if not torch.cuda.is_available(): print("CUDA required for profiling."); return
    device = torch.device("cuda")
    is_wikitext = task.lower().startswith('wikitext')
    current_seq_length = 512 if is_wikitext else config.seq_length
    print(f"\n--- Profiling Original Model Backward for Task: {task} ---")
    print(f"Model: {config.model_name}, BS: {config.batch_size}, SL: {current_seq_length}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
        is_regression = (task == 'stsb')
        task_labels_map = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2}
        num_labels = 1 if is_regression else (2 if is_wikitext else task_labels_map.get(task, 2))
        model_config.num_labels = num_labels
        if getattr(model_config, "id2label", None) is None: model_config.id2label = {i: f"LABEL_{i}" for i in range(model_config.num_labels)}; model_config.label2id = {v: k for k, v in model_config.id2label.items()}
    except Exception as e: print(f"Load Error: {e}"); traceback.print_exc(); return
    profile_batches = prepare_data(task, tokenizer, config.batch_size, current_seq_length, num_batches=1)
    if not profile_batches: print(f"Cannot prepare data for profiling task {task}."); return
    profile_batch = {k: v.cuda() for k, v in profile_batches[0].items() if isinstance(v, torch.Tensor)}
    model = None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            config=model_config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        ).to(device); model.train()
        outputs = model(**profile_batch); logits = outputs.logits; labels = profile_batch['labels']
        loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
        if is_regression:
             if logits.shape[-1] != 1: logits = logits[:, 0]
             loss = loss_fn(logits.squeeze(-1), labels.float())
        else: loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        print("Starting profiler...");
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof: loss.backward()
        print("Profiling complete.")
        print("\nProfiler Results (Top 20 GPU Operators, Grouped by Input Shape):"); print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
        trace_file = f"./{config.model_name.replace('/', '_')}_{task}_sl{current_seq_length}_profile.json"; prof.export_chrome_trace(trace_file)
        print(f"\nDetailed trace saved to: {trace_file}"); print("You can upload this file to chrome://tracing or https://ui.perfetto.dev/ for visualization.")
    except Exception as e: print(f"Error during profiling: {e}"); traceback.print_exc()
    finally: del model; del profile_batch; clear_gpu_cache()

print("Profiler function defined.")
print("-" * 40)


def run_analysis(config):
    if not torch.cuda.is_available(): print("CUDA required."); return None, None, None
    device = torch.device("cuda"); print(f"\n=== Running Analysis for {config.model_name} ===")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Tasks: {config.tasks}")
    print(f"Default GLUE SL: {config.seq_length}")
    print(f"BS: {config.batch_size}, Num Batches Per Task: {config.num_batches}")
    print(f"Sparsity Thr: {SPARSITY_THRESHOLD}")
    LinearFunctionAccelarate._plot_target_layer_idx = config.plot_layer_idx
    if config.layers_to_modify is None: print("Analyzing ALL modified layers.")
    else: print(f"Analyzing modified layers: {config.layers_to_modify}")

    try: tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except Exception as e: print(f"Load Tokenizer Error: {e}"); traceback.print_exc(); return None, None, None

    task_sparsity_results = {task: defaultdict(lambda: {'element': [], 'row': []}) for task in config.tasks}
    task_grad_samples = {}

    for task in config.tasks:
        print(f"\n--- Task: {task} ---")
        is_regression = (task == 'stsb')
        is_wikitext = task.lower().startswith('wikitext')
        current_seq_length = 512 if is_wikitext else config.seq_length
        print(f"Using Sequence Length: {current_seq_length}")
        reset_current_grad_sample()

        try:
            model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            task_labels_map = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2}
            num_labels = 1 if is_regression else (2 if is_wikitext else task_labels_map.get(task, 2))
            model_config.num_labels = num_labels
            if is_regression: print(f"Configuring model for regression (num_labels=1) for task {task}.")
            elif is_wikitext: print(f"Configuring model with dummy classification head (num_labels=2) for task {task}.")
            else: print(f"Setting num_labels={num_labels} for task {task}.")
            if getattr(model_config, "id2label", None) is None: model_config.id2label = {i: f"LABEL_{i}" for i in range(model_config.num_labels)}; model_config.label2id = {v: k for k, v in model_config.id2label.items()}
        except Exception as e: print(f"Load Config Error for task {task}: {e}"); traceback.print_exc(); continue

        task_batches = prepare_data(task, tokenizer, config.batch_size, current_seq_length, config.num_batches)
        if not task_batches: print(f"Skipping task {task} due to data prep failure."); continue
        run_batch = task_batches[0]

        print("\nRunning modified model for analysis (Forward/Backward)..."); clear_gpu_cache();
        reset_current_task_sparsity_results()
        mod_base = None; mod = None;
        try:
            mod_base = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            is_bert = mod_base.config.model_type.lower() == 'bert'; is_roberta = mod_base.config.model_type.lower() == 'roberta'
            replace_func = replace_bert_layers if (is_bert or not is_roberta) else replace_roberta_layers
            print(f"Applying replacements using {replace_func.__name__}...");
            mod = replace_func(mod_base, layers_to_modify=config.layers_to_modify);
            mod = mod.to(device)
            mod.train()

            input_batch_gpu = {k: v.cuda() for k, v in run_batch.items() if isinstance(v, torch.Tensor)}
            outputs = mod(**input_batch_gpu)
            logits = outputs.logits
            labels = input_batch_gpu['labels']
            loss_fn = nn.MSELoss() if is_regression or is_wikitext else nn.CrossEntropyLoss()

            if is_regression:
                 if logits.shape[-1] != 1: logits = logits[:, 0]
                 loss = loss_fn(logits.squeeze(-1), labels.float())
            else: loss = loss_fn(logits.view(-1, mod.config.num_labels), labels.view(-1))

            loss.backward()
            mod.zero_grad(set_to_none=True)
            print("Backward pass complete. Sparsity/Sample collected.")

            task_sparsity_results[task] = CURRENT_TASK_SPARSITY_RESULTS.copy()
            if not task_sparsity_results[task]:
                 print(f"Warning: No sparsity data collected for task {task}. Check if modified layers ran.")

            if CURRENT_GRAD_OUTPUT_SAMPLE is not None:
                task_grad_samples[task] = (CURRENT_GRAD_OUTPUT_SAMPLE, CURRENT_PLOT_LAYER_KEY)
                print(f"--- Stored gradient sample for task {task} from layer {CURRENT_PLOT_LAYER_KEY} ---")

        except Exception as e: print(f"Analysis Error: {e}"); traceback.print_exc()
        finally: del mod; del mod_base; clear_gpu_cache()


    all_sparsity_data = defaultdict(lambda: {'element': [], 'row': []})
    for task, task_data in task_sparsity_results.items():
        for layer_key, layer_data in task_data.items():
             all_sparsity_data[layer_key]['element'].extend(layer_data['element'])
             all_sparsity_data[layer_key]['row'].extend(layer_data['row'])

    sparsity_df_agg = summarize_sparsity(all_sparsity_data)

    print("\n--- Per-Task Sparsity Summary (Avg/Max/Min Row Sparsity per Layer Type) ---");
    sparsity_per_task_df = summarize_sparsity_per_task(task_sparsity_results); print(sparsity_per_task_df.to_markdown(index=False))


    if config.output_dir:
        try:
            os.makedirs(config.output_dir, exist_ok=True); safe_name = config.model_name.replace('/', '_')
            layer_suffix = "_all" if config.layers_to_modify is None else f"_layers_{'_'.join(map(str, config.layers_to_modify))}"
            sp_agg_path = os.path.join(config.output_dir, f"{safe_name}{layer_suffix}_agg_sparsity.csv");
            sp_task_path = os.path.join(config.output_dir, f"{safe_name}{layer_suffix}_per_task_sparsity.csv");

            if not sparsity_df_agg.empty: sparsity_df_agg.to_csv(sp_agg_path, index=False); print(f"\nAggregated sparsity results saved to {sp_agg_path}")
            if not sparsity_per_task_df.empty: sparsity_per_task_df.to_csv(sp_task_path, index=False); print(f"Per-task sparsity results saved to {sp_task_path}")
        except Exception as e: print(f"Save Error: {e}")

    return sparsity_df_agg, task_grad_samples, task_sparsity_results


print("Main analysis function defined.")
print("-" * 40)


def run_and_plot(config):
    print(f"\n\n{'='*20} Starting Run for Model: {config.model_name} {'='*20}")

    sparsity_df_agg, task_grad_samples, task_sparsity_results = None, {}, {}
    try:
        sparsity_df_agg, task_grad_samples, task_sparsity_results = run_analysis(config)
    except Exception as e:
         print(f"An unexpected error occurred during analysis execution for {config.model_name}: {e}")
         traceback.print_exc()
         if sparsity_df_agg is None: sparsity_df_agg = pd.DataFrame()

    print(f"\n === Generating Summary Plots for {config.model_name} ===\n")

    print(f"\n--- Generating Heatmap Plots for Layer {config.plot_layer_idx} (if captured) ---")
    if task_grad_samples:
        for task, sample_info in task_grad_samples.items():
            plot_gradient_heatmap(
                sample_info[0], sample_info[1], SPARSITY_THRESHOLD,
                task_name=task, model_name=config.model_name,
                slice_dims=config.heatmap_slice
            )
    else:
         print(f"Could not generate heatmaps (no samples captured for layer {config.plot_layer_idx}).")

    if sparsity_df_agg is not None and not sparsity_df_agg.empty:
        plot_sparsity_vs_layer(sparsity_df_agg, config.model_name)
    else:
        print(f"Aggregated Sparsity DataFrame not available for {config.model_name}, skipping sparsity plot.")

    if sparsity_df_agg is not None and not sparsity_df_agg.empty:
        plot_layer_sparsity_histogram(sparsity_df_agg, config.model_name)
    else:
        print(f"Aggregated Sparsity DataFrame not available for {config.model_name}, skipping sparsity histogram.")


    print(f"\n === Plot Generation Finished for {config.model_name} ===\n")


if __name__ == "__main__":

    base_config = {
        "tasks": ["sst2", "mrpc", "rte", "mnli", "wikitext", "cola", "wnli", "stsb", "qqp", "qnli"],
        "batch_size": 16,
        "seq_length": 128,
        "repeats": 5,
        "warmup": 1,
        "num_batches": 1, 
        "layers_to_modify": None,
        "heatmap_slice": (50, 50),
        "plot_layer_idx": 0
    }

    # --- Run for BERT ---
    config_bert = types.SimpleNamespace(**base_config)
    config_bert.model_name = "bert-base-uncased"
    config_bert.output_dir = "./results_bert_plots_final"
    config_bert.plot_layer_idx = 0
    run_and_plot(config_bert)

    # --- Run for RoBERTa ---
    config_roberta = types.SimpleNamespace(**base_config)
    config_roberta.model_name = "roberta-base"
    config_roberta.output_dir = "./results_roberta_plots_final"
    config_roberta.plot_layer_idx = 0
    run_and_plot(config_roberta)

    print("\n === All Plotting Runs Finished ===\n")