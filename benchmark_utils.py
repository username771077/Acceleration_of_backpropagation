import torch
import torch.nn as nn
import gc
import numpy as np
import traceback
import time
from datasets import load_dataset
from transformers import AutoTokenizer

def get_gpu_memory_usage():
    """Returns current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def clear_gpu_cache():
    """Clears GPU cache."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

def measure_latency(model: nn.Module, input_batch: dict, num_repeats: int = 10, num_warmup: int = 3, is_regression: bool = False):
    """Measures forward and backward latency for a given model and batch."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency measurement.")

    model.eval() 
    input_batch_gpu = {k: v.cuda() for k, v in input_batch.items() if isinstance(v, torch.Tensor)}

    forward_times = []
    backward_times = []
    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    for _ in range(num_warmup):
        try:
            # Forward pass
            with torch.no_grad():
                _ = model(**input_batch_gpu)

            # Backward pass (requires grad)
            model.train() 
            outputs = model(**input_batch_gpu)
            logits = outputs.logits
            labels = input_batch_gpu['labels']
            if is_regression:
                 loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                 loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
            loss.backward()
            model.zero_grad(set_to_none=True) 
            model.eval()
        except Exception as e:
            print(f"Warmup Exception: {e}")
            traceback.print_exc()
            return float('nan'), float('nan')

    torch.cuda.synchronize()

    for i in range(num_repeats):
        try:
            # --- Forward Pass Measurement ---
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()
            with torch.no_grad():
                 outputs = model(**input_batch_gpu)
            fwd_end.record()
            torch.cuda.synchronize() 
            forward_times.append(fwd_start.elapsed_time(fwd_end))

            # --- Backward Pass Preparation ---
            model.train() 
            outputs = model(**input_batch_gpu) 
            logits = outputs.logits
            labels = input_batch_gpu['labels']
            if is_regression:
                 loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                 loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
            torch.cuda.synchronize() 

            # --- Backward Pass Measurement ---
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start.record()
            loss.backward()
            bwd_end.record()
            torch.cuda.synchronize() 
            backward_times.append(bwd_start.elapsed_time(bwd_end))

            model.zero_grad(set_to_none=True) 
            model.eval() 

        except Exception as e:
            print(f"Measurement Exception (Repeat {i+1}/{num_repeats}): {e}")
            traceback.print_exc()
            forward_times.append(float('nan'))
            backward_times.append(float('nan'))
            model.zero_grad(set_to_none=True) 
            model.eval() 


    avg_forward = np.nanmean(forward_times) if forward_times else 0.0
    avg_backward = np.nanmean(backward_times) if backward_times else 0.0
    return avg_forward, avg_backward

def prepare_data(task_name: str, tokenizer: AutoTokenizer, batch_size: int, seq_length: int, num_batches: int):
    """Loads, tokenizes, and prepares batches for a given GLUE task."""
    print(f"Loading dataset: {task_name}...")
    try:
        dataset = load_dataset("glue", task_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load dataset {task_name}: {e}")
        return None

    task_to_keys = {
        "cola": ("sentence", None), "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"), "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"), "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None), "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        "ax": ("premise", "hypothesis"),
    }
    sentence1_key, sentence2_key = task_to_keys.get(task_name, ("sentence1", "sentence2")) # Default guess

    split = "validation_matched" if task_name == "mnli" else ("test" if task_name == "ax" else "validation")
    if task_name == "mnli" and split not in dataset:
        split = "validation_mismatched"
        print(f"Using {split} split for mnli.")
    if split not in dataset:
        split = "train" 
        print(f"Warning: Using {split} split for task {task_name}.")
    if split not in dataset:
        print(f"Error: Could not find a suitable split ('validation', 'test', or 'train') for task {task_name}.")
        return None

    print(f"Tokenizing {task_name} (using split: {split})...")
    ds_split = dataset[split]

    if sentence1_key not in ds_split.column_names or \
       (sentence2_key is not None and sentence2_key not in ds_split.column_names):
        print(f"Error: Keys '{sentence1_key}'/'{sentence2_key}' not found in {split} columns: {ds_split.column_names}.")
        if sentence2_key is None and 'text' in ds_split.column_names:
            print("Attempting to use 'text' as sentence key.")
            sentence1_key = 'text'
        elif sentence2_key is None and 'question' in ds_split.column_names:
             print("Attempting to use 'question' as sentence key.")
             sentence1_key = 'question'
        else:
             return None 

    is_stsb = (task_name == "stsb") 

    def tokenize_function(examples):
        texts = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        processed_texts = []
        for text_or_list in texts:
            if isinstance(text_or_list, list):
                 processed_texts.append(list(map(lambda x: "" if x is None else str(x), text_or_list)))
            else:
                 processed_texts.append("" if text_or_list is None else str(text_or_list))

        return tokenizer(*processed_texts, padding="max_length", truncation=True, max_length=seq_length)

    try:
        cols_to_keep_base = ['input_ids', 'attention_mask']
        if 'bert' in tokenizer.name_or_path.lower():
            cols_to_keep_base.append('token_type_ids')
        cols_to_keep = cols_to_keep_base + ['label', 'labels', 'idx'] # Keep potential label cols + index
        cols_to_remove = [c for c in ds_split.column_names if c not in cols_to_keep]

        tokenized_ds = ds_split.map(tokenize_function, batched=True, remove_columns=cols_to_remove)
    except Exception as e:
        print(f"Error during tokenization for task {task_name}: {e}")
        traceback.print_exc()
        return None


    label_column = 'label'
    final_label_column = 'labels'

    if label_column not in tokenized_ds.column_names:
         print(f"Warning: '{label_column}' column not found in tokenized data for {task_name}. Adding dummy labels.")
         num_rows = len(tokenized_ds)
         dummy_label = 0.0 if is_stsb else 0
         tokenized_ds = tokenized_ds.add_column(final_label_column, [dummy_label] * num_rows)
    elif is_stsb:
         print(f"Converting labels to float for regression task {task_name}.")
         tokenized_ds = tokenized_ds.map(lambda x: {final_label_column: float(x[label_column])})
         if label_column != final_label_column and label_column in tokenized_ds.column_names:
             tokenized_ds = tokenized_ds.remove_columns([label_column])
    elif label_column != final_label_column:
         tokenized_ds = tokenized_ds.rename_column(label_column, final_label_column)

    required_model_cols = ['input_ids', 'attention_mask']
    model_name_lower = tokenizer.name_or_path.lower()
    if 'bert' in model_name_lower and 'roberta' not in model_name_lower: # Add token_type_ids for BERT-like models
        required_model_cols.append('token_type_ids')

    final_columns = [c for c in required_model_cols + [final_label_column] if c in tokenized_ds.column_names]
    tokenized_ds.set_format("torch", columns=final_columns)

    dataloader = torch.utils.data.DataLoader(tokenized_ds, batch_size=batch_size, drop_last=True) # drop_last for consistent batch sizes
    batches = []
    print(f"Preparing batches (Batch Size: {batch_size}, Num Batches: {num_batches})...")
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        if 'bert' in model_name_lower and 'roberta' not in model_name_lower and 'token_type_ids' not in batch:
            batch['token_type_ids'] = torch.zeros_like(batch['input_ids'])
        elif 'roberta' in model_name_lower and 'token_type_ids' in batch:
             del batch['token_type_ids']

        batches.append(batch)

    if not batches:
        print(f"Error: No batches were prepared for task {task_name}. Check data or batch size.")
        return None

    print(f"Data prepared successfully for {task_name} ({len(batches)} batches).")
    return batches


print("Benchmarking helper functions defined.")
print("-" * 40)