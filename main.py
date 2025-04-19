import torch
import torch.nn as nn
import os
import types
import time
import traceback
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from sparse_method_triton import SparseMethodLinearFunction 
from model_utils import replace_bert_layers, replace_roberta_layers
from benchmark_utils import measure_latency, prepare_data, clear_gpu_cache, get_gpu_memory_usage
from profiler import profile_original_model_backward

print("Imports complete.")
print("-" * 40)

def run_benchmark(config):
    """Main benchmarking function, takes config object including layers_to_modify."""
    if not torch.cuda.is_available():
        print("CUDA is required to run the benchmark.")
        return
    device = torch.device("cuda")
    print(f"Using Device: {torch.cuda.get_device_name(device)}")
    print(f"Benchmarking Model: {config.model_name}")
    print(f"Tasks: {config.tasks}")
    print(f"Batch Size: {config.batch_size}, Sequence Length: {config.seq_length}")
    print(f"Measurement Repeats: {config.repeats}, Warmup Runs: {config.warmup}")
    print(f"Number of Batches per Task: {config.num_batches}")

    SparseMethodLinearFunction._sparse_rows_to_keep = config.sparse_rows_to_keep
    print(f"*** Sparse Method (Triton): Set to keep top {config.sparse_rows_to_keep} rows for backward pass. ***")

    if config.layers_to_modify is None:
        print("Applying modification to ALL encoder FFN layers.")
    else:
        print(f"Applying modification to encoder FFN layers: {config.layers_to_modify}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except Exception as e:
        print(f"Fatal Error: Could not load tokenizer '{config.model_name}'. Benchmark aborted. Error: {e}")
        traceback.print_exc()
        return

    results = []

    for task in config.tasks:
        print(f"\n{'='*10} Task: {task} {'='*10}")
        is_regression = (task == 'stsb')

        try:
            model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)

            if is_regression:
                model_config.num_labels = 1
                print(f"Configuring model for regression (num_labels=1) for task {task}.")
            else:
                task_labels = {
                    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
                    "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2, "ax": 3
                }
                num_labels = task_labels.get(task, 2)
                if getattr(model_config, "num_labels", None) != num_labels:
                    print(f"Setting num_labels={num_labels} for task {task}.")
                    model_config.num_labels = num_labels

                if getattr(model_config, "id2label", None) is None or len(model_config.id2label) != model_config.num_labels:
                     model_config.id2label = {i: f"LABEL_{i}" for i in range(model_config.num_labels)}
                     model_config.label2id = {v: k for k, v in model_config.id2label.items()}

        except Exception as e:
            print(f"Error loading model config for task {task}: {e}. Skipping task.")
            traceback.print_exc()
            continue

        task_batches = prepare_data(task, tokenizer, config.batch_size, config.seq_length, config.num_batches)
        if not task_batches:
            print(f"Skipping task {task} due to data preparation failure.")
            continue
        timing_batch = task_batches[0] 

        print("\nBenchmarking Original Model...")
        clear_gpu_cache()
        model_orig = None
        orig_fwd_ms, orig_bwd_ms = float('nan'), float('nan')
        try:
            model_orig = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True
            ).to(device)
            orig_fwd_ms, orig_bwd_ms = measure_latency(
                model_orig, timing_batch, config.repeats, config.warmup, is_regression=is_regression
            )
            print(f"Original - Forward: {orig_fwd_ms:.3f} ms, Backward: {orig_bwd_ms:.3f} ms")
        except Exception as e:
            print(f"Error benchmarking original model for task {task}: {e}")
            traceback.print_exc()
        finally:
            del model_orig
            clear_gpu_cache()

        print("\nBenchmarking Modified Model (Sparse Method - Triton)...")
        clear_gpu_cache()
        mod_base = None 
        mod = None      
        mod_fwd_ms, mod_bwd_ms = float('nan'), float('nan')
        try:
            mod_base = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True
            )

            is_bert = mod_base.config.model_type.lower() == 'bert'
            is_roberta = mod_base.config.model_type.lower() == 'roberta'
            if is_bert or not is_roberta: 
                 replace_func = replace_bert_layers
            else: 
                 replace_func = replace_roberta_layers
            print(f"Applying layer replacements using {replace_func.__name__}...")

            mod = replace_func(mod_base, layers_to_modify=config.layers_to_modify)
            mod = mod.to(device)

            mod_fwd_ms, mod_bwd_ms = measure_latency(
                mod, timing_batch, config.repeats, config.warmup, is_regression=is_regression
            )
            print(f"Modified - Forward: {mod_fwd_ms:.3f} ms, Backward: {mod_bwd_ms:.3f} ms")

        except Exception as e:
            print(f"Error benchmarking modified model for task {task}: {e}")
            traceback.print_exc()
        finally:
            del mod      
            del mod_base 
            clear_gpu_cache()

        fwd_speedup = (orig_fwd_ms / mod_fwd_ms) if not np.isnan(mod_fwd_ms) and mod_fwd_ms > 1e-9 and not np.isnan(orig_fwd_ms) else 0.0
        bwd_speedup = (orig_bwd_ms / mod_bwd_ms) if not np.isnan(mod_bwd_ms) and mod_bwd_ms > 1e-9 and not np.isnan(orig_bwd_ms) else 0.0

        results.append({
            "Task": task,
            "Original Forward (ms)": f"{orig_fwd_ms:.3f}" if not np.isnan(orig_fwd_ms) else "Error",
            "Modified Forward (ms)": f"{mod_fwd_ms:.3f}" if not np.isnan(mod_fwd_ms) else "Error",
            "Forward Speedup": f"{fwd_speedup:.2f}x" if fwd_speedup > 0 else "-",
            "Original Backward (ms)": f"{orig_bwd_ms:.3f}" if not np.isnan(orig_bwd_ms) else "Error",
            "Modified Backward (ms)": f"{mod_bwd_ms:.3f}" if not np.isnan(mod_bwd_ms) else "Error",
            "Backward Speedup": f"{bwd_speedup:.2f}x" if bwd_speedup > 0 else "-",
        })
        print(f"{'-'*10} Task {task} Finished {'-'*10}")

    print("\n--- Performance Comparison Summary ---")
    if not results:
         print("No results generated.")
         return

    perf_df = pd.DataFrame(results)
    print(perf_df.to_markdown(index=False))

    if config.output_dir:
        try:
            os.makedirs(config.output_dir, exist_ok=True)
            safe_model_name = config.model_name.replace('/', '_')
            layer_suffix = "_all_layers" if config.layers_to_modify is None else f"_layers_{'_'.join(map(str, config.layers_to_modify))}"
            perf_path = os.path.join(
                config.output_dir,
                f"{safe_model_name}{layer_suffix}_sparse_method_triton_N{config.sparse_rows_to_keep}_perf.csv"
            )
            perf_df.to_csv(perf_path, index=False)
            print(f"\nPerformance results saved to: {perf_path}")
        except Exception as e:
            print(f"\nError saving performance results: {e}")

print("Main benchmark function defined.")
print("-" * 40)


if __name__ == "__main__":

    config = types.SimpleNamespace()
    config.model_name = "bert-base-uncased"
    config.tasks = ["sst2", "mrpc", "rte", "cola", "stsb", "qnli", "qqp", "mnli"] # not configured for wikitext yet, coming in later patches, maybe
    config.batch_size = 16
    config.seq_length = 128
    config.repeats = 20     # measurement repetitions
    config.warmup = 5       # warmup runs, better not change
    config.num_batches = 10 # batches to prepare/use per task (reduce for quicker testing)
    config.output_dir = "./results_sparse_method_triton_N20"

    config.sparse_rows_to_keep = 20
    print(f"*** CONFIG: Setting sparse_rows_to_keep = {config.sparse_rows_to_keep} for Sparse Method ***")

    # <<< Ablation Study Configuration >>>
    # set to None to modify ALL layers when testing the sparse method
    config.layers_to_modify = None # Modify all layers
    # example: To modify only layers 0, 1, 10, 11: config.layers_to_modify = [0, 1, 10, 11]
    if config.layers_to_modify is None:
         print(f"*** CONFIG: Running Sparse Method (Triton Fixed Slice): Modifying ALL layers of {config.model_name} *** \n")
    else:
         print(f"*** CONFIG: Running Sparse Method (Triton Fixed Slice): Modifying layers {config.layers_to_modify} of {config.model_name} *** \n")


    # --- Control Flags ---
    DO_PROFILING = False # set to True to run the profiler on a specific task
    PROFILING_TASK = "sst2" # task to use for profiling if DO_PROFILING is True
    DO_BENCHMARK = True  # set to True to run the main benchmark

    if DO_PROFILING:
        print(f"\n === Starting Profiler Run (Task: {PROFILING_TASK}) ===\n")
        try:
            profile_original_model_backward(config, PROFILING_TASK)
        except Exception as e:
            print(f"An unexpected error occurred during profiling: {e}")
            traceback.print_exc()
        print(f"\n === Profiler Run Finished (Task: {PROFILING_TASK}) ===\n")


    if DO_BENCHMARK:
        print(f"\n === Starting Benchmark Run (Sparse Method - Triton N={config.sparse_rows_to_keep}) ===\n")
        print("!!! REMINDER: Sparse Method results are likely NUMERICALLY INCORRECT. !!!")
        time.sleep(3) 

        try:
            run_benchmark(config)
        except Exception as e:
            print(f"An unexpected error occurred during benchmark execution: {e}")
            traceback.print_exc()
        print("\n === Benchmark Run Finished ===\n")

    if not DO_PROFILING and not DO_BENCHMARK:
        print("Neither DO_PROFILING nor DO_BENCHMARK is set to True. Set one or both to True to run.")