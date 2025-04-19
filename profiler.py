import torch
import torch.nn as nn
import torch.profiler
import traceback
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from benchmark_utils import prepare_data, clear_gpu_cache

print("Defining profiler function...")

def profile_original_model_backward(config, task):
    """Profiles the backward pass of the original model for a given task."""
    if not torch.cuda.is_available():
        print("CUDA is required for profiling.")
        return

    device = torch.device("cuda")
    print(f"\n--- Profiling Original Model Backward Pass for Task: {task} ---")
    print(f"Model: {config.model_name}, Batch Size: {config.batch_size}, Seq Length: {config.seq_length}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)

        is_regression = (task == 'stsb')
        if is_regression:
            model_config.num_labels = 1
            print(f"Configuring model for regression (num_labels=1) for task {task}.")
        else:
            task_labels = {
                "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
                "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2, "ax": 3 
            }
            num_labels = task_labels.get(task, 2)
            model_config.num_labels = num_labels
            print(f"Configuring model for classification (num_labels={num_labels}) for task {task}.")

        if getattr(model_config, "id2label", None) is None or len(model_config.id2label) != model_config.num_labels:
             model_config.id2label = {i: f"LABEL_{i}" for i in range(model_config.num_labels)}
             model_config.label2id = {v: k for k, v in model_config.id2label.items()}

    except Exception as e:
        print(f"Error loading tokenizer or config for {config.model_name}: {e}")
        traceback.print_exc()
        return

    profile_batches = prepare_data(task, tokenizer, config.batch_size, config.seq_length, num_batches=1)
    if not profile_batches:
        print(f"Cannot prepare data for profiling task {task}. Skipping profiling.")
        return
    profile_batch_gpu = {k: v.cuda() for k, v in profile_batches[0].items() if isinstance(v, torch.Tensor)}

    model = None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            config=model_config,
            trust_remote_code=True
        ).to(device)
        model.train() 

        outputs = model(**profile_batch_gpu)
        logits = outputs.logits
        labels = profile_batch_gpu['labels']
        loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
        if is_regression:
            loss = loss_fn(logits.squeeze(-1), labels.float())
        else:
            loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))

        print("Starting profiler...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,       
            profile_memory=True,       
            with_stack=True            
        ) as prof:
            loss.backward() 

        print("Profiling complete.")

        print("\nProfiler Results (Top 20 GPU Operators by Total Time, Grouped by Input Shape):")
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))

        # Export detailed trace for Chrome Tracing / Perfetto UI
        trace_file = f"./{config.model_name.replace('/', '_')}_{task}_backward_profile.json" 
        try:
             prof.export_chrome_trace(trace_file)
             print(f"\nDetailed trace saved to: {trace_file}")
             print("You can upload this file to chrome://tracing or https://ui.perfetto.dev/ for visualization.")
        except Exception as export_e:
             print(f"Error exporting trace file: {export_e}")


    except Exception as e:
        print(f"Error during profiling execution for task {task}: {e}")
        traceback.print_exc()

    finally:
        del model
        del profile_batch_gpu
        del profile_batches
        if 'loss' in locals(): del loss
        clear_gpu_cache()
        print(f"--- Finished Profiling for Task: {task} ---")


print("Profiler function defined.")
print("-" * 40)