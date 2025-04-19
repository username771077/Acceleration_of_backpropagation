# ✨ Triton Sparse Backward Benchmark ✨

This project benchmarks a custom sparse backward pass implementation for Transformer Feed-Forward Network (FFN) layers using PyTorch and [Triton](https://github.com/openai/triton). It specifically implements and evaluates a 'fixed slice' method where only the top N rows of the gradient tensor (`dY`) are computationally involved in the `dX = dY @ W` calculation during the backward pass.

**⚠️ Important Warning:** The 'sparse method' implemented here **aggressively truncates gradient information** by only considering the top `N` rows based on their initial layout in the flattened tensor. This is **highly likely to be numerically incorrect** for actual model training and will likely prevent model convergence. Its purpose in this project is *purely* to benchmark the potential speedup of such a restricted kernel *if* such sparsity could be correctly achieved or justified by other means (which is not the case here). **Do not use this specific backward pass implementation for actual model training.**

---

## 🚀 Features

* **Custom Triton Kernel:** A highly optimized `fixed_slice_matmul_kernel` written in Triton for the sparse backward computation.
* **Benchmarking Framework:** Compares the latency (Forward & Backward passes) of a standard Hugging Face Transformer model against the same model modified to use the custom Triton kernel for selected FFN layers.
* **Model Flexibility:** Supports benchmarking on different Transformer architectures like **BERT** and **RoBERTa**. The appropriate layer replacement logic is chosen automatically based on the loaded model configuration.
* **Configurable:** Easily change benchmark parameters:
    * Model name (from Hugging Face Hub)
    * GLUE tasks for evaluation
    * Batch size & Sequence length
    * Number of sparse rows (`N`) to keep in the backward pass
    * Specific layers to apply the modification (or all layers)
* **Data Handling:** Uses the `datasets` library to download and preprocess GLUE task data automatically.
* **Optional Profiling:** Includes functionality to profile the backward pass of the original model using `torch.profiler`.

---

## ⚙️ Prerequisites

* **Python:** Version 3.8 or higher.
* **pip:** Python package installer.
* **NVIDIA GPU:** CUDA-enabled GPU is required. Triton performance benefits significantly from newer architectures (Ampere/Hopper, Compute Capability 7.0+ recommended).
* **CUDA Toolkit:** Must be installed and compatible with your PyTorch and Triton versions. Check the PyTorch [website](https://pytorch.org/get-started/locally/) for compatibility.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your repo URL
    cd <repository-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # Linux / macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows (Git Bash / WSL)
    python -m venv venv
    source venv/Scripts/activate

    # Windows (Command Prompt / PowerShell)
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    The `triton` package often requires specific CUDA versions. Ensure your environment is set up correctly before installing.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system and CUDA setup, you might need to install PyTorch separately first, following instructions from the official PyTorch website.*

---

## ▶️ Usage

### Running the Benchmark

The main script `main.py` controls the benchmark execution.

```bash
python main.py
