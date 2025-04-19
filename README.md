# ✨ Triton Sparse Backward Benchmark & Analysis ✨

This project explores and benchmarks a custom sparse backward pass implementation for Transformer Feed-Forward Network (FFN) layers using PyTorch and [Triton](https://github.com/openai/triton). It specifically implements and evaluates an efficient 'fixed slice' method where only the top N rows of the gradient tensor (`dY`) are computationally involved in the `dX = dY @ W` calculation during the backward pass, aiming for reduced latency.

The suite includes tools for:
* Latency benchmarking of the sparse Triton kernel against the original model.
* Sparsity analysis and gradient visualization.
* Comparative analysis of accuracy trajectories and latency between the original model, the sparse Triton method, and optionally the DropBP technique.

---

## 🚀 Features

* **Custom Triton Kernel:** A highly optimized `fixed_slice_matmul_kernel` written in Triton for the sparse backward computation, designed for potential speedups.
* **Latency Benchmarking:** Tools to compare the forward and backward pass latency of standard Hugging Face models vs. models using the custom sparse Triton kernel (`main.py`).
* **Sparsity Analysis & Visualization:** Script (`sparsity_plotter.py`) to analyze gradient sparsity patterns and generate insightful visualizations (heatmaps, histograms).
* **Accuracy & Speed Comparison:** Comprehensive script (`compare_methods.py`) to compare accuracy trajectories (across multiple runs with error bars) and latency between Original, Sparse (Triton), and optionally DropBP methods during fine-tuning on GLUE tasks.
* **Model Flexibility:** Supports analysis on different Transformer architectures like **BERT** and **RoBERTa**, automatically adapting layer replacement logic.
* **Configurable:** Easily change parameters within each script (model name, tasks, batch size, sequence length, sparse method parameters, layers to modify, etc.).
* **Data Handling:** Uses the `datasets` library to download and preprocess GLUE task data automatically.

---

## ⚙️ Prerequisites

* **Python:** Version 3.8 or higher.
* **pip:** Python package installer.
* **NVIDIA GPU:** CUDA-enabled GPU is required. Triton performance benefits significantly from newer architectures (Ampere/Hopper, Compute Capability 7.0+ recommended).
* **CUDA Toolkit:** Must be installed and compatible with your PyTorch and Triton versions. Check the PyTorch [website](https://pytorch.org/get-started/locally/) for compatibility.
* **(Optional) DropBP:** For the `compare_methods.py` script, install `dropbp` (`pip install dropbp`) if you wish to include it in the comparison.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    # git clone <your-repository-url> # If applicable
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
    Ensure your CUDA environment is correctly set up before installing `triton` and `torch`.
    ```bash
    pip install -r requirements.txt
    # You might also need: pip install matplotlib evaluate scikit-learn scipy # If not already included
    # Optional: pip install dropbp
    ```
    *Note: Depending on your system and CUDA setup, you might need to install PyTorch and Triton separately first, following instructions from their official websites.*

---

## ▶️ Usage

This project contains multiple scripts for different analyses:

### 1. Running Latency Benchmark (`main.py`)

This script focuses purely on measuring the forward and backward pass latency speedup achieved by replacing standard linear layers with the sparse Triton kernel implementation.

```bash
python main.py
