# SimulGen-VAE

A high-performance, mixed-precision Variational Auto-Encoder (VAE) designed for fast generation and parameter inference of time-dependent simulation data.  
This repository also includes a Physics-Inspired Neural Network (PINN) that can learn physical parameters from the latent space produced by the VAE.

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Dataset & Configuration](#dataset--configuration)  
5. [Quick-Start Examples](#quick-start-examples)  
6. [Running With Optimisations](#running-with-optimisations)  
7. [Command-line Arguments](#command-line-arguments)  
8. [Monitoring & Visualisation](#monitoring--visualisation)  
9. [Troubleshooting](#troubleshooting)  
10. [Acknowledgements](#acknowledgements)

---

## Project Structure
```text
SimulGenVAE/
├── SimulGen-VAE.py           # Main training / inference entry-point
├── accelerate_training.py    # Helper script for choosing optimisation presets
├── optimization_config.py    # Centralised optimisation settings
├── modules/                  # Core library (VAE, PINN, utils, ...)
├── input_data/
│   └── condition.txt         # Dataset-specific hyper-parameters
├── datasetX.pickle           # Your dataset(s) → see below
├── checkpoints/              # Model checkpoints (auto-created)
├── model_save/               # Saved .pth models & artefacts (auto-created)
├── runs/ , PINNruns/         # TensorBoard logs (auto-created)
└── requirements.txt          # Python dependencies
```

---

## Prerequisites
* **Python ≥ 3.9** (tested on 3.10)  
* **PyTorch ≥ 2.0** with CUDA support (for `torch.compile` & mixed precision)  
* A CUDA-capable GPU with ≥12 GB memory *(RTX 30/40-series, A100, H100 recommended)*

Install NVIDIA driver + CUDA Toolkit beforehand (see [PyTorch installation guide](https://pytorch.org/)).

---

## Installation
```bash
# 1) Clone the repository
git clone <your fork or URL>
cd SimulGenVAE

# 2) Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip 📝** `requirements.txt` lists generic CPU + CUDA-agnostic wheels.  
> For maximum GPU performance install the official NVIDIA/PyTorch wheels for your CUDA version.

---

## Dataset & Configuration
### 1. Dataset format
The VAE expects a **3-D NumPy array** saved as a Pickle file:
```
[num_param, num_time, num_node]
```
* `num_param`  – number of simulation parameter sets  
* `num_time`    – temporal resolution (timesteps)  
* `num_node`    – spatial nodes / sensors per timestep

Save it as `datasetX.pickle` in the project root, where **X** is an integer (e.g. `dataset1.pickle`).  
You can store multiple pickles (`dataset2.pickle`, …) and select which one to load via `--preset`.

### 2. `input_data/condition.txt`
This text file controls most hyper-parameters (dimensions, training epochs, learning-rate, …).  
It is automatically parsed by `SimulGen-VAE.py`.  A minimal example:
```text
# --- Dimensions ---
Dim1           1000     # num_param
Dim2           128      # num_time (original)
Dim2_red       128      # num_time after reduction
Dim3           256      # num_node (original)
Dim3_red       256      # num_node after reduction

# --- Training ---
Training_epochs     5000
Batch_size          64
LearningR           5e-4

Latent_dim          32
Latent_dim_end      32
Loss_type           0        # 0=MSE, 1=MAE, 2=SmoothL1, 3=Huber
Stretch             0        # Data augmentation flag
```
Adjust to match your dataset.

---

## Quick-Start Examples
### 1. Minimal run (default settings)
```bash
python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=0
```

### 2. Full GPU preload + safe optimisations *(recommended for small-variety datasets)*
```bash
# Pre-loads all data to GPU, mixed precision, conservative torch.compile
python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=1
```

### 3. Use optimisation presets via helper script
```bash
# List of scenarios: small_variety_large_batch | maximum_speed | memory_constrained | safe_mode
python accelerate_training.py --scenario maximum_speed
# The script prints the fully-qualified command to run
```

---

## Running With Optimisations
The central switchboard is `optimization_config.py`. Each *scenario* enables/disables:
* **Data loading** (`load_all`, cached dataset, num_workers, …)
* **Model compilation** (`torch.compile` mode or disabled)
* **Batch-size multiplier** and gradient-accumulation
* **TF32 / mixed-precision flags**

Use **Safe Mode** if you encounter compilation errors:
```bash
python accelerate_training.py --scenario safe_mode
```

### Torch 2 Model Compilation
The VAE supports `torch.compile` with selectable modes:
* `default`            – conservative, most stable  
* `reduce-overhead`    – faster compile time  
* `max-autotune`       – highest performance but may break on exotic ops

The compile mode is set inside the selected scenario (`compile_model` field).  
If compilation fails the model **automatically falls back** to an un-compiled version.

---

## Command-line Arguments
| Flag | Description | Example |
|------|-------------|---------|
| `--preset` | Pick dataset preset (int) | `--preset=1` |
| `--plot` | Plot option (1 = show, 2 = off) | `--plot=2` |
| `--train_pinn_only` | 1 = skip VAE and train only PINN | `--train_pinn_only=0` |
| `--size` | Network size preset (`small` / `large`) | `--size=small` |
| `--load_all` | 1 = preload entire dataset to GPU | `--load_all=1` |

PINN-specific flags can be edited inside `condition.txt`.

---

## Monitoring & Visualisation
* **TensorBoard** – training loss curves
  ```bash
  tensorboard --logdir=runs --port 6001        # VAE
  tensorboard --logdir=PINNruns --port 6002    # PINN
  ```
* **GPU usage** – real-time monitoring
  ```bash
  watch -n 0.5 nvidia-smi
  ```

Generated artefacts:
* `checkpoints/SimulGen-VAE.pth` – final state-dict  
* `model_save/SimulGen-VAE`     – full scripted model  
* `model_save/latent_vectors.npy`, `xs.npy` – saved latent spaces

---

## Troubleshooting
| Issue | Fix |
|-------|-----|
| **OOM / CUDA out-of-memory** | Reduce `Batch_size` in `condition.txt` *or* run `accelerate_training.py --scenario memory_constrained` |
| **NaN losses** | Lower `gradient_clipping` in `modules/train.py` (default 5.0 → 1.0) |
| **Slow first epoch** | Expected if compilation is enabled (one-off cost) |
| **torch.compile error (permute dims)** | Switch to safe mode (`compile_model=False`) |

---

## Acknowledgements
* **SiHun Lee, Ph.D.** – Original LSH-VAE implementation  
* **PyTorch 2.0** – `torch.compile`, mixed-precision, and TF32 support  
* **NVIDIA** – CUDA/cuDNN performance libraries

Feel free to open issues or pull-requests for improvements ❤️ 