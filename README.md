# SimulGen-VAE

A high-performance VAE for fast generation and parameter inference of time-dependent simulation data with Physics-Aware Neural Network (PANN) integration.
Supports three tasks
- Parametric estimations
- Non-parametric estimations
- Probabilistic estimations

## Author
SiHun Lee, Ph. D, kevin1007kr@gmail.com, [LinkedIn Profile](https://www.linkedin.com/in/%EC%8B%9C%ED%9B%88-%EC%9D%B4-13009a172/?originalSubdomain=kr)


## Table of Contents
1. [Prerequisites](#prerequisites)  
2. [Quick-Start](#quick-start)  
3. [Performance Optimizations](#performance-optimizations)
4. [Multi-GPU Training](#multi-gpu-training)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Acknowledgements](#acknowledgements)

## Prerequisites
* **Python ≥ 3.9** (tested on 3.10)  
* **PyTorch ≥ 2.0** with CUDA support
* CUDA-capable GPU with ≥12 GB memory *(RTX 30/40-series, A100, H100 recommended)*
* For multi-GPU: NCCL backend support

## Quick-Start

### Single GPU Training
```bash
# Default settings
python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=0

# With optimizations (recommended for small-variety datasets)
python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=1

# Use optimization presets
python accelerate_training.py --scenario maximum_speed
```

### Multi-GPU Training with DDP
```bash
# Replace NUM_GPUS with the number of GPUs you want to use
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=1
```

## Performance Optimizations

SimulGen-VAE includes several advanced optimizations for maximum training speed:

### 1. Data Loading Optimizations
- **GPU Prefetching**: Preloads entire dataset to GPU when `--load_all=1`
- **Pinned Memory**: Uses pinned CPU memory for faster transfers when `--load_all=0`
- **Async Data Transfers**: Non-blocking transfers with `non_blocking=True`

### 2. Model Architecture Optimizations
- **Channels Last Memory Format**: Better GPU memory access patterns
- **Mixed Precision Training**: FP16/BF16 for forward pass, FP32 for critical operations
- **TF32 Support**: Enabled on Ampere+ GPUs for faster matrix operations

### 3. Training Loop Optimizations
- **CUDA Graphs**: Captures and replays computation graphs for operations with consistent shapes
- **Gradient Accumulation**: Available via optimization configs for larger effective batch sizes
- **Memory-Efficient Operations**: Uses `set_to_none=True` for zero_grad and other optimizations

### 4. Multi-GPU Support
- **Distributed Data Parallel (DDP)**: Scales training across multiple GPUs
- **Automatic Batch Size Adjustment**: Maintains global batch size across GPUs
- **Efficient Parameter Synchronization**: Uses NCCL backend for fast GPU-to-GPU communication

## Multi-GPU Training

### Key Features
- **Automatic Batch Size Scaling**: Global batch size is maintained by dividing by the number of GPUs
- **Efficient Gradient Synchronization**: Uses NCCL backend for optimal GPU-to-GPU communication
- **Model Replication**: Each GPU maintains its own copy of the model with synchronized gradients
- **Dataset Partitioning**: Data is automatically sharded across GPUs using DistributedSampler

### Performance Expectations
- Near-linear scaling with number of GPUs for compute-bound workloads
- 1.8-1.9x speedup with 2 GPUs, 3.5-3.8x with 4 GPUs (typical values)

## Configuration

### Command-line Arguments
| Flag | Description | Example |
|------|-------------|---------|
| `--preset` | Pick dataset preset (int) | `--preset=1` |
| `--plot` | Plot option (1 = show, 2 = off) | `--plot=2` |
| `--train_pinn_only` | 1 = skip VAE and train only PINN | `--train_pinn_only=0` |
| `--size` | Network size preset (`small` / `large`) | `--size=small` |
| `--load_all` | 1 = preload entire dataset to GPU | `--load_all=1` |
| `--use_ddp` | Enable distributed data parallel training | `--use_ddp` |
| `--local_rank` | Local rank for distributed training | `--local_rank=0` |

### Dataset Format
The VAE expects a 3-D NumPy array saved as a Pickle file: `[num_param, num_time, num_node]`

### Optimization Presets
Use `optimization_config.py` to select predefined scenarios:
- `small_variety_large_batch`: For datasets with limited variety but large batch sizes
- `maximum_speed`: Aggressive optimizations for maximum training speed
- `memory_constrained`: For limited GPU memory
- `safe_mode`: Safe optimizations without model compilation

### Model Compilation
The VAE supports `torch.compile` with selectable modes:
- `default`: Conservative, most stable  
- `reduce-overhead`: Faster compile time  
- `max-autotune`: Highest performance but may break on exotic ops

## Troubleshooting
| Issue | Fix |
|-------|-----|
| **OOM / CUDA out-of-memory** | Reduce `Batch_size` in `condition.txt` or use `--scenario memory_constrained` |
| **NaN losses** | Lower `gradient_clipping` in `modules/train.py` (default 5.0 → 1.0) |
| **Slow first epoch** | Expected if compilation is enabled (one-off cost) |
| **torch.compile error** | Switch to safe mode (`compile_model=False`) |
| **CUDA Graph errors** | Set `use_cuda_graphs = False` in `modules/train.py` |
| **DDP initialization failures** | Check that all GPUs are visible with `nvidia-smi` |
| **Different results across GPUs** | Set `torch.backends.cudnn.deterministic = True` for reproducibility |
| **RuntimeError: in-place operation** | Disable CUDA graphs by setting `use_cuda_graphs = False` in `modules/train.py` |

## Monitoring
* **TensorBoard**: `tensorboard --logdir=runs --port 6001`
* **GPU usage**: `watch -n 0.5 nvidia-smi`

## Acknowledgements
* **SiHun Lee, Ph.D.** – Original LSH-VAE implementation, Developer of this code
* **PyTorch 2.0** – `torch.compile`, mixed-precision, and TF32 support  
* **NVIDIA** – CUDA/cuDNN performance libraries