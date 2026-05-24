# GPU Training Guide for Kronos Classification

This guide explains how to efficiently use GPUs for maximum training and inference speed.

## Hardware Overview

```
GPU 0: RTX3090 24GB (cuda:0)
GPU 1: RTX3090 24GB (cuda:1)
```

Total: 48GB GPU Memory

## Key Optimizations Implemented

### 1. Automatic GPU Selection
- Auto-detects GPU with most free memory
- Can manually specify any GPU
- Intelligent resource allocation

### 2. Data Loading Optimizations
- **Pin Memory**: Enabled for faster CPU→GPU transfer
- **Prefetching**: 2 batches preloaded per worker (configurable)
- **Persistent Workers**: Reduces worker startup overhead
- **Multiple Workers**: 2 parallel data loaders (configurable)

### 3. Training Optimizations
- **FP16 Mixed Precision**: ~2x speedup, ~50% memory reduction
- **Gradient Accumulation**: Larger effective batch size
- **Distributed Data Parallel (DDP)**: Efficient multi-GPU training
- **find_unused_parameters=False**: Faster DDP synchronization

### 4. Memory Optimizations
- Gradient checkpointing for large models
- Efficient padding strategy
- Optimized attention masks

## Usage Examples

### Single GPU Training (Auto-Select Fastest)

```bash
# Automatically selects GPU with most free memory
python classification/kronos_pretrain.py \
    --data_dir /path/to/data \
    --num_classes 2 \
    --fp16 \
    --batch_size 32
```

### Single GPU Training (Manual Selection)

```bash
# Use specific GPU (e.g., cuda:1 - V100 32GB)
python classification/kronos_pretrain.py \
    --data_dir /path/to/data \
    --num_classes 2 \
    --device cuda:1 \
    --fp16 \
    --batch_size 64
```

### Multi-GPU Training (All 2 GPUs)

```bash
# Use all 2 GPUs with torchrun
torchrun --standalone --nproc_per_node=2 classification/kronos_pretrain.py \
    --data_dir /path/to/data \
    --num_classes 2 \
    --fp16 \
    --batch_size 32
```

With torchrun:
- Each GPU gets `batch_size / num_gpus` = 8 samples per GPU
- Effective batch size = 32 × 2 = 64
- Training speed ~2x faster than single GPU

### Multi-GPU Training (Specific GPUs Only)

```bash
# Use only 3 GPUs (e.g., the 32GB ones)
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 \
    classification/kronos_pretrain.py \
    --data_dir /path/to/data \
    --num_classes 2 \
    --fp16 \
    --batch_size 48
```

### Fine-tuning with GPU Selection

```bash
# Single GPU (auto-detect)
python classification/kronos_finetune.py \
    --data_dir /path/to/data \
    --pretrained_checkpoint ./pretrain_checkpoints/best_model \
    --num_classes 2 \
    --fp16 \
    --batch_size 32

# Multi-GPU (all 2)
torchrun --standalone --nproc_per_node=2 classification/kronos_finetune.py \
    --data_dir /path/to/data \
    --pretrained_checkpoint ./pretrain_checkpoints/best_model \
    --num_classes 2 \
    --fp16 \
    --batch_size 32
```

### RL Fine-tuning

```bash
# Single GPU (auto-detect fastest)
python classification/kronos_rl_finetune.py \
    --model_path ./finetuned_checkpoints/best_model \
    --data_dir /path/to/data \
    --fp16 \
    --batch_size 16

# Multi-GPU
torchrun --standalone --nproc_per_node=4 classification/kronos_rl_finetune.py \
    --model_path ./finetuned_checkpoints/best_model \
    --data_dir /path/to/data \
    --fp16 \
    --batch_size 16
```

### Inference

```python
from classification.kronos_inference import KronosClassificationPipeline

# Auto-detect fastest GPU
pipeline = KronosClassificationPipeline(
    model_path="./checkpoints/best_model",
    batch_size=64  # Larger batch for faster inference
)

# Or specify GPU
pipeline = KronosClassificationPipeline(
    model_path="./checkpoints/best_model",
    device="cuda:2",  # Use specific GPU
    batch_size=64
)
```

## Recommended Batch Sizes

### Single GPU (RTX3090 24GB)
```
Without FP16:
- batch_size: 12-18 (conservative)
- batch_size: 24 (if model fits)

With FP16 (recommended):
- batch_size: 32-48 (standard)
- batch_size: 64 (aggressive, may OOM)
```

### Multi-GPU (2× RTX3090)
```
With FP16, per-GPU batch sizes:
- batch_size: 16 per GPU (total 64) - safe for all
- batch_size: 24 per GPU (total 48) - recommended
```

**Note**: When using multi-GPU, ensure batch_size fits the **smallest GPU** (16GB).

## Performance Optimization Tips

### 1. Use FP16 Whenever Possible
```bash
--fp16
```
- ~2x training speedup
- ~50% memory reduction
- Negligible accuracy loss (often improves)

### 2. Adjust Workers and Prefetch
```bash
--num_workers 8 --prefetch_factor 4
```
- For systems with many CPU cores
- More workers = faster data loading
- Too many workers can slow down (I/O bottleneck)

### 3. Use Gradient Accumulation
```bash
--batch_size 16 --gradient_accumulation_steps 4
```
- Effective batch size = 16 × 4 = 64
- Uses less memory per batch
- Good when close to OOM

### 4. Optimize Data Loading
```bash
# For fast storage (SSD/NVMe)
--num_workers 8 --prefetch_factor 4

# For slower storage (HDD)
--num_workers 4 --prefetch_factor 2
```

### 5. Choose Right Context Length
```bash
--max_context 256  # Instead of 512
```
- Shorter sequences = faster training
- Trade-off: less historical context

## Memory Optimization

### If You Run Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 8

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Reduce max_context
--max_context 256

# Reduce num_workers
--num_workers 2

# Use FP16
--fp16
```

### Monitor GPU Memory

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

### Profile Memory Usage

```python
import torch
print(torch.cuda.memory_summary())
```

## Production Training Pipeline

### Stage 1: Pre-training (All 2 GPUs)
```bash
torchrun --standalone --nproc_per_node=2 classification/kronos_pretrain.py \
    --data_dir /path/to/labeled_data \
    --kronos_model NeoQuasar/Kronos-base \
    --tokenizer_path NeoQuasar/Kronos-Tokenizer-base \
    --num_classes 2 \
    --output_dir ./pretrain_checkpoints \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --fp16 \
    --save_format safetensors
```

### Stage 2: Fine-tuning (Single 24GB GPU)
```bash
python classification/kronos_finetune.py \
    --data_dir /path/to/new_data \
    --pretrained_checkpoint ./pretrain_checkpoints/best_model \
    --num_classes 2 \
    --output_dir ./finetuned_checkpoints \
    --device cuda:1 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --freeze_backbone_epochs 1 \
    --fp16
```

### Stage 3: RL Fine-tuning (All 2 GPUs)
```bash
torchrun --standalone --nproc_per_node=2 classification/kronos_rl_finetune.py \
    --model_path ./finetuned_checkpoints/best_model \
    --data_dir /path/to/data \
    --output_dir ./rl_checkpoints \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --fp16
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce `--batch_size`
2. Add `--fp16`
3. Reduce `--max_context`
4. Increase `--gradient_accumulation_steps`

### Issue: "NCCL errors in multi-GPU"
**Solutions:**
1. Check NCCL version: `python -c "import torch; print(torch.cuda.nccl.version())"`
2. Set environment variable: `export NCCL_P2P_DISABLE=1`
3. Use fewer GPUs: `CUDA_VISIBLE_DEVICES=0,1,2`

### Issue: Slow data loading
**Solutions:**
1. Increase `--num_workers`
2. Increase `--prefetch_factor`
3. Use faster storage (SSD instead of HDD)
4. Disable debug logging

### Issue: Uneven GPU utilization
**Solutions:**
1. Check all GPUs are visible: `torch.cuda.device_count()`
2. Ensure batch_size is divisible by num_gpus
3. Check for CPU bottleneck (increase num_workers)

## Monitoring and Debugging

### Real-time Monitoring
```bash
# Terminal 1: Run training
torchrun --standalone --nproc_per_node=4 classification/kronos_pretrain.py ...

# Terminal 2: Monitor GPUs
watch -n 0.5 nvidia-smi
```

### Log Training Speed
Add to training script or use tensorboard:
```python
import time
start = time.time()
# ... training code ...
samples_per_second = len(dataset) * num_epochs / (time.time() - start)
print(f"Training speed: {samples_per_second:.2f} samples/second")
```

## Best Practices Summary

1. **Always use FP16** (`--fp16`) for 2x speedup
2. **Auto-detect GPU** for single GPU training (default)
3. **Use all GPUs** for pre-training with `torchrun`
4. **Batch size** 20-32 per GPU (fits 24GB GPU with FP16)
5. **Num workers** 4-8 for data loading
6. **Prefetch factor** 2-4 for overlapping CPU/GPU work
7. **Gradient accumulation** 2-4 for larger effective batch size
8. **Monitor GPU usage** with `nvidia-smi` during training

## Quick Reference

```bash
# Auto-detect fastest GPU (single GPU)
python classification/kronos_pretrain.py --data_dir ... --fp16

# Use specific GPU
python classification/kronos_pretrain.py --data_dir ... --device cuda:2 --fp16

# Use all 2 GPUs
torchrun --standalone --nproc_per_node=2 classification/kronos_pretrain.py --data_dir ... --fp16

# Use specific GPUs only
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=3 ...

# Maximum speed (all optimizations)
torchrun --standalone --nproc_per_node=2 classification/kronos_pretrain.py \
    --data_dir ... --fp16 --batch_size 24 --num_workers 8 --prefetch_factor 4
```

With these optimizations, you should achieve **2x speedup** on 2 GPUs compared to single GPU training!
