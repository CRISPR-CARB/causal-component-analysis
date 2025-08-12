#!/usr/bin/env bash
# Minimal Slurm launcher for Causal Component Analysis (CCA)
# Usage: update sbatch script below and run using `sbatch slurm/run_cca.sbatch`

#SBATCH --job-name=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                   
#SBATCH --time=08:00:00              
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=<your_email@example.com>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=<your_partition>   # pick multi GPU cluster
#SBATCH --gres=gpu:4

set -euo pipefail
echo "[CCA] Job: $SLURM_JOB_ID | Host: $(hostname) | PWD: $(pwd)"

# Load modules and activate environment (REQUIRED: edit these)
module purge
module load <cuda_module>
module load <python_module>
source <path_to_env>/bin/activate

# Default training parameters (edit as needed)
SEED=0
TSEED=0
MODEL=nonlinear
BATCH_SIZE=4096
LR=5e-4
MAX_EPOCHS=100
K_FLOWS=12
DGP_NAME=graph-4-random-1
N_NONLIN=2
LR_SCHED=cosine
LR_MIN=1e-7
ACCELERATOR=gpu

# Distributed CPU/GPU parameters for Lightning PyTorch
DEVICES=4
NUM_NODES=1
STRATEGY=ddp

# Optional NCCL hints for multi-GPU runs (edit if desired)
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# Basic validation for accelerator value
if [[ "$ACCELERATOR" != "cpu" && "$ACCELERATOR" != "gpu" ]]; then
  echo "[CCA][ERROR] ACCELERATOR must be 'cpu' or 'gpu'." >&2
  exit 1
fi

# run CCA command
CMD=( python -m experiments.cauca.main \
  --seed "$SEED" \
  --training-seed "$TSEED" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-epochs "$MAX_EPOCHS" \
  --k-flows "$K_FLOWS" \
  --dgp "$DGP_NAME" \
  --n-nonlinearities "$N_NONLIN" \
  --lr-scheduler "$LR_SCHED" \
  --lr-min "$LR_MIN" \
  --accelerator "$ACCELERATOR" \
  --devices "$DEVICES" \
  --num-nodes "$NUM_NODES" \
  --strategy "$STRATEGY" \
  --no-wandb \
)

echo "[CCA] Launching: ${CMD[*]}"
srun "${CMD[@]}"
