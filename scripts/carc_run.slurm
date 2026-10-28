#!/bin/bash
# The whole corpus on USC CARC (Discovery) with one GPU: every stage, retraining, per-ride outputs and the report.
#
# One-time setup, on a login node (it has internet access; compute nodes may not):
#   git clone <this repository> && cd <repository>
#   module load python                     # any Python 3.10 or newer
#   python -m venv venv && source venv/bin/activate
#   pip install -r requirements-dev.txt
#   HF_HOME=$PWD/_cache/hf python -m bikesafe.run --download-models
#   mkdir -p videos work/logs              # copy the ride videos into videos/ (scp or Globus); to reuse stages already
#                                          # finished on another machine, copy its work/ folder as well
#
# Then:  sbatch scripts/carc_run.slurm     (log: work/logs/slurm-<job id>.out; VIDEOS=/path sbatch ... for another folder)
# Every stage is resumable: if the job reaches its time limit, submit it again and it continues where it stopped.

#SBATCH --job-name=bikesafe
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=work/logs/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
module load python 2>/dev/null || true
source venv/bin/activate
export HF_HOME="$PWD/_cache/hf"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m bikesafe.run "${VIDEOS:-videos}" --out-root work --render-minutes 1 \
    --jobs $(( ${SLURM_CPUS_PER_TASK:-8} / 2 ))
# The tables are in results/corpus and the evaluation in results/typology. The September 2026 upgrade report
# (scripts/build_upgrade_report.py) describes the pipeline of that milestone and is not rebuilt here.
