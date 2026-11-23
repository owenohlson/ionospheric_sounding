#!/bin/bash
#SBATCH --job-name=simulate_lwa_rdc2
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/simulate_lwa_rdc2.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onohlson@gmail.com

set -euo pipefail

unset PYTHONPATH
unset PYTHONHOME

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load gsl
module load fftw
module load hdf5

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

#Virtual environment
source /project/6003738/onohlson/lwa_env/bin/activate
cd /project/6003738/onohlson/tools

# Run script
python simulate_lwa_rdc2.py \
  --output ../data/sim_dd_box_rdc2.h5 \
  --scenario box \
  --frames 100 \
  --max-channels 0 \
  --bad-stands status \
  --pol=x

python simulate_lwa_rdc2.py \
  --output ../data/sim_dd_angle_rdc2.h5 \
  --scenario angle \
  --frames 100 \
  --max-channels 0 \
  --bad-stands status \
  --pol=x

python simulate_lwa_rdc2.py \
  --output ../data/sim_dd_multi_rdc2.h5 \
  --scenario multi \
  --frames 100 \
  --max-channels 0 \
  --bad-stands status \
  --pol=x

python simulate_lwa_rdc2.py \
  --output ../data/sim_dd_amplitude_rdc2.h5 \
  --scenario amplitude \
  --frames 100 \
  --max-channels 0 \
  --bad-stands status \
  --pol=x
