#!/bin/bash
#SBATCH --job-name=export_rdc0
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/export_rdc0.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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

#python export_rdc0.py ../data/060287_010274515_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274515_rdc0.h5

python export_rdc0.py ../data/060287_010274635_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274635_rdc0.h5

python export_rdc0.py ../data/060287_010274747_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274747_rdc0.h5

python export_rdc0.py ../data/060287_010274866_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274866_rdc0.h5

python export_rdc0.py ../data/060287_010274978_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274978_rdc0.h5

python export_rdc0.py ../data/060287_010275122_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275122_rdc0.h5

python export_rdc0.py ../data/060287_010275234_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275234_rdc0.h5

python export_rdc0.py ../data/060287_010275353_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275353_rdc0.h5

python export_rdc0.py ../data/060287_010275465_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275465_rdc0.h5

python export_rdc0.py ../data/060287_010275584_preprocessed.dat --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275584_rdc0.h5

