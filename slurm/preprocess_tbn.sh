#!/bin/bash
#SBATCH --job-name=preprocess_tbn
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/preprocess_tbn.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
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
python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010274515 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274515_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010274635 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274615_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010274747 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274747_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010274866 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274866_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010274978 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010274978_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010275122 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275122_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010275234 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275234_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010275353 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275353_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010275465 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275465_preprocessed.dat --report-only

python preprocess_tbn.py /home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09/060287_010275584 --output=/home/onohlson/projects/def-peterdri/onohlson/data/060287_010275584_preprocessed.dat --report-only

