#!/bin/bash
#SBATCH --job-name=export_rdc2p
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/export_rdc2p.out
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

# Virtual environment
source /project/6003738/onohlson/lwa_env/bin/activate
cd /project/6003738/onohlson/tools

# Load station data
export LWA_SSMIF=$(pwd)/resources/ssmif/LD021_0001.tgz   # shipped in tools/resources

# Run script
python export_rdc2p.py ../data/060287_010274515_preprocessed.dat --output ../data/060287_010274515_rdc2p.h5 --sweep-frequency 2 --bandwidth=-99.259377e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010274635_preprocessed.dat --output=../data/060287_010274615_rdc2p.h5 --sweep-frequency 2 --bandwidth=-99.259377e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010274747_preprocessed.dat --output=../data/060287_010274747_rdc2p.h5 --sweep-frequency 2 --bandwidth=-99.259377e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010274866_preprocessed.dat --output=../data/060287_010274866_rdc2p.h5 --sweep-frequency 2 --bandwidth=-150.727203e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010274978_preprocessed.dat --output=../data/060287_010274978_rdc2p.h5 --sweep-frequency 4 --bandwidth=-220.57639848e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010275122_preprocessed.dat --output=../data/060287_010275122_rdc2p.h5 --sweep-frequency 2 --bandwidth=-99.259377e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010275234_preprocessed.dat --output=../data/060287_010275234_rdc2p.h5 --sweep-frequency 4 --bandwidth=-49.629688e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010275353_preprocessed.dat --output=../data/060287_010275353_rdc2p.h5 --sweep-frequency 2 --bandwidth=-99.259377e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010275465_preprocessed.dat --output=../data/060287_010275465_rdc2p.h5 --sweep-frequency 2 --bandwidth=-150.727203e3 --pol both --bad-stands status --gap-fill zero 

python export_rdc2p.py ../data/060287_010275584_preprocessed.dat --output=../data/060287_010275584_rdc2p.h5 --sweep-frequency 4 --bandwidth=-220.57639848e3 --pol both --bad-stands status --gap-fill zero 
