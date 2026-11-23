#!/bin/bash
#SBATCH --job-name=plot_phase_doppler
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/plot_phase_doppler.out
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
python plot_phase_doppler.py ../data/060287_010274515_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010274515_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274635_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010274635_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274747_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010274747_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274866_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010274866_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274978_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010274978_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275122_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010275122_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275234_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010275234_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275353_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010275353_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275465_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010275465_x_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275584_rdc2p.h5 --pol x --bad-stands=status --output ../plots/060287_010275584_x_phase_doppler.png

# y
python plot_phase_doppler.py ../data/060287_010274515_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010274515_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274635_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010274635_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274747_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010274747_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274866_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010274866_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010274978_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010274978_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275122_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010275122_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275234_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010275234_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275353_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010275353_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275465_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010275465_y_phase_doppler.png

python plot_phase_doppler.py ../data/060287_010275584_rdc2p.h5 --pol y --bad-stands=status --output ../plots/060287_010275584_y_phase_doppler.png
