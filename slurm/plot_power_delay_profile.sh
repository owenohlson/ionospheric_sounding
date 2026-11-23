#!/bin/bash
#SBATCH --job-name=plot_power_delay_profile
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/plot_power_delay_profile.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:59:00
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
#python plot_power_delay_profile.py ../data/060287_010274515_rdc1.h5 --output ../plots/060287_010274515_pdp.png  --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010274635_rdc1.h5 --output ../plots/060287_010274635_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010274747_rdc1.h5 --output ../plots/060287_010274747_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010274866_rdc1.h5 --output ../plots/060287_010274866_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010274978_rdc1.h5 --output ../plots/060287_010274978_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010275122_rdc1.h5 --output ../plots/060287_010275122_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010275234_rdc1.h5 --output ../plots/060287_010275234_pdp.png --navg 4 --stands all

python plot_power_delay_profile.py ../data/060287_010275234_rdc1_bw99_fsweep2.h5 --output ../plots/060287_010275234_pdp_bw99_fsweep2.png --navg 4 --stands all

#python plot_power_delay_profile.py ../../LWA_Data/2023-12-09/060287_010275234 --output ../plots/060287_010275234_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010275353_rdc1.h5 --output ../plots/060287_010275353_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010275465_rdc1.h5 --output ../plots/060287_010275465_pdp.png --navg 4 --stands all
#python plot_power_delay_profile.py ../data/060287_010275584_rdc1.h5 --output ../plots/060287_010275584_pdp.png --navg 4 --stands all
