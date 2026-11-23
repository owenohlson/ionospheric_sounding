#!/bin/bash
#SBATCH --job-name=plot_doppler_time
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/plot_doppler_time.out
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
#python plot_doppler_time.py ../data/060287_010274515_rdc1.h5 --output ../plots/2023-12-09/060287_010274515/060287_010274515_doppler_time_10int_5hop.png  --stands all --gap-fill=zero --integration-time=10 --hop-time=5

#python plot_doppler_time.py ../data/060287_010274635_rdc1.h5 --output ../plots/2023-12-09/060287_010274635/060287_010274635_doppler_time.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010274747_rdc1.h5 --output ../plots/2023-12-09/060287_010274747/060287_010274747_doppler_time.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010274866_rdc1.h5 --output ../plots/2023-12-09/060287_010274866/060287_010274866_doppler_time.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010274978_rdc1.h5 --output ../plots/2023-12-09/060287_010274978/060287_010274978_doppler_time.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010275122_rdc1.h5 --output ../plots/2023-12-09/060287_010275122/060287_010275122_doppler_time.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010275234_rdc1.h5 --output ../plots/2023-12-09/060287_010275234/060287_010275234_doppler_time.png --stands all --gap-fill=zero

python plot_doppler_time.py ../data/060287_010275234_rdc1_bw99_fsweep2.h5 --output ../plots/2023-12-09/060287_010275234/060287_010275234_doppler_time_bw99_fsweep2.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010275353_rdc1.h5 --output ../plots/2023-12-09/060287_010275353/060287_010275353_doppler_time.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010275465_rdc1.h5 --output ../plots/2023-12-09/060287_010275465/060287_010275465_doppler_time.png --stands all --gap-fill=zero

#python plot_doppler_time.py ../data/060287_010275584_rdc1.h5 --output ../plots/2023-12-09/060287_010275584/060287_010275584_doppler_time.png --stands all --gap-fill=zero
