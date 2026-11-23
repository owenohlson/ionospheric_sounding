#!/bin/bash
#SBATCH --job-name=scan_lfm
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/scan_lfm.out
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

# Chirp waveform scan: which bandwidths / sweep rates compress, and their echo delays.
# Needs raw TBN (or an RDC0): an RDC1 is already dechirped with one bandwidth.
# Check the raw file names in $RAW (add the extension if they have one).
# Only --tstart .. --tstart + --duration is cut from each file (into $SLURM_TMPDIR).
# 8 stands x both pols = 16 inputs; about 5-10 min per file.
RAW=/home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09
STANDS=1,2,20,60,100,150,200,250

# Run script
python scan_lfm.py $RAW/060287_010274515 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010274515_lfm_scan.png --json ../plots/060287_010274515_lfm_scan.json   # 13.5 MHz

#python scan_lfm.py $RAW/060287_010274635 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010274635_lfm_scan.png --json ../plots/060287_010274635_lfm_scan.json   # 16.15 MHz

#python scan_lfm.py $RAW/060287_010274747 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010274747_lfm_scan.png --json ../plots/060287_010274747_lfm_scan.json   # 24.55 MHz

#python scan_lfm.py $RAW/060287_010274866 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010274866_lfm_scan.png --json ../plots/060287_010274866_lfm_scan.json   # 25.6 MHz

#python scan_lfm.py $RAW/060287_010274978 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010274978_lfm_scan.png --json ../plots/060287_010274978_lfm_scan.json   # 26.31 MHz

#python scan_lfm.py $RAW/060287_010275122 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010275122_lfm_scan.png --json ../plots/060287_010275122_lfm_scan.json   # 13.5 MHz

#python scan_lfm.py $RAW/060287_010275234 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010275234_lfm_scan.png --json ../plots/060287_010275234_lfm_scan.json   # 16.15 MHz

#python scan_lfm.py $RAW/060287_010275353 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010275353_lfm_scan.png --json ../plots/060287_010275353_lfm_scan.json   # 24.55 MHz

#python scan_lfm.py $RAW/060287_010275465 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010275465_lfm_scan.png --json ../plots/060287_010275465_lfm_scan.json   # 25.6 MHz

#python scan_lfm.py $RAW/060287_010275584 --stands $STANDS --pol both --tstart 60 --duration 20 --output ../plots/060287_010275584_lfm_scan.png --json ../plots/060287_010275584_lfm_scan.json   # 26.31 MHz

# Every plot written to ../plots adds its findings to ../plots/processing_summary.csv;
# print the merged table (one row per signal) at the end of the job
python summary_utils.py ../plots/processing_summary.csv
