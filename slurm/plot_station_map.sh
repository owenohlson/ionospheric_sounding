#!/bin/bash
#SBATCH --job-name=plot_station_map
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/plot_station_map.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
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

# No recording or station description needed: stations, coastlines, GIRO ionosondes and their
# 2023-12-09 data (hop counts) come from resources/ (copied with the tools). Receiver: LWA1.
# --from-summary reads ../plots/processing_summary.csv (the log next to --output): every signal
# that array processing (sliding/plot delay-Doppler with a beamformer or sky map) gave an
# azimuth/elevation is matched to the stations whose waveform, band, bearing and hop geometry fit,
# and those are drawn with the hop count that elevation gives. Run it after those jobs.
# The signals used and the path table are printed to the .out file.

# Run script: every file with array directions in the summary
python plot_station_map.py --from-summary --min-snr 10 --output ../plots/station_map_all.png --csv ../plots/station_map_all.csv

# One recording, plus every other station using that waveform in the band (grey)
#python plot_station_map.py --from-summary --files 4515 --band 13.5 --bandwidth -99259.377 --sweep-rate 2 --date 2023-12-09 --output ../plots/station_map_4515.png --csv ../plots/station_map_4515.csv

#python plot_station_map.py --from-summary --files 5465 --band 24.55 --bandwidth -150727.203 --sweep-rate 2 --date 2023-12-09 --output ../plots/station_map_5465.png --csv ../plots/station_map_5465.csv

# Stations can still be added by hand: CODE@ELEV (hop count from that elevation, with the ionosonde
# profile at --time for --freq) or CODE:N (N hops)
#python plot_station_map.py --stations PCYC@11.78,FOCF@10.3 --freq 25.6 --time 2023-12-09T16:46:00Z --output ../plots/station_map_manual.png

# Hop table for one station at one elevation (printed only)
#python codar_stations.py --hops FOCF 10.3

# Every plot written to ../plots adds its findings to ../plots/processing_summary.csv;
# print the merged table (one row per signal) at the end of the job
python summary_utils.py ../plots/processing_summary.csv
