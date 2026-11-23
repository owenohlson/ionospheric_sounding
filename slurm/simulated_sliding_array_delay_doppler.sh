#!/bin/bash
#SBATCH --job-name=simulated_sliding_array_delay_doppler
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/simulated_sliding_array_delay_doppler.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
#python tbn_sliding_array_delay_doppler.py --rdc2-input=/home/onohlson/projects/def-peterdri/onohlson/data/4747_rdc2_x.h5 --integration-time=120 --hop-time=30 --bandwidth=-99.25937700e3 --sweep-frequency=2 --vmin=50 --vmax=110 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/4747_sliding_array_delay_doppler_120int_cb --framerate=5 --gap-fill=zero --angle-plot-output=/home/onohlson/projects/def-peterdri/onohlson/plots/4747_sliding_array_delay_doppler_120int_cb/cb_angle_plot --beamformer=conventional --d-min=217.5 --d-max=219.5 --carrier-frequency=24.55e6 --pol=x --angle-method=conventional --angle-plot-vmin=-12 --keep-frames --beam-pattern-output=/home/onohlson/projects/def-peterdri/onohlson/plots/4747_sliding_array_delay_doppler_120int_cb/beam_pattern.png

#python tbn_sliding_array_delay_doppler.py --rdc2-input=/home/onohlson/projects/def-peterdri/onohlson/data/4747_rdc2.h5 --integration-time=120 --hop-time=30 --bandwidth=-99.25937700e3 --sweep-frequency=2 --vmin=50 --vmax=90 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/4747_sliding_array_delay_doppler_120int_cb --nfft-doppler=1024 --framerate=5 --gap-fill=zero --angle-plot-output=/home/onohlson/projects/def-peterdri/onohlson/plots/4747_sliding_array_delay_doppler_120int_cb/cb_angle_plot --beamformer=conventional --d-min=217.5 --d-max=219.5 --carrier-frequency=24.55e6 --pol=x --angle-method=conventional --angle-plot-vmin=-15 --keep-frames --beam-pattern-output=/home/onohlson/projects/def-peterdri/onohlson/plots/4747_sliding_array_delay_doppler_120int_cb/beam_pattern.png

#python tbn_sliding_array_delay_doppler.py --rdc2-input ../data/sim_dd_box_rdc2.h5 --output ../plots/sim_sliding_dd_box --angle-plot-output ../plots/sim_sliding_dd_box/sim_angle_video --beam-pattern-output ../plots/sim_sliding_dd_box/sim_beam_pattern.png --beamformer=conventional --pol=x --gap-fill=zero --vmin=-35 --vmax=10 --framerate=10

#python tbn_sliding_array_delay_doppler.py --rdc2-input ../data/sim_dd_angle_rdc2.h5 --output ../plots/sim_sliding_dd_angle --angle-plot-output ../plots/sim_sliding_dd_angle/sim_angle_video --beam-pattern-output ../plots/sim_sliding_dd_angle/sim_beam_pattern.png --beamformer=conventional --pol=x --gap-fill=zero --vmin=-35 --vmax=10 --framerate=10

#python tbn_sliding_array_delay_doppler.py  --rdc2-input ../data/sim_dd_multi_rdc2.h5 --output ../plots/sim_sliding_dd_multi --angle-plot-output ../plots/sim_sliding_dd_multi/sim_angle_video --beam-pattern-output ../plots/sim_sliding_dd_multi/sim_beam_pattern.png --beamformer=conventional --pol=x --gap-fill=zero --vmin=-35 --vmax=10 --framerate=10

python tbn_sliding_array_delay_doppler.py --rdc2-input ../data/sim_dd_amplitude_rdc2.h5 --output ../plots/sim_sliding_dd_amplitude_cb --angle-plot-output ../plots/sim_sliding_dd_amplitude_cb/sim_angle_video --beam-pattern-output ../plots/sim_sliding_dd_amplitude_cb/sim_beam_pattern.png --beamformer=conventional --pol=x --gap-fill=zero --vmin=-35 --vmax=10 --framerate=10

python tbn_sliding_array_delay_doppler.py --rdc2-input ../data/sim_dd_amplitude_rdc2.h5 --output ../plots/sim_sliding_dd_amplitude_mvdr --angle-plot-output ../plots/sim_sliding_dd_amplitude_mvdr/sim_angle_video --beam-pattern-output ../plots/sim_sliding_dd_amplitude_mvdr/sim_beam_pattern.png --beamformer=mvdr --angle-method=mvdr --angle-flat-fallback=none --pol=x --gap-fill=zero --vmin=-35 --vmax=10 --framerate=10 --angle-plot-vmin=-75 --angle-plot-vmax=-45
