#!/bin/bash
#SBATCH --job-name=sliding_phase_delay_doppler
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/sliding_phase_delay_doppler.out
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
python sliding_phase_delay_doppler.py ../data/060287_010274515_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010274515_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274515_x_APDD_mvdr --angle-output ../plots/060287_010274515_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010274635_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010274635_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274635_x_APDD_mvdr --angle-output ../plots/060287_010274635_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010274747_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010274747_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274747_x_APDD_mvdr --angle-output ../plots/060287_010274747_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010274866_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010274866_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274866_x_APDD_mvdr --angle-output ../plots/060287_010274866_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010274978_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010274978_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274978_x_APDD_mvdr --angle-output ../plots/060287_010274978_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275122_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010275122_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275122_x_APDD_mvdr --angle-output ../plots/060287_010275122_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275234_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010275234_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275234_x_APDD_mvdr --angle-output ../plots/060287_010275234_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275353_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010275353_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275353_x_APDD_mvdr --angle-output ../plots/060287_010275353_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275465_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010275465_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275465_x_APDD_mvdr --angle-output ../plots/060287_010275465_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10 

python sliding_phase_delay_doppler.py ../data/060287_010275584_rdc2p.h5 --pol x --beamformer mvdr --calibration ../data/060287_010275584_x_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275584_x_APDD_mvdr --angle-output ../plots/060287_010275584_x_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

# y 
python sliding_phase_delay_doppler.py ../data/060287_010274515_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010274515_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274515_y_APDD_mvdr --angle-output ../plots/060287_010274515_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10 

python sliding_phase_delay_doppler.py ../data/060287_010274635_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010274635_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274635_y_APDD_mvdr --angle-output ../plots/060287_010274635_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010274747_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010274747_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274747_y_APDD_mvdr --angle-output ../plots/060287_010274747_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010274866_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010274866_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274866_y_APDD_mvdr --angle-output ../plots/060287_010274866_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010274978_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010274978_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010274978_y_APDD_mvdr --angle-output ../plots/060287_010274978_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275122_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010275122_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275122_y_APDD_mvdr --angle-output ../plots/060287_010275122_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275234_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010275234_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275234_y_APDD_mvdr --angle-output ../plots/060287_010275234_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275353_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010275353_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275353_y_APDD_mvdr --angle-output ../plots/060287_010275353_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275465_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010275465_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275465_y_APDD_mvdr --angle-output ../plots/060287_010275465_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10

python sliding_phase_delay_doppler.py ../data/060287_010275584_rdc2p.h5 --pol y --beamformer mvdr --calibration ../data/060287_010275584_y_cal.h5 --vmin 120 --vmax 70 --output ../plots/060287_010275584_y_APDD_mvdr --angle-output ../plots/060287_010275584_y_APDD_mvdr/angle_plot --angle-cells 0 --angle-method mvdr --angle-range 10
