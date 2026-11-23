#!/bin/bash
#SBATCH --job-name=sliding_phase_doppler_all
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/sliding_phase_doppler_all.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onohlson@gmail.com

# Virtual environment
source ~/projects/def-peterdri/onohlson/lwa_env/bin/activate

# Run script
python tools/tbn_sliding_phase_doppler.py data/060287_010274515_s0896.dat --bandwidth=-99.25937700e3 --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/4515_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010274635_s0895.dat --bandwidth=-99.25937700e3 --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/4635_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010274747_s0895.dat --bandwidth=-99.25937700e3 --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/4747_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010274866_s0895.dat --bandwidth=-150.72720300e3 --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/4866_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010274978_s0895.dat --bandwidth=-220.57639848e3 --sweep-frequency=4 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/4978_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010275122_s0895.dat --bandwidth=-99.25937700e3 --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/5122_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010275234_s0895.dat --bandwidth=-49.62968800e3 --sweep-frequency=4 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/5234_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010275353_s0895.dat --bandwidth=-99.25937700e3 --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/5353_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010275465_s0895.dat --bandwidth=-150.72720300e3 --sweep-frequency=2 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/5465_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

python tools/tbn_sliding_phase_doppler.py data/060287_010275584_s0895.dat --bandwidth=-220.5763984e3 --sweep-frequency=4 --output=/home/onohlson/projects/def-peterdri/onohlson/plots/5584_sliding_phase_doppler --gap-fill=zero --vmin=60 --vmax=110

