#!/bin/bash
#SBATCH --job-name=slurm_all
#SBATCH --account=def-peterdri
#SBATCH --output=/project/6003738/onohlson/slurm/out/slurm_all.out
#SBATCH --open-mode=truncate
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onohlson@gmail.com

# Every plot for every 2023-12-09 file, one file after another, then the station maps. One job,
# one log (slurm/out/slurm_all.out, replaced each run).
#
#   sbatch slurm_all.sh                                         # all ten files
#   sbatch --export=ALL,FILES="060287_010274515 060287_010275465" slurm_all.sh     # only these
#   sbatch --export=ALL,SKIP_EXISTING=0 slurm_all.sh            # redo steps that finished earlier
#   sbatch --export=ALL,REUSE=0 slurm_all.sh                    # make new RDC1s / calibrations, not reuse old ones
#
# Steps that finished in an earlier run are skipped, so if the job hits its time limit, submit it
# again and it carries on where it stopped. (slurm/out must exist: mkdir -p slurm/out.)
#
# Per file:
#   1. spectrogram of the first seconds (raw TBN, first stand)
#   2. scan_lfm -> which waveforms (bandwidth, sweep rate) are present, their echo delays and SNR
#   3. for each strong waveform (best echo SNR >= MIN_SNR, or >= MIN_SNR_UNTABLED for a waveform in no
#      station's table; at most MAX_SIGNALS per file):
#      a. RDC1 for that waveform (matched filter, gating sidebands of its strongest echo removed: RANGE_METHOD,
#         SUPPRESS_SIDEBANDS), both pols, all good stands, only the delays from WINDOW_BEFORE_MS ahead of
#         its strongest echo to WINDOW_AFTER_MS after it (its other modes; echoes outside are left out and
#         listed in the log; hand-set windows go in WINDOWS below, and steps a-e run once per window):
#         ../data/<file>_rdc1_<tag>_<window>.h5, e.g. 060287_010274515_rdc1_bw99p259_fsweep2_d14p2-18p2ms.h5
#         (tag = bandwidth and sweep rate, window = delays held; see rdc_info.py --rename) -- unless an
#         RDC1 / RDC2 already in ../data has that waveform and those echoes, made the same way (it is used instead; the plots
#         still show only the delay window, clipped to what the file holds)
#      b. power-delay profile (all stands, both pols, NAVG_S seconds of sweeps per column)
#      c. Doppler vs time (all stands, both pols)
#      d. x-pol self-calibration: ../data/<file>_x_cal_<tag>.h5 (reused if present; else any other x_cal of the file)
#      e. sliding array delay-Doppler (x pol, calibrated) over the delay window: MVDR sky maps, one MVDR beam
#         (panel) per sky-map peak (up to LOOKS), and station identification of each peak (their directions
#         go to the summary for the station map)
#      Plots are named <file>_<plot>_<tag>, or <file>_<plot>_<tag>_<window> when a waveform has several windows.
# Then: the station maps from the summary's array directions (always redrawn), and the merged summary.
# A failed step is reported and the steps that need its output are skipped; the rest carry on.

set -uo pipefail

FILES_ALL="060287_010274515 060287_010274635 060287_010274747 060287_010274866 060287_010274978
060287_010275122 060287_010275234 060287_010275353 060287_010275465 060287_010275584"
FILES=${FILES:-$FILES_ALL}

# ---------------------------------------------------------------- settings
RAW=/home/onohlson/projects/def-peterdri/LWA_Data/2023-12-09
DATA=../data                                     # preprocessed .dat in, RDC1 / calibration files out
PLOTS=../plots/2023-12-09                        # <PLOTS>/<file>/<file>_<plot>[_<tag>].png
SCAN_STANDS=1,2,20,60,100,150,200,250            # scan_lfm: 8 stands x 2 pols
SCAN_TSTART=60                                   # scan_lfm: seconds of the file used
SCAN_DURATION=20
SPEC_SECONDS=5                                   # spectrogram: first seconds
MIN_SNR=15                                       # dB: a waveform's best echo must reach this ...
MIN_SNR_UNTABLED=25                              # ... or this, if it matches no station's table (noise picks)
MAX_SIGNALS=3                                    # ... and at most this many waveforms per file
WINDOW_BEFORE_MS=1.5                             # RDC1 delay window: strongest echo -1.5 ms ... (E / F1 arrive
WINDOW_AFTER_MS=2.5                              # ... to +2.5 ms (high rays, 2-3 hops)      a few 0.1 ms early)
# Hand-set delay windows (any width), one "file  tag  d_min_ms  d_max_ms" per line. The tag may be
# just its start (bw51 matches bw51p468_fsweep2: see the '#####' lines of slurm_all.out). A waveform
# listed here gets only these windows; repeat a line with other delays for more of them.
WINDOWS="
# 060287_010274515  bw51  12  30
"
RANGE_METHOD=matched                             # RDC1 range compression: matched (matched filter against the whole
                                                 # sweep: every delay compresses fully) or dechirp (the older files)
SUPPRESS_SIDEBANDS=1                             # 1: remove the gating-sideband comb of the waveform's strongest echo
SIDEBAND_BLOCK=20                                #    while exporting (--suppress-sidebands; needs matched); sweeps per block
NAVG_S=2                                         # PDP: seconds of sweeps per column (4 sweeps at 2 Hz)
CAL_INT=10                                       # calibration: integration / hop [s]
DD_INT=120                                       # sliding delay-Doppler: integration / hop [s]
DD_HOP=30
DD_VMIN=0                                        # colour range of the delay-Doppler frames [dB above each
DD_VMAX=45                                       # frame's median]; fixed so video frames compare (Peter: 0-45)
BEAMFORMER=mvdr                                  # delay-Doppler beams: mvdr (sharper, suppresses the other arrivals)
ANGLE_METHOD=mvdr                                # sky maps (the DoA in the summary): mvdr gives each arrival its own
ANGLE_STEP=1                                     # sharp peak (1 deg grid: mvdr peaks are narrow)
LOOKS=3                                          # up to this many sky-map peaks, within PEAK_DB of the strongest:
PEAK_DB=10                                       # each gets a delay-Doppler beam (panel), a station name and a DoA row
SKIP_EXISTING=${SKIP_EXISTING:-1}                # 1: keep outputs from an earlier run
REUSE=${REUSE:-1}                                # 1: use matching RDC1/RDC2 and x-pol calibrations already in
                                                 # DATA (python rdc_info.py --find; REUSE=0 makes new ones)
REUSE_COVER=${REUSE_COVER:-all}                  # an old RDC must hold: all the waveform's echoes (all), or only
                                                 # its strongest (strongest: more reuse, weaker echoes may be cut)

# ---------------------------------------------------------------- environment
unset PYTHONPATH
unset PYTHONHOME
module purge             # the 'modules were not unloaded' notice it prints is normal (sticky modules)
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load gsl
module load fftw
module load hdf5
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
source /project/6003738/onohlson/lwa_env/bin/activate
cd /project/6003738/onohlson/tools
export LWA_SSMIF=$(pwd)/resources/ssmif/LD021_0001.tgz   # shipped in tools/resources
mkdir -p "$PLOTS"
export LFM_SUMMARY=$(realpath "$PLOTS")/processing_summary.csv    # one summary for every plot below

FAILED=()
SKIPPED=()
DONE=$PLOTS/.done                                # markers of finished steps (an output left by a killed
mkdir -p "$DONE"                                 # or failed step is redone, not trusted)
step() {        # step NAME OUTPUT command...   (skips when OUTPUT finished earlier and SKIP_EXISTING=1)
    local name=$1 out=$2
    shift 2
    local mark=$DONE/$(echo "$out" | tr '/ ' '__')
    if [[ "$SKIP_EXISTING" == 1 && -e "$out" && -e "$mark" ]]; then
        echo "--- $name: $out done earlier, skipped"
        return 0
    fi
    echo "=== $name  ($(date '+%F %T'))"
    echo "    $*"
    rm -f "$mark"
    local rc=0
    "$@" || rc=$?
    if (( rc == 0 )); then
        touch "$mark"
        return 0
    fi
    if (( rc == 3 )); then                       # exit 3: nothing to do for this input (not an error)
        echo "--- $name: skipped (nothing to do)"
        SKIPPED+=("${CURRENT:-}$name")
        return 0
    fi
    echo "!!! $name FAILED"
    FAILED+=("${CURRENT:-}$name")
    return 1
}

# ---------------------------------------------------------------- one file
process_file() {
    local F=$1 OUT IN line BW RATE DMIN DMAX TAG SNR ECHOES WIN LABEL NAVG RDC1 CAL KEEP OTHER fid t lo hi
    local -a FIND RANGE HAND=()
    OUT=$PLOTS/$F
    mkdir -p "$OUT"
    IN=$DATA/${F}_preprocessed.dat                   # gap-checked copy if there is one, else the raw TBN
    [[ -e "$IN" ]] || IN=$RAW/$F
    echo "File $F (input $IN)"

    # 1. spectrogram
    step "spectrogram" "$OUT/${F}_spectrogram.png" \
        python plot_spectrogram.py "$RAW/$F" --tstart 0 --tend "$SPEC_SECONDS" --output "$OUT/${F}_spectrogram.png"

    # 2. waveform scan
    step "scan_lfm" "$OUT/${F}_lfm_scan.json" \
        python scan_lfm.py "$RAW/$F" --stands "$SCAN_STANDS" --pol both --tstart "$SCAN_TSTART" \
        --duration "$SCAN_DURATION" --output "$OUT/${F}_lfm_scan.png" --json "$OUT/${F}_lfm_scan.json"
    if [[ ! -e "$OUT/${F}_lfm_scan.json" ]]; then
        echo "No scan result for $F: nothing else to do for it"
        return
    fi

    # 3. per strong waveform
    while read -r fid t lo hi _; do
        [[ "$fid" == "$F" ]] && HAND+=(--window "$t:$lo:$hi")
    done <<< "$(grep -v '^[[:space:]]*#' <<< "$WINDOWS" || true)"
    mapfile -t WAVES < <(python strong_waveforms.py "$OUT/${F}_lfm_scan.json" --max "$MAX_SIGNALS" \
                         --min-snr "$MIN_SNR" --min-snr-untabled "$MIN_SNR_UNTABLED" \
                         --before-ms "$WINDOW_BEFORE_MS" --after-ms "$WINDOW_AFTER_MS" ${HAND[@]+"${HAND[@]}"})
    echo "${#WAVES[@]} delay window(s) of strong waveforms in $F (best echo SNR >= $MIN_SNR dB, $MIN_SNR_UNTABLED dB if in no table):"
    printf '    %s\n' "${WAVES[@]}"

    for line in "${WAVES[@]}"; do
        IFS=$'\t' read -r BW RATE DMIN DMAX TAG SNR ECHOES WIN LABEL <<< "$line"
        NAVG=$(python -c "print(max(1, round($NAVG_S * $RATE)))")
        RDC1=$DATA/${F}_rdc1_${TAG}_${WIN}.h5          # names as in rdc_info.py (--rename)
        CAL=$DATA/${F}_x_cal_${TAG}.h5
        echo "##### $F $LABEL: bandwidth $BW Hz, $RATE Hz sweeps, delays $DMIN-$DMAX ms, echoes at $ECHOES ms, SNR $SNR dB"

        # an RDC1/RDC2 already in DATA with this waveform, the delays and x pol is used as it is
        OLD=""
        KEEP=$ECHOES                                 # echoes an old file must hold to be reused
        [[ "$REUSE_COVER" == strongest ]] && KEEP=${ECHOES%%,*}
        FIND=(--echoes "$KEEP")
        [[ "$ECHOES" == - ]] && FIND=(--d-min "$DMIN" --d-max "$DMAX")      # hand-set window without echoes
        # (and made the same way: RANGE_METHOD, sidebands cleaned or not, as set above)
        RANGE=(--range-method "$RANGE_METHOD")
        if [[ "$SUPPRESS_SIDEBANDS" == 1 ]]; then
            RANGE+=(--suppress-sidebands --sideband-block "$SIDEBAND_BLOCK")
            [[ "$ECHOES" != - ]] && RANGE+=(--sideband-echo "${ECHOES%%,*}")    # this waveform's strongest echo
        fi
        [[ "$REUSE" == 1 ]] && OLD=$(python rdc_info.py --find "$DATA" --file "$F" --bandwidth "$BW" \
            --sweep-rate "$RATE" "${FIND[@]}" --pol x --range-method "$RANGE_METHOD" \
            --sidebands "$([[ "$SUPPRESS_SIDEBANDS" == 1 ]] && echo cleaned || echo kept)")
        if [[ -n "$OLD" ]]; then
            echo "--- RDC1 $LABEL: reusing $OLD"
            RDC1=$OLD
        else
            # a file of this name made another way (e.g. an older dechirped export) is made again
            [[ -e "$RDC1" ]] && echo "--- RDC1 $LABEL: $RDC1 was made another way; exporting it again" \
                && rm -f "$DONE/$(echo "$RDC1" | tr '/ ' '__')"
            step "RDC1 $LABEL" "$RDC1" \
                python export_rdc1.py "$IN" --output "$RDC1" --sweep-frequency "$RATE" --bandwidth="$BW" \
                --pol both --bad-stands status --gap-fill zero --d-min "$DMIN" --d-max "$DMAX" "${RANGE[@]}" || continue
        fi
        # the stand gains do not depend on the waveform: an earlier x-pol calibration of this file is kept
        # (this waveform's first, else the newest other one)
        if [[ "$REUSE" == 1 && ! -e "$CAL" ]]; then
            OTHER=$(ls -t "$DATA/${F}"_x_cal_*.h5 "$DATA/${F}_x_cal.h5" 2>/dev/null | head -1)
            [[ -n "$OTHER" ]] && CAL=$OTHER
        fi

        step "PDP $LABEL" "$OUT/${F}_pdp_${LABEL}.png" \
            python plot_power_delay_profile.py "$RDC1" --stands all --d-min "$DMIN" --d-max "$DMAX" --navg "$NAVG" \
            --output "$OUT/${F}_pdp_${LABEL}.png"

        step "Doppler vs time $LABEL" "$OUT/${F}_doppler_time_${LABEL}.png" \
            python plot_doppler_time.py "$RDC1" --stands all --gap-fill zero --d-min "$DMIN" --d-max "$DMAX" \
            --output "$OUT/${F}_doppler_time_${LABEL}.png"

        if [[ "$REUSE" == 1 && -e "$CAL" ]]; then
            echo "--- calibration x $TAG: using $CAL"
        else
            step "calibration x $TAG" "$CAL" \
                python calibrate_array.py "$RDC1" --pol x --integration-time "$CAL_INT" --hop-time "$CAL_INT" \
                --output "$CAL" --csv "${CAL%.h5}.csv" || continue
        fi

        step "sliding array delay-Doppler x $LABEL" "$OUT/${F}_x_ADD_${BEAMFORMER}_${LABEL}" \
            python sliding_delay_doppler.py "$RDC1" --pol x --beamformer "$BEAMFORMER" --calibration "$CAL" \
            --integration-time "$DD_INT" --hop-time "$DD_HOP" --d-min "$DMIN" --d-max "$DMAX" --vmin "$DD_VMIN" --vmax "$DD_VMAX" \
            --output "$OUT/${F}_x_ADD_${BEAMFORMER}_${LABEL}" --angle-output "$OUT/${F}_x_ADD_${BEAMFORMER}_${LABEL}/angle_plot" \
            --angle-cells 0 --angle-method "$ANGLE_METHOD" --angle-step "$ANGLE_STEP" --angle-range-db 10 \
            --angle-peaks "$LOOKS" --looks "$LOOKS" --angle-peak-db "$PEAK_DB" --codar-stations
    done
}

# ---------------------------------------------------------------- every file, then the maps
for F in $FILES; do
    echo
    echo "################################################################ $F  ($(date '+%F %T'))"
    CURRENT="$F "
    process_file "$F"
done
CURRENT=""

echo
echo "################################################################ station maps  ($(date '+%F %T'))"
SKIP_EXISTING=0                                  # the summary may have new directions: always redraw
step "station map (all files)" "$PLOTS/station_map_all.png" \
    python plot_station_map.py --from-summary --min-snr 10 \
    --output "$PLOTS/station_map_all.png" --csv "$PLOTS/station_map_all.csv"
for F in $FILES; do
    step "station map $F" "$PLOTS/$F/${F}_station_map.png" \
        python plot_station_map.py --from-summary --files "$F" --min-snr 10 --skip-empty \
        --output "$PLOTS/$F/${F}_station_map.png" --csv "$PLOTS/$F/${F}_station_map.csv"
done
python summary_utils.py "$LFM_SUMMARY"

echo
if (( ${#SKIPPED[@]} )); then
    echo "Skipped ${#SKIPPED[@]} step(s) with nothing to do: ${SKIPPED[*]}"
fi
if (( ${#FAILED[@]} )); then
    echo "Finished with ${#FAILED[@]} failed step(s): ${FAILED[*]}"
    exit 1
fi
echo "Finished"
