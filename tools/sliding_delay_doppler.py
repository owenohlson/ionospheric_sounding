"""Delay-Doppler video: one frame per integration window, stepped by --hop-time.

Same inputs and options as plot_delay_doppler.py; --output is a directory.
With array beamforming and no --look-az/--look-el the beam is re-aimed each
frame at the strongest signal and the chosen directions are saved to
look_directions.csv.
"""

import cli
import plot_delay_doppler

if __name__ == "__main__":
    cli.run(lambda: plot_delay_doppler.main(video=True))
