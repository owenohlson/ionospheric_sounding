"""Cut a raw TBN/WAV recording into sweeps: RDC0 (adc, pulses, channels)."""

import cli

if __name__ == "__main__":
    cli.run(lambda: cli.export_main("rdc0"))
