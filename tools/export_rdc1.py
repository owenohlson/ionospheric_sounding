"""Dechirp + range FFT: RDC1 (delay, pulses, channels). Input: raw recording or RDC0."""

import cli

if __name__ == "__main__":
    cli.run(lambda: cli.export_main("rdc1"))
