"""Doppler FFT: RDC2 (frames, delay, doppler, channels), keeping the RDC1 it came from. Input: raw recording, RDC0, RDC1, or an RDC2 that retained RDC1."""

import cli

if __name__ == "__main__":
    cli.run(lambda: cli.export_main("rdc2"))
