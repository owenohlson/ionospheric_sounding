"""Phase-derived Doppler: RDC2P (delay, estimates, channels) of sweep-to-sweep lag products. Input: raw recording, RDC0 or RDC1."""

import cli

if __name__ == "__main__":
    cli.run(lambda: cli.export_main("rdc2p"))
