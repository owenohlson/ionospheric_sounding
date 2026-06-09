#!/usr/bin/env python3

"""
Example script for splitting a TBN file into smaller pieces.
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime

from lsl.reader import tbn
from lsl.common.progress import ProgressBar
from lsl.misc import parser as aph

from lsl.misc import telemetry
telemetry.track_script()


def split_file(fhIn, fhOut, nCaptures, nAntpols):
    pb = ProgressBar(max=nCaptures)
    
    for c in range(int(nCaptures)):
        for i in range(nAntpols):
            cFrame = fhIn.read(tbn.FRAME_SIZE)
            fhOut.write(cFrame)
            
        pb.inc(amount=1)
        if c != 0 and c % 100 == 0:
            sys.stdout.write(pb.show()+'\r')
            sys.stdout.flush()
            
    sys.stdout.write(pb.show()+'\r')
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(args):
    filename = args.filename
    
    sizeB = os.path.getsize(filename)
    
    # Open the file
    fh = open(filename, 'rb')

    # Align to boundary
    frame = tbn.read_frame(fh)
    stand, pol = frame.id
    skip = 0
    while (2*(stand-1)+pol) != 0:
        frame = tbn.read_frame(fh)
        stand, pol = frame.id
        skip += 1
    fh.seek(fh.tell() - tbn.FRAME_SIZE)

    if skip != 0:
        print(f"Skipped {skip} frames at the beginning of the file")

    boundary_pos = fh.tell()

    # Probe metadata *from the boundary*
    good_pos = None

    for attempt in range(100):

        test_pos = boundary_pos + attempt * tbn.FRAME_SIZE
        fh.seek(test_pos)

        try:
            sample_rate = tbn.get_sample_rate(fh)
            nFramesX, nFramesY = tbn.get_frames_per_obs(fh)

            if 95e3 < sample_rate < 105e3:
                print(f"Valid sample rate: {sample_rate/1e3:.3f} kHz")

                good_pos = test_pos
                break

            else:
                print(f"Invalid sample rate ({sample_rate}), trying again...")

        except Exception as e:
            print(f"Error reading metadata: {e}, trying again...")

    else:
        print("Failed to read valid metadata after 100 attempts, exiting.")
        sys.exit(1)

    # Go back to the GOOD boundary for actual splitting
    fh.seek(good_pos)

    nCaptures = sizeB // tbn.FRAME_SIZE // (nFramesX + nFramesY)
    
    print(f"Filename:    {filename}")
    print(f"Size:        {sizeB/1024**2:.1f} MB")
    print(f"Captures:    {nCaptures} ({nCaptures*512/sample_rate:.3f} seconds)")
    print(f"Stands:      {nFramesX+nFramesY} ({nFramesX} x pol., {nFramesY} y pol.)")
    print(f"Sample Rate: {sample_rate/1e3:.3f} kHz")
    print("===")

    if args.tend is None:
        args.tend = nCaptures * 512 // sample_rate
        nCaptures -= args.tstart * sample_rate // 512
        print(f"No --tend provided, using full duration of file: {args.tend:.3f} seconds")
    else:
        nCaptures = (args.tend - args.tstart) * sample_rate // 512

    nSkip = int(args.tstart * sample_rate / 512)

    print(f"Seconds to skip at start:  {args.tstart:.3f} ({nSkip} captures)")
    print(f"Seconds to trim from end: {args.tend:.3f} ({nCaptures} captures)")
    print(f"New duration after trimming: {nCaptures*512/sample_rate:.3f} seconds")

    # Make sure that the first frame in the file is the first frame of a capture 
    # (stand 1, pol 0).  If not, read in as many frames as necessary to get to 
    # the beginning of a complete capture.
    frame = tbn.read_frame(fh)
    stand, pol = frame.id

    skip = 0
    while (2*(stand-1)+pol) != 0:
        frame = tbn.read_frame(fh)
        stand, pol = frame.id
        skip += 1
    fh.seek(fh.tell() - tbn.FRAME_SIZE)

    if skip != 0:
        print(f"Skipped {skip} frames at the beginning of the file")
    
    for c in list(range(nSkip)):
        if c < nSkip:
            fh.seek(fh.tell() + tbn.FRAME_SIZE*(nFramesX+nFramesY))
            continue
            
    nFramesRemaining = (sizeB - fh.tell()) // tbn.FRAME_SIZE
    nRecursions = int(nFramesRemaining // (nCaptures*(nFramesX+nFramesY)))
    if not args.recursive:
        nRecursions = 1
        
    scale = int(math.log10(nRecursions)) + 1
    ifString = "Working on #%%%ii of %i (%%s)" % (scale, nRecursions)
    
    for r in range(nRecursions):
        if args.date:
            filePos = fh.tell()
            junkFrame = tbn.read_frame(fh)
            fh.seek(filePos)
            
            dt = junkFrame.time.datetime
            captFilename = f"{os.path.splitext(os.path.basename(filename))[0]}_{dt.isoformat()}.dat"
        else:
            captFilename = "%s_s%04i_p%%0%ii.dat" % (os.path.splitext(os.path.basename(filename))[0], (args.tend - args.tstart), scale)
            captFilename = captFilename % r
            if not args.recursive:
                captFilename = f"{os.path.splitext(os.path.basename(filename))[0]}_s{int(args.tend - args.tstart):04d}.dat"
                
        print(ifString % (r+1, captFilename))
        
        t0 = time.time()
        fhOut = open(captFilename, 'wb')
        split_file(fh, fhOut, nCaptures, nFramesX+nFramesY)
        fhOut.close()
        t1 = time.time()
        print(f"  Copied {os.path.getsize(captFilename)} bytes in {t1-t0:.3f} s ({os.path.getsize(captFilename)/1024.0**2/(t1-t0):.3f} MB/s)")
    fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='split a TBN file into several files', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to split')
    parser.add_argument('-s', '--tstart', type=aph.positive_or_zero_float, default=0.0, 
                        help='number of seconds to skip before splitting')
    parser.add_argument('-e', '--tend', type=aph.positive_or_zero_float, default=None, 
                        help='number of seconds to trim from the end of the file')
    parser.add_argument('-d', '--date', action='store_true', 
                        help='label the split files with a date rather than a sequence number')
    parser.add_argument('-r', '--recursive', action='store_true', 
                        help='recursively split the file')
    args = parser.parse_args()
    main(args)
    
