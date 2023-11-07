# python3
"""
    sound_tools.py

    Collection of functions to process audio files
"""

from config import get_config
cfg = get_config()

import numpy as np
import os
import soundfile


def block_filename(recd, chap, blkno):
    fname = f"b_{chap:03d}_{blkno:03d}.wav"
    ifile = cfg.recording / 'blocks' / fname
    return str(ifile).format(recd=recd)

def read_wav_block(recd, chap, blkno, fmt="int16"):
    full = block_filename(recd, chap, blkno)
    return read_wav_file(full, fmt)

def write_wav_block(wav, recd, chap, blkno, framerate):
    full = block_filename(recd, chap, blkno)
    write_wav_file(wav, full, framerate)
    return full

def write_wav_file(wav, filename, framerate):
    soundfile.write(filename, wav, framerate)
    return

def read_wav_file(filename, fmt="int16"):
    # return the wave vector from a wav file as a numpy array
    try:
        data, fr = soundfile.read(filename, dtype=fmt)
    except Exception as ex:
        print(f"st.read_wav_file soundfile read: {ex}")
        raise FileNotFoundError(filename)
    return fr, data


def running_mean(wav, winsize, iterations=1, offs=None):
    # calculating the loudness of an audio, b
    # x is the input vector, N = the window size in frames
    # This is about 8 times faster than my floating_avgf function
    # Size of output is exactly size of input
    # the output curve shifts slightly to the left, so we prepend a buffer of zeroes
    x, N = wav, winsize
    olen = len(x)
    for _ in range(iterations):
        if offs is None:
            offs = int(winsize/2)
        if offs:
            x = np.append(np.zeros(offs), x)
            x = np.convolve(x, np.ones((N,)) / N)[(N - 1):N-1+olen]
    return x


if __name__ == "__main__":
    print(f"this module {__file__} is for importing")