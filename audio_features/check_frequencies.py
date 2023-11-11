# python3
'''
    Frequencies are extracted with the parselmoth/praats library (extract_ampl_freq.py)
    lets read the resulting data and check:
        detect the areas where there is no frequency: write a list of long No-Frequency sections

'''

from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog
from splib.toolbox import get_block_counters
import splib.sound_tools as st
import splib.toolbox as tbx
import splib.attrib_tools as attribs

import numpy as np
from numpy.fft import fft
from splib.sound_tools import read_wav_file
from matplotlib import pyplot as plt
import parselmouth
import librosa
import os

class G:
    pass

class dialog:
    recording = "hus1h"
    chapter = 100

    layout = """
    title   Extract frequency from audio files
    text    recording  recording id, example: 'hus1h'
    text    chapter    koran chapter number or '*'
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    stt = tbx.Statistics()

    # the dialog window may specify a specific block (number)
    # or "*" for all available blocks
    if dialog.chapter == "*":
        sel_chap = '*' # which means all
    else:
        sel_chap = int(dialog.chapter)

    recd = dialog.recording
    blk_counters = get_block_counters(recd)
    print(f"recording: {recd}")


    for chap, blkno, fn, full in tbx.select_files("freq_ampl",
                                    recd=recd, chap=sel_chap, must_exist=False):
        #print(full)
        freq_vect, pars_vect, rosa_vect = attribs.load_freq_ampl(recd, chap, blkno)
        nofreq_start = 0
        hasfreq = True
        for ndx, f in enumerate(freq_vect):
            if f == 0:
                if hasfreq:
                    # switch to no_freq
                    hasfreq = False
                    nofreq_start = ndx
            else:
                if not hasfreq:
                    # switch to freq
                    hasfreq = True
                    silence = ndx - nofreq_start
                    if silence > 200:  # 200 intervals (5ms) => 1 second
                        print(f"long silence {silence*5} at mspos {nofreq_start*5} in {chap}, {blkno}")

main()