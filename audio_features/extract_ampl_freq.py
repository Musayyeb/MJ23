# python3
''''''"""
    Extract amplitude and frequency from blocks of audio (.wav) files
    The files have a chapter and a block number in the name.

    Extraction of frequency is done with the Parselmouth_Praat tool.
    The frequency is captured from the to_pitch() method in a 10 ms interval by default
    We will go for a 5 millisecond interval (time_step is the right parameter)

    save the file by numpy.save()
    this is considered more secure and allows for saving complex arrays. It is optimized
    and adds only little overhead (space and time)

"""

from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog
from splib.toolbox import get_block_counters
import splib.sound_tools as st
import splib.toolbox as tbx
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
    chapter = 14

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


    for chap, blk, fn, full in tbx.select_files("blocks",
                                    recd=recd, chap=sel_chap, must_exist=False):
        print(f"process {recd}, {chap}, {blk}")
        print(f"\nLoad: {full}")
        stt.files =+ 1

        fr, wav = st.read_wav_file(full, fmt='float')  # frame rate and data vector (int16)

        sound = parselmouth.Sound(wav, fr)
        print(f"wavelen {len(wav)/fr*1000}")

        # go for frequency (pitch)
        pitch = sound.to_pitch(time_step=0.005)  # 5ms interval
        freq = pitch.selected_array["frequency"]
        # now we have the frequency vector as a numpy array

        # go for amplitude (in decibels)  # the parselmouth version - for now:ignored
        intens = sound.to_intensity(time_step=0.005)
        pars_ampl = intens.values[0]
        pars_ampl = np.insert(pars_ampl, 0, pars_ampl[0])  # insert one 5ms interval ==> shift right 5 ms

        # calculate RMS - frame_length (window) and hop_length (interval) are in frames
        hoplen = int(5 * fr/1000)  # gives a ms interval
        winsize = int(100 * fr/1000)  # ms for window size
        print("total time (s):", len(wav) / fr)

        # librosa seems to shift the output data a bit to the right - here we try to compensate for that
        ampl_vector = librosa.feature.rms(y=wav, frame_length=winsize, hop_length=hoplen)  # Plot the RMS energy

        rosa_ampl = ampl_vector[0]
        rosa_ampl = rosa_ampl[5:]  #remove the first 5 elements ==> shift left 25 ms
        # store it!

        fl, pl, rl = len(freq), len(pars_ampl), len(rosa_ampl)
        print(f"lengths f={fl}, p={pl}, r={rl}")
        # make the arrays the same length (shorten the longer one)
        minlen = min(fl, pl, rl)
        pars_ampl = pars_ampl[:minlen]
        freq      = freq[:minlen]
        rosa_ampl = rosa_ampl[:minlen]

        #combine the 3 arrays
        data = np.stack((freq,  pars_ampl, rosa_ampl))
        print("hstack", data.shape)

        fn = f'freq_ampl_{chap:03d}_{blk:03d}'
        path = str(cfg.attribs / "freq_ampl").format(recd=recd)
        full = os.path.join(path, fn)

        np.save(full, data)
        print(f"Created {full}")
        """
        to retrieve the data vectors, just use np.load() :
            b = np.load(full)
            pitch, ampl = b
            
            plt.plot(pitch)
            plt.plot(ampl)
            plt.show()
        """

    print("runtime",stt.runtime())


main()