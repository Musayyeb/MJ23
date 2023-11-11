# python
"""
    calculate average amplitude for millisecond intervals
    plot the amplitude curve over a whole chapter (does not work
        for large chapters)
    Calculation of block boundaries is moved to 'find_blocks_in_source.py'
"""

from config import get_config
cfg = get_config()
import splib.sound_tools as st
from splib.cute_dialog import start_dialog
import splib.toolbox as tbx
import numpy as np
from matplotlib import pyplot as plt
from splib.sound_input_lib import get_amplitude_average


class dialog:
    recording = "hus1h"
    chapter = 0
    winsize   = 350
    max_ampl = 200
    min_dura = 400

    layout = """
    title   Convert video file into wav format
    text    recording  recording id, example: 'hus1h'
    int     chapter    koran chapter number
    int     winsize    window size (ms) for the average
    int     max_ampl   maximum amplitude in a gap
    int     min_dura   minimum duration of a gap
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    stt = tbx.Statistics()

    fname = f"chap_{dialog.chapter:03d}.wav"
    ifile = cfg.recording / 'source' / fname
    ifile = str(ifile).format(recd=dialog.recording)
    print(f"load {ifile}")

    fr, wav = st.read_wav_file(ifile)

    print(f"got wav, size={len(wav)} frames")
    print(wav.shape)

    win = dialog.winsize
    avg, total_loudness = get_amplitude_average(wav, win, iterations=2)


    # find_gaps(avg, dialog.max_ampl, dialog.min_dura)

    print(stt.runtime())

    avg = np.clip(avg, 0, 800)  # limit y-axis
    curves = ((avg, "royalblue", str(win)), )
    plot(curves, len(avg),  sample_rate=1000, title=fname)


def plot(curves, plotlen, sample_rate, title):

    time_axis = np.linspace(start=0,
                stop=plotlen / sample_rate,
                num=plotlen)
    # Set up plot
    f, ax = plt.subplots(figsize=(18, 4))  # Setup the title and axis titles
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')  # Add the audio data to the plot
    for level in range(0,500,100):
        plt.plot(time_axis, np.array([level]*plotlen), linewidth=0.5, color='black')

    # plt.plot(time_axis, loudness, linewidth=0.3, color='orange')
    for data, color, lbl in curves:
        plt.plot(time_axis, data, linewidth=1.5, color=color, label=lbl)
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()