# python
''''''"""
    experiment with Fast Fourier Transformation
    
    plot a spectogram-like chart with the main (=strongest) frequencies
"""

from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog

import splib.sound_tools as st
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from splib.sound_tools import read_wav_file
import parselmouth

class G:

    reduce_colors = 5
    freq_low_cut = 60
    loud_window = 1000  #this is in frames

class dialog:
    recording = "hus1h"
    chapter = 114
    max_freq = 350
    winsize = 25  # ms
    pos     = 400  # there is an 'i'
    time_res = 5   # ms
    samples  = 200  #

    layout = """
    title   Play with FFT
    text    recording  recording id, example: 'hus1h'
    int     chapter    koran chapter number
    label   * specify fft parameters *
    int     max_freq   limit for high frequency
    int     winsize    window size for the fft
    label   * specify plot parameters
    int     pos        sample position (ms)
    int     time_res   time resolution ms
    int     samples    number of samples
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    recd, chap, winsize = dialog.recording, dialog.chapter, dialog.winsize
    fname = f"chap_{chap:03d}.wav"
    ifile = cfg.recording / 'source' / fname
    ifile = str(ifile).format(recd=recd)
    print('load file:', ifile)

    sr, wav = read_wav_file(ifile)
    fpms = int(sr/1000)
    print('wave', type(wav), 'len', len(wav), wav)

    #snd = parselmouth.Sound(ifile)
    snd = parselmouth.Sound(wav, sr)

    print(f"samples={dialog.samples} - too many samples will eat your time")
    pos = dialog.pos
    tmslo = pos
    total_span = dialog.samples * dialog.time_res
    tmshi = tmslo + total_span

    seclo = tmslo / 1000
    sechi = tmshi / 1000
    snd_part = snd.extract_part(from_time=seclo, to_time=sechi, preserve_times=True)
    print("snd_part", snd_part)
    print("snd xs()", snd_part.xs())
    print("snd values", snd_part.values)
    print("snd T", snd_part.values.T)


    fig, ax = plt.subplots(2, 1, figsize=(20,8), gridspec_kw={'height_ratios': [3, 1]} )
    fig.tight_layout()
    # fig, ax = plt.subplots()
    colors='red orange gold yellow green blue violet cyan wheat silver'.split()[:G.reduce_colors]
    scatter = []
    for samp in range(dialog.samples):
        pos = dialog.pos + samp * dialog.time_res
        mslo = pos
        mshi = mslo + dialog.winsize

        frlo, frhi = mslo * fpms, mshi * fpms
        # print(f'window ms-range {mslo, mshi} fr-range {frlo, frhi}')
        segm = wav[frlo:frhi]
        print('window in frames',len(segm))

        yf = fft(segm, sr)
        ya = np.abs(yf)
        # now we got the fft output

        # get the {n} highest peeks in the range of upto {hicut} Hz
        highest = []
        maxfreq = dialog.max_freq
        up = False
        for ndx, (v1, v2) in enumerate(zip(ya[:maxfreq], ya[1:maxfreq])):
            if up:
                if v2 < v1:
                    highest.append((v1, ndx))  # value is power, ndx is frequency
                    up = False
            else:
                if v2 > v1:
                    up = True
        highest.sort(reverse=True)
        for (p, freq), cor in zip(highest, colors):  # get power and frequency of the highest peaks
            if freq < G.freq_low_cut:
                continue
            dot = np.sqrt(p*0.005)
            scatter.append((pos, freq, dot, cor))

    p = ax[0]
    plt.title(f'plot_frequency_scatter {recd} {fname} - winsize={winsize} ')


    p.set_yscale('log')
    p.set_yticks([80, 100, 120, 160, 200, 240, 300, 320, 400, 480, 540, 600, 720])

    p.set_yticklabels(['','100','','','200','','300','','400','','','600',''])
    # print(scatter[:3])
    for pos, freq, dot, cor in scatter:
        p.scatter(pos, freq, s=dot, color=cor)

    for y in (100, 200, 300, 400):
        p.hlines(y, tmslo, tmshi, color='silver')

    pitch = snd_part.to_pitch(time_step=0.001)
    pitch_val = pitch.selected_array["frequency"]

    print('pitch val',pitch_val)
    pitch_val[pitch_val == 0] = np.nan
    pitch_xs = pitch.xs() * 1000
    print('pitch_xs',pitch_xs)
    p.plot(pitch_xs, pitch_val, 'o', markersize=2, color='k')

    print("pitch", pitch)

    # plot the parsl data


    # plot the wave data

    #plt.plot(yf)
    frlo, frhi = tmslo * fpms, tmshi * fpms
    print(f'ms-range {tmslo, tmshi} fr-range {frlo,frhi}')

    span = frhi - frlo
    x_scale = np.linspace(tmslo, tmshi, span)
    segm = wav[frlo:frhi]


    # a smoother average from the dialo parameters
    segm = np.abs(segm)
    avg = st.running_mean(segm, G.loud_window)
    y = segm
    p = ax[1]
    p.plot(x_scale, y, color='silver')

    p.plot(x_scale, avg*2, color='red')
    snd_xs = snd_part.xs() * 1000
    snd_val = snd_part.values.T * 10000
    plt.plot(snd_xs, snd_val, linewidth=0.5)

    plt.show()

main()