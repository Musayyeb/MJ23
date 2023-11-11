# python
''''''"""
    experiment with parselmouth attributes
    
    plot a a single block with some attributes
"""

from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog

import splib.sound_tools as st
import splib.attrib_tools as at
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from splib.sound_tools import read_wav_file
from splib.sound_input_lib import get_amplitude_average
import parselmouth
import librosa

class G:
    pass

class dialog:
    recording = "hus1h"
    chapter = 114
    block = 1

    layout = """
    title   Play with Parselmouth
    text    recording  recording id, example: 'hus1h'
    int     chapter    koran chapter number
    int     block      block number
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    recd, chap, blkno = dialog.recording, dialog.chapter, dialog.block

    fr, wav = st.read_wav_block(recd, chap, blkno, fmt='float')
    fr, wavi = st.read_wav_block(recd, chap, blkno, fmt='int16')
    my_ampl, total_loudness = get_amplitude_average(wavi, 10, 2, fpms=24)

    # ampl_rosa is also shifted in the extract_ampl_freq module
    freq_vect, pars_ampl, rosa_ampl = at.load_freq_ampl(recd, chap, blkno)

    bndl = at.load_pyaa_other(recd, chap, blkno)
    # zcr, energy, energy_entropy, spectral_centroid, spectral_spread,
    # spectral_entropy, spectral_flux, spectral_rolloff
    pyaa_zcr = bndl[0]
    pyaa_ampl = bndl[1]  # energy

    snd = parselmouth.Sound(wav, fr)

    fts = snd.to_formant_burg(time_step=0.005, max_number_of_formants=7,
          maximum_formant= 10000.0, window_length=0.025,
          pre_emphasis_from=50.0)
    print("fts",fts)
    band_1 = []
    band_2 = []
    band_3 = []
    band_4 = []
    band_5 = []
    band_6 = []
    band_7 = []
    xs = fts.xs()
    xs2 = np.array(xs) * 1000
    #print("x-scale", xs)
    for t in xs:
        band_1.append(0.2 * fts.get_bandwidth_at_time(formant_number=1,time=t))
        band_2.append(0.04 * fts.get_bandwidth_at_time(formant_number=2, time=t))
        band_3.append(0.04 * fts.get_bandwidth_at_time(formant_number=3, time=t))
        band_4.append(0.04 * fts.get_bandwidth_at_time(formant_number=4, time=t))
        band_5.append(0.04 * fts.get_bandwidth_at_time(formant_number=5, time=t))
        band_6.append(0.04 * fts.get_bandwidth_at_time(formant_number=6, time=t))
        band_7.append(0.04 * fts.get_bandwidth_at_time(formant_number=7, time=t))

    #print("fts band 1", fts.get_bandwidth_at_time(formant_number=1,time=0.200))
    #print("fts band 2", fts.get_bandwidth_at_time(formant_number=2,time=0.200))
    #print("fts band 3", fts.get_bandwidth_at_time(formant_number=3,time=0.200))

    '''
    print(dir(fts))
    for name in dir(fts):
        print(name, type(getattr(fts, name)), getattr(fts, name))
        if not name.startswith('__'):
            try:
                print(name, getattr(fts, name)())
            except Exception as ex:
                print("   excp:",ex)
    ''' # show content of the formant object

    # get the amplitude (RMS) from librosa
    hop = int(fr/200)
    stft = librosa.stft(wav, hop_length=hop)
    S = librosa.magphase(stft) # librosa.stft(wav, window=np.ones, center=False))[0]

    rosa_rms = librosa.feature.rms(S=S[0])
    rosa_rms = interpolate(rosa_rms[0], 0)

    zcr = librosa.feature.zero_crossing_rate(wav, frame_length=int(fr/50) ,hop_length=hop)
    # zcr = librosa.feature.zero_crossing_rate(S=S[0], frame_length=int(fr / 50))

    fig, ax = plt.subplots(1, 1, figsize=(20,8), gridspec_kw={} )
    fig.tight_layout()

    ax.grid()
    # Show the major grid lines with dark grey lines
    ax.grid(which='major', color='#888888')

    # Show the minor grid lines with very faint and almost transparent grey lines
    ax.minorticks_on()
    ax.grid(which='minor', color='#cccccc') #, alpha=0.2)

    # The adjustments of the time scale (+5, +25, -4) are here to get the data synchronized
    # Wherever we read the data, we have to apply exaxt these time adjustments (ms)


    pyaa_zcr = interpolate(pyaa_zcr, 2000, 0)
    pyaa_ampl = interpolate(pyaa_ampl, 600,  0)
    freq_vect = interpolate(freq_vect, 1, 0)
    pars_ampl = interpolate(pars_ampl+100, 0.5, 0)
    rosa_ampl = interpolate(rosa_ampl, 600, 0)
    rosa_zcr = interpolate(zcr[0], 3000, 0)
    band_1 = interpolate(band_1, 3, 0)
    band_2 = interpolate(band_2, 2, 0)
    band_3 = interpolate(band_3, 1, 0)
    band_4 = interpolate(band_4, 1, 0)
    band_5 = interpolate(band_5, 1, 0)
    my_ampl = (my_ampl*0.02)[25:]   # no interpolation



    #ax.plot(pyaa_ampl, color='purple')
    #ax.plot(pyaa_zcr, color='pink')          # zero crossing pyaa
    #ax.plot(rosa_zcr, color='violet')         # zero crossing librosa

    ax.plot(pars_ampl, color="purple")             # librosa rms
    ax.plot(rosa_ampl, color="green")             # librosa rms
    ax.plot(my_ampl, color="black")             # manual averaging ampl

    ax.plot(freq_vect, color="orange")             # freq from parselmouth
    #ax.plot(pars_ampl+300, color="gold")             # praat ampl
    #ax.plot(avg/100, color="peru")
    #ax.plot(rosa_rms*900, color='red')
    #ax.plot(band_1, color="red", linewidth=1)
    ax.plot(band_2, color="blue", linewidth=1)
    ax.plot(band_3, color="teal", linewidth=1)
    #ax.plot(band_4, color="orange", linewidth=1)
    #ax.plot(band_5, color="yellow", linewidth=1)
    # ax.legend("pyaa_amp pyaa_zcr rosa_zcr rosa_ampl my_ampl formant_1".split())
    ax.legend("pars_ampl rosa_ampl my_ampl formant_1 formant_2 formant_3 formant_4 formant_5".split())
    """
    ax.plot(xs2, band_2, color="blueviolet", linewidth=1)
    ax.plot(xs2, band_3, color="blue", linewidth=1)
    ax.plot(xs2, band_4, color="teal", linewidth=1)
    ax.plot(xs2, band_5, color="green", linewidth=1)
    ax.plot(xs2, band_6, color="darkgreen", linewidth=1)
    ax.plot(xs2, band_7, color="olive", linewidth=1)
    """  # plot the formants
    # ax.legend(["freq", "ampl", "B1", "B2", "B3", "B4", "B5", "B6", "B7"])
    plt.show()

    return


def interpolate(vect_5ms, yfact=0, move=0):
    # our data comes at 5ms intervals. For the plotting we want the data in 1 ms interval
    # let numpy do the interpolation
    # eventually apply some data shifting
    # move is the where the vector data is shifted
    #   +10 means insert 10 ms at the beginning
    #   -10 means remove 10 ms from the beginning
    l = len(vect_5ms)
    x      = np.linspace(0, l, num= l * 5)
    points = np.linspace(0, l, num= l)
    vect_1ms = np.interp(x, points, vect_5ms)
    vect_1ms *= yfact
    if move < 0:
        final = vect_1ms[-move:]
    elif move > 0:
        vlen = len(vect_1ms)
        final = np.ndarray((vlen + move,), dtype="float")
        final[move:vlen + move] = vect_1ms
    else:
        final = vect_1ms

    return final


main()