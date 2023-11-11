# python3
'''
    just show the attrib curves for a block
'''
import splib.sound_tools as st
from splib.cute_dialog import start_dialog
from matplotlib import pyplot as plt
import splib.attrib_tools as attribs
import numpy as np
from splib.sound_input_lib import get_amplitude_average



class dialog:
    recording = "hus1h"
    chapter = 100
    blkno = 0

    layout = """
    title   show plottings of audio attributes
    text    recording  recording id, example: 'hus1h'
    text    chapter    koran chapter number or '*'
    text    blkno      block number
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    recd, chap, blkno = dialog.recording, int(dialog.chapter), int(dialog.blkno)
    fr, wav = st.read_wav_block(recd, chap, blkno)

    avg, avgampl = get_amplitude_average(wav, 100, 2, fpms=24)

    freq_vect, pars_vect, rosa_vect = attribs.load_freq_ampl(recd, chap, blkno)
    #_, pyaa_vect, _, _, _, _, _, _ = attribs.load_pyaa_other(recd, chap, blkno)
    freq_vect *= 30
    rosa_vect *= 15000
    #pyaa_vect *= 25000
    pars_vect *= 50
    pars_vect[pars_vect < 0] = 0
    freqx = np.linspace(0, len(avg), len(freq_vect))
    #pyaax = np.linspace(0, len(avg), len(pyaa_vect))
    avg, = plt.plot(avg, linewidth=1, label='average')
    fv, = plt.plot(freqx, freq_vect, linewidth=1, label='freq')
    rv, = plt.plot(freqx, rosa_vect, linewidth=1, label='rosa_ampl')
    pv, = plt.plot(freqx, pars_vect, linewidth=1, label='pars_ampl')
    #yv, = plt.plot(pyaax, pyaa_vect, linewidth=1, label='pyaa_ampl')

    # line1, = ax.plot([1, 2, 3], label='label1')
    # line2, = ax.plot([1, 2, 3], label='label2')
    plt.legend(handles=[avg,fv,rv,pv])  #,yv])

    plt.show()
    return

main()