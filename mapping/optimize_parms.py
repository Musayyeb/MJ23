# python3
'''
    Optimize mapping poarameters

    The mapping algorithm, which tries to map letters and predictions
    depends on many parameters

    This code tries to use scipy optimization routines to fin an optimal
    combination of these paramaters.
'''

from splib.cute_dialog import start_dialog
import sys

class dialog:
    recording = "hus9h"
    chapno = 0
    blkno = 0
    avgwin = 5

    layout = """
    title Nullmap - start mapping
    text     recording  recording id, example: 'hus1h'
    int      chapno    Chapter
    int      blkno     Block

    int      avgwin   window size for average probability (slices)
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config
cfg = get_config()

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize, differential_evolution
import functools
import splib.toolbox as tbx
import splib.sound_tools as st
from itertools import product
AD = tbx.AttrDict
import splib.attrib_tools as att
import splib.text_tools as tt
import pickle
import random
import time

class G:
    lc = att.LabelCategories()
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None

vowels = 'aiuAYNW*'

def main():
    colormix()

    stt = tbx.Statistics()
    recd, chapno, blkno = dialog.recording, dialog.chapno, dialog.blkno

    # prediction data selection parameters:

    avg_pred = load_pred_data(dialog.recording, chapno, blkno)

    # prepare stripe for text, loudness and frequency
    freq_vect, pars_ampl, rosa_ampl = att.load_freq_ampl(recd, chapno, blkno)
    freq_vect[freq_vect > 300] = 300  # limit max value to 300 Hz

    pyaa_ampl = interpolate(pars_ampl, 600, 0)
    pyaa_ampl[pyaa_ampl < 0] = 0
    pyaa_ampl = pyaa_ampl / 200
    freq_vect = interpolate(freq_vect, 1, 0)

    xdim = len(freq_vect)  # duration of block determins the x dimension of chart
    xinch = xdim * 0.01   # zoom factor for x-axis (in inches)

    text = tt.get_block_text(recd, chapno, blkno) #, spec="ml")
    text = text.strip('.')

    # calulate relation between text and audio length
    time_factor = (xdim - 800) / len(text)   # milliseconds per letter
    letter_length = (xdim - 800) / (len(text) - 2)

    print("timing relation", int(time_factor))
    # text = adjust_timing(text)

    print(f"average probabilities: {avg_pred.shape}")


    optimize(avg_pred, text, time_factor)


def load_pred_data(recd, chapno, blkno):
    full = cfg.data / dialog.recording / 'probs' / f"{chapno:03d}_{blkno:03d}.npy"
    data = np.load(full)
    print(f"loaded numpy data: {data.shape}")
    return data

def optimize(preds, text, time_factor):
    trigger = 0.05
    limit_search = 100
    after_found = 10
    not_found = 40
    min_len = 25

    avg_tab = {}
    ltr_rows = np.swapaxes(preds, 0, 1)
    for ndx, ltr_seq in enumerate(ltr_rows):

        letter = G.lc.ltrs[ndx]
        avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
        avg_tab[letter] = avg

    optmap_p = functools.partial(optmap, avg_tab, text, time_factor)


    x0 = np.array([trigger, limit_search, after_found, not_found, min_len])
    bounds = np.array([[0.01, 0.3], [10, 100], [0, 50], [10, 60], [10, 40]])

    res = differential_evolution(optmap_p, bounds, disp=True, maxiter=10000)
    final = [round(x,2) for x in res.x]
    print(res)
    print(final)

def optmap(avg_tab, text, time_factor, plist):
    final = [round(x,2) for x in plist]
    print("optmap.plist", final)
    # parmvect is a list of parameters
    trigger, limit_search, after_found, not_found, min_len = list(plist)
    # time_factor is approx. ms per letter
    adjust = time_factor / 250
    after_found  = int(after_found * adjust)
    limit_search = int(limit_search * adjust)
    not_found    = int(not_found * adjust)
    skip_dot     = int(5 * adjust)
    double_ltr   = int(35 * adjust)
    minlen       = int(min_len * adjust)

    rating = 0

    # print(text)
    curpos = 50   # Position is in slices of 5ms !!!
    prev_ltr = ''
    for ndx, ltr in enumerate(text):
        #if ltr in '.~':
        #    curpos += skip_dot
        #    continue

        # double letters are detected only once
        if ltr == prev_ltr:
            #curpos += double_ltr
            continue
        prev_ltr = ltr

        double = ndx + 1  < len(text) and ltr == text[ndx+1]

        # if currpos is too far behind a calculated letter position,
        # then advance the currpos
        calcpos = (ndx * time_factor + 400) / 5 # /5 for slices
        if calcpos - curpos > limit_search:
            # print(f"curpos {curpos*5} too far behind {int(calcpos*5)}", end = ' ')
            curpos = int(calcpos - limit_search/2)
            # print(f" move to {curpos*5}  .. (ms)")

        # print(f"from {curpos*5:5d} ms scan for ltr '{ltr}' ", end=" ")
        seq = avg_tab[ltr]
        for p in range(curpos, min(len(seq)-1, curpos+limit_search)):
            if seq[p] < trigger:   # probability
                continue
            # print(f"found at {p*5:5d}", end=" ")
            rating += 1
            curpos = p
            break
        else:  # not found
            curpos += not_found
            # print()
            continue

        # search end of current letter
        at_least = curpos + minlen + (minlen if double else 0)
        while True:
            if (seq[curpos] > trigger * 2) or curpos < at_least:
                curpos += 1
                continue
            break
        # print(f"end at {curpos * 5: 5d}")

    print("rating:", round(len(text) / rating, 2) )
    return rating


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

def colormix():
    cormix = []
    val = [200, 150, 100, 50]
    for r,g,b in product(val, val, val):
        if 350 <= (r+g+b) <= 500:
            cormix.append((r/255,g/255,b/255))
    print("colormix, found:", len(cormix))
    cortab = {l : c for l, c in zip(G.lc.ltrs, cormix)}
    return cortab

main()
