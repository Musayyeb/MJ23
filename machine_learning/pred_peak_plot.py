# python3
'''
    Machine predictions - plot the result

    plot the predictions along with some audio attributes
'''

from splib.cute_dialog import start_dialog
import sys

class dialog:
    recording = "hus9h"
    chapno = 0
    blkno = 0

    iters = 0
    avgwin = 5
    savename = ''

    layout = """
    title  Predictions and plotting
    text     recording  recording id, example: 'hus1h'
    int      chapno    Chapter
    int      blkno     Block

    label specs for input data
    text     savename name of the saved model
    int      iters    model iterations
    int      avgwin   window size for average probability (slices)
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config
cfg = get_config()

# from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from prepare_model import get_model
from ml_data import get_pred_data
from matplotlib import pyplot as plt
import numpy as np
import splib.toolbox as tbx
AD = tbx.AttrDict
import splib.sound_tools as st
from itertools import product
import splib.attrib_tools as att
import splib.text_tools as tt
from ml_tools import PredLoader
import pickle
import random
import time

class G:
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None

vowels = 'aiuAYNW*'

def main():
    print("letter categories label:", G.lc.label)
    print("letter categories categ:", G.lc.categ)

    make_letter_groups()

    print("letter_groups:", G.ltr_groups)

    recd, chapno, blkno = dialog.recording, dialog.chapno, dialog.blkno
    pred_loader = PredLoader(recd, chapno, blkno)

    name = dialog.savename

    # prepare stripe for text, loudness and frequency
    freq_vect, pars_ampl, rosa_ampl = att.load_freq_ampl(recd, chapno, blkno)
    freq_vect[freq_vect > 300] = 300  # numpy: limit max value to 300 Hz

    rosa_ampl = interpolate(rosa_ampl, 1000, 0)

    pyaa_ampl = interpolate(pars_ampl, 600, 0)
    pyaa_ampl[pyaa_ampl < 0] = 0
    pyaa_ampl = pyaa_ampl / 200
    freq_vect = interpolate(freq_vect, 1, 0)

    xdim = len(freq_vect)  # duration of block determins the x dimension of chart

    xinch = xdim * 0.005   # zoom factor for x-axis (in inches) - image has to stay below 32k pixels

    text = tt.get_block_text(recd, chapno, blkno) #, spec="ml")
    text = text.replace('.', '')
    text = adjust_timing(text)  # make text more linear in time

    texttab = []
    textpos = np.linspace(430, xdim-400, len(text))
    for l, p in zip(text, textpos):
        texttab.append((l, p))

    # start plotting
    avg_plot = 0
    # reserve two stripes for every iteration (model) - one for the original, other for modified predictions
    stripes = dialog.iters * 2 + avg_plot + 1

    dimy = stripes * 2.5 + 2
    fig, ax = plt.subplots(stripes, 1, figsize=(xinch, dimy), dpi=120, gridspec_kw={})
    print(f"plot stripes:", ax, len(ax))

    fig.tight_layout()
    G.colors = colormix()

    text_stripe = ax[0]  # stripe for loudness frequeny and text

    text_stripe.plot(rosa_ampl, color="peru", linewidth=0.8)  # librosa rms
    text_stripe.plot(pyaa_ampl, color="purple", linewidth=0.8)  #
    text_stripe.plot(freq_vect, color="green", linewidth=0.8)  #
    text_stripe.set_xlim(0, xdim)
    for l, p in texttab:
        text_stripe.text(p, 20, l, color="black", fontsize=20)


    allpreds = []  # collect predictions from all model instances
    group_preds = []

    for iter in range(dialog.iters):

        preds = pred_loader.get_calculated_predictions(name, iter)

        allpreds.append(preds)

        #extended = add_group_probabilities(preds)
        #group_preds.append(extended)

    #allpreds.extend(group_preds)


    if avg_plot:
        # calcualting the average over all predicitions
        avg_pred = np.mean(allpreds, axis=0)  # average predictions

        avg_stripe = ax[1]             # second stripe for the average of the remaining stripes
        avg_stripe.set_ylim(0, 140)
        avg_stripe.set_xlim(0, xdim)

        print(f"average probabilities: {avg_pred.shape}")

        plot_pred(avg_pred, avg_stripe)      # plot the average

        full = cfg.data / dialog.recording / 'probs' / f"{chapno:03d}_{blkno:03d}.npy"
        np.save(full, avg_pred)


    # Plot the individual predictions
    for ndx, pred in enumerate(allpreds):
        print(f"access the stripes, ndx={ndx}")
        stripe_ndx = ndx*2+1+avg_plot
        stripe = ax[stripe_ndx]

        stripe.set_ylim(0, 150)
        stripe.set_xlim(0, xdim)
        for l, p in texttab:
            stripe.text(p, 135, l, color="black", fontsize=16)

        ltr_segments = prep_pred_base(pred)  # full display

        plot_pred(ltr_segments, stripe)

        ltr_segments = find_peaks(pred, text, xdim)     # reduced display

        stripe = ax[stripe_ndx + 1]

        stripe.set_ylim(0, 150)
        stripe.set_xlim(0, xdim)
        for l, p in texttab:
            stripe.text(p, 135, l, color="black", fontsize=16)

        plot_pred(ltr_segments, stripe)

    # save final chart

    chart_name = f"{chapno:03d}_{blkno:03d}_{name}.png"
    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print(f"chart is saved as {chart}")
    #plt.show()

def adjust_timing(text):
    # make text more linear (in time) by inserting extra letters
    newtxt = []
    for l1, l2 in zip(text, text[1:]):
        newtxt.append(l1)
        if isvow(l1) and isvow(l2):
            newtxt.append(l1)
        if iscons(l1) and iscons(l2):
            newtxt.append(l1)

    newtxt.append(l2)
    newtxt = ''.join(newtxt)
    newtxt = newtxt.replace('.', '')
    print("old:", text)
    print("new:", newtxt)
    return newtxt

def isvow(l):
    return l in vowels
def iscons(l):
    return l not in vowels

def find_peaks(pred, text, xdim):
    # create a new prediction matrix
    # reduce predictions to the letters that have clear peak predictions
    # turn predicitions into plottable line segments - base algorithm
    lines = []  # collection of plottable prediction curve segments
    pred_rows = pred  # one row of predictions per letter
    ltext = len(text)

    for ndx, pred_seq in enumerate(pred_rows):
        letter = G.letter_table[ndx]
        # print(letter, pred_seq.shape, pred_seq)


        avg = st.running_mean(pred_seq, winsize=dialog.avgwin, iterations=3)

        seq = []
        prevtext = ''
        for pos, n in enumerate(avg):  # the probabilities for a given letter
            relpos = pos*5 / xdim
            lolim = relpos * ltext - (xdim * 0.0001 + 5)
            hilim = relpos * ltext + (xdim * 0.0001 + 5)
            lolim = int(max(0, lolim))
            hilim = int(min(len(text)-1, hilim))
            subtext = text[lolim:hilim]
            if subtext != prevtext:
                # print(f"subtext letter: {letter}, relpos: {relpos:5.3f}, lolim: {lolim}, hilim: {hilim}, subtext: {subtext} ")
                prevtext = subtext
            if letter in subtext:
                if n > 0.05:
                    seq.append((pos*5, int(n*100)))
                else:
                    if len(seq) > 2:
                        lines.append((letter, seq))
                    seq = []
    return lines

def prep_pred_base(preds):
    # turn predicitions into plottable line segments - base algorithm
    lines = []  # collection of plottable prediction curve segments
    pred_rows = preds  # one row of predictions per letter
    for ndx, pred_seq in enumerate(pred_rows):
        letter = G.letter_table[ndx]
        # print(letter, pred_seq.shape, pred_seq)


        avg = st.running_mean(pred_seq, winsize=dialog.avgwin, iterations=3)

        seq = []
        for pos, n in enumerate(avg):  # the probabilities for a given letter
            if n > 0.05:
                seq.append((pos*5, int(n*100)))
            else:
                if len(seq) > 10:
                    lines.append((letter, seq))
                seq = []
    return lines

def plot_pred(ltr_segments, stripe):
    # letter segments is a collection of (ltr, sequence)
    plines = []
    for ltr, seq in ltr_segments:
        p, v = [], []
        p = [i[0] for i in seq]
        v = [i[1] for i in seq]

        plines.append((ltr, p, v))


    def avg(v):
        return sum(v) / len(v)


    ltrs = []
    for ltr, p, v in plines:
        lpos_x = avg(p)
        lpos_y = 20+max(v)
        color = G.colors.get(ltr, "grey")
        ltrs.append((ltr, lpos_x, lpos_y, color))
        stripe.plot(p, v, color=color, linewidth=1.0)

    for ltr, x, y, cor in ltrs:
        stripe.text(x, y, ltr, color=cor, size=12, weight='bold')


def add_group_probabilities(pred):
    # pred comes as (1182, 39)  (number of intervals, number of labels)
    print("pred", pred.shape)
    n_of_ltrs, vect_len = pred.shape
    #ltr_vectors = np.swapaxes(pred, 0, 1)  # turn this into (39, 1182)
    ltr_vectors = pred
    print("ltr_rows", ltr_vectors.shape)

    grp_pred_lst = []  # list of the prediction sums (vector for each group)
    for grp, ndxlst in G.ltr_groups:
        grp_preds = np.zeros(vect_len)
        for ndx in ndxlst:
            # ndx is the index of the vector for the specific letter
            grp_preds += ltr_vectors[ndx]  # add the probabilities
            ltr_vectors[ndx] -= ltr_vectors[ndx]

        grp_pred_lst.append(grp_preds) # collect the group predictions

    pred_extended = np.append(ltr_vectors, grp_pred_lst)
    total_rows = n_of_ltrs + len(G.ltr_groups)
    pred_extended = pred_extended.reshape(total_rows, vect_len)
    print(f"pred_extended: {pred_extended.shape}")

    return pred_extended



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


def make_letter_groups():
    # prepare probability groups
    # add up the probabilities of letters, which are in the same group
    # here we prepare the data structures for this
    letter_groups = "lL NWY tdṭ ǧbṭqd ḥhḫġ ḏṯẓ ẓszḍšṣṯ Aa* kqġ".split()
    groups = []
    letter_tab = [ltr for ltr in G.lc.ltrs]

    for grp in letter_groups:
        letter_tab.append(grp)   # extend the list of letters by the names of groups
        ndxlist = []
        for ltr in grp:
            ndxlist.append(G.lc.categ[ltr])
        groups.append((grp, ndxlist))

    G.ltr_groups = groups          #  [[grp, ndxlist], ...]
    G.letter_table = letter_tab    #  [ltr, ..., grp, ...]

    return


def colormix():
    cormix = []
    val = [200, 150, 100, 50]
    for r,g,b in product(val, val, val):
        if 350 <= (r+g+b) <= 500:
            cormix.append((r/255,g/255,b/255))
    print("colormix, found:", len(cormix))
    cortab = {l : c for l, c in zip(G.letter_table, cormix)}
    return cortab

main()
