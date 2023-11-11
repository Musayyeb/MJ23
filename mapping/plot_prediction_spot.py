                                                                                                                                                                                                                                                   # python3
'''
    Plot a small spot of the predictions
    
    For analyzing mapping problems, it may be helpful to look at a specific spot
    of the predictions of a block
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus1h"
    chapno = ''
    blkno = ''
    name = ''     # model name
    lpos = 0
    length = 1000  # milliseconds
    use_support = True

    layout = """
    title prediction splot - look at a specific part of a block
    text     recording  recording id, example: 'hus1h'
    int      chapno    Chapter (n)
    int      blkno     Block (n)
    text     name      model name
    int      lpos      start position (ms)
    int      length    length (ms)
    bool     use_support  duplicate predictions for nbr support
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config, AD
cfg = get_config()

from matplotlib import pyplot as plt
import numpy as np
import splib.toolbox as tbx
import splib.sound_tools as st
from itertools import product
import splib.attrib_tools as att
import splib.text_tools as tt
import splib.project_db as pdb
from machine_learning.ml_tools import PredLoader, get_training_specs
from scipy.stats import norm
from dataclasses import dataclass, field
from collections import Counter
from typing import List
import json
import random
import time
import os
import traceback

class G:
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None  # colormix stored here
    mapped_letters = []
    results = []  # ratings for blocks and iterations
    friends = {}
    support_list = []
    all_data = {}  # collection of block related results
    debug = False  # trigger print statements
    statfile = ""  # statistics file name

vowels = 'aiuAYNW*'

# all ltrs: '.abdfhiklmnoqrstuwyzġšǧʻʼḍḏḥḫṣṭṯẓ*ALNWY'
support_weights = {0:3, 1:5, 2:8, 3:11, 4:15, 5:20, 6:30, 7:40, 8:50, 9:60}
friends = [ # the x-axis supports the y-axis
'''
_ b d k q r t ǧ ṭ l L ġ 
b _ 4 . 4 . . 4 4 . . 4
d 4 _ . 4 . 4 4 4 . . 4
k . . _ 3 . . . . . . 3
q 4 4 4 _ . . 4 4 . . 5
r . . . . _ . . . 4 2 .
t . 4 . . . _ . 4 . . .
ǧ 4 4 . 4 . . _ 4 . . .
ṭ 4 4 . 4 . 4 4 _ . . .
l . . . . 4 . . . _ 4 .
L . . . . 4 . . . 4 _ .
ġ . . 3 5 1 1 . . . . _
''',
'''
_ a A * i y Y u w W
a _ 5 4 . . . . . .
A 7 _ 4 . . . . . .
* 4 4 _ . . . 3 . .
i . . . _ 3 3 . . .
y . . . 3 _ 5 . . .
Y . . . 6 5 _ . . .
u . 3 . . . . _ 3 3
w . . . . . . 5 _ 4
W . . . . . . 3 2 _
''',
'''
_ m n N w h ʼ ḥ h ḫ d ḏ ẓ l
m _ 3 2 . . . . . . . . . .
n 3 _ 4 . . . . . . . . . 3
N 2 4 _ . . . . . . . . . .
w . . . _ 3 3 . . . . . . .
h . . . 3 _ 3 3 . 4 . . . .
ʼ . . . 3 3 _ . 4 4 . . . .
ḥ . . . . 3 . _ 4 4 . . . .
h . . . . . 4 4 _ 4 . . . .
ḫ . . . . 4 4 4 4 _ . . . .
d . . . . . . . . . _ 4 . .
ḏ . . . . . . . . . . _ 4 .
ẓ . . . . . . . . . . 4 _ .
l . 3 . . . . . . . . . . _

'''
]


@dataclass
class MappedLetter:
    ltr : str
    lndx: int  # letter index
    repeat: int  # ltr repetition
    spos : int  # start position (ms)
    epos : int  # end position
    hipos : int = 0  # where the prediction is a maximum


def main():

    G.colors = colormix()
    G.support_list = parse_support()  # collect all supporting neighbor relations into one list
    # print("support list: ", G.support_list)

    d = dialog
    recd, chapno, blkno, lpos, length = d.recording, d.chapno, d.blkno, d.lpos, d.length

    model_name = d.name
    specs = get_training_specs(recd, model_name)
    itno_list = range(specs.iter)


    pred_loader = PredLoader(recd, chapno, blkno)
    vect_loader = att.VectLoader(recd, chapno, blkno)

    xdim = vect_loader.get_xdim()  # duration of block determins the x dimension of chart

    text, letter_length = prepare_text(recd, chapno, blkno, xdim)

    pred_list = prepare_prediction_data(itno_list, pred_loader, model_name)

    plot_mappings(pred_list, chapno, blkno, text, lpos, length)


def prepare_prediction_data(itno_list, pred_loader, model_name):

    pred_list = []
    # print(f"use model {model_name} with [{itno_list}] iterations")
    for itno in itno_list:

        ltr_rows = pred_loader.get_calculated_predictions(model_name, itno)
        # the predictions are for 5ms intervals - to avoid confusion, convert the data into 1ms
        ltr_rows = np.array([interpolate(row) for row in ltr_rows])
        for row in ltr_rows:  # each row has some noise: flatten 350 ms at both ends
            row[0:350] = 0
            row[-350:] = 0

        sup_rows = add_nbr_support(ltr_rows)
        # reversed_rows = np.flip(new_rows, axis=1)  # dont use the reverse approach

        # for each iteration of the model,
        #     try a mapping on the original predictions (ltr_rows)
        #     and on the supported predictions  (sup_rows)
        alt_mapping = (("orig", ltr_rows), ("supp", sup_rows))
        for mode, preds in alt_mapping:

            if not dialog.use_support and mode == "supp":
                # if not selected, skip supported mappings
                continue

            # print(f"chapno: {chapno:03d}, blkno: {blkno:03d}, iteration: {itno:2d},  mode: {mode}")

            avg_tab = calculate_predictions_average(preds)
            pred_list.append((itno, mode, avg_tab))
    return pred_list

def calculate_predictions_average(np_mat):
    # calculate a smoothed curve for the predictions
    # also turn the data into a letter-based dictionary
    avg_tab = {}  # smoothed single letter probabilities

    for ndx, ltr_seq in enumerate(np_mat):
        # put the averaged probability curves into a new matrix
        letter = G.lc.ltrs[ndx]
        avg = st.running_mean(ltr_seq, winsize=20, iterations=3)
        avg_tab[letter] = avg

    return avg_tab


def get_weights(line, colx):
    # get the weights from the line,
    benf = line[0] # the beneficiary letter is in column 0
    for ndx, ltr in enumerate(line):
        if not ltr in "._ " and ndx > 1:
            col = int(ndx/2)
            nbr = colx[col]  # the neighbor who gives support
            pcnt = support_weights[int(ltr)] / 100
            yield nbr, benf, pcnt

def get_columns(line):
    # dictionary of letters for each column
    cols = {}
    for ndx, ltr in enumerate(line):
        if ltr not in "_ ":
            col = int(ndx/2) # this column index refers to the line index
            cols[col] = ltr
    return cols

def parse_lines(mat):
    # basic check of lines, yield lines one by one
    lines = [l.strip() for l in mat.splitlines() if l.strip()]
    if lines[0] == '#':
        return
    len0 = len(lines[0])
    cols = int(len0/2) + 1
    assert len(lines) == cols, f"len of line 0: {len0} does not match number of lines: {len(lines)}"
    for l in lines:
        assert len(l) == len0, f"line:{l} does not match the length {len0} of the first line"
        yield l

def prepare_text(recd, chapno, blkno, xdim):
    text = tt.get_block_text(recd, chapno, blkno)  # , spec="ml")
    print(f"db_text: [{text}] len: {len(text)}")

    text = text.strip('.').strip()  # remove '.', also remove space
    text = adjust_timing(text)  # make text more linear in time

    # calulate relation between text and audio length
    letter_length = (xdim - 800) / len(text)  # milliseconds per letter
    print(f"adjusted: [{text}] len:{len(text)}")
    print(f"letter_length: {int(letter_length)} ms")
    return text, letter_length


def add_nbr_support(ltr_rows):
    new_rows = np.copy(ltr_rows)
    # print("new_rows:", new_rows.shape)
    for nbr, benf, pcnt in G.support_list:
        nbrx = G.lc.categ[nbr]
        benx = G.lc.categ[benf]
        # print(f"nbr_supp: {nbr}:{nbrx}, {benf}:{benx}")
        new_rows[benx] += new_rows[nbrx]*pcnt
    return new_rows


def parse_support():
    # build a list of support weights
    support_list = []
    for matx, mat in enumerate(friends):
        for lx, line in enumerate(parse_lines(mat)):
            if lx == 0:
                colx = get_columns(line)
            else:
                for weight in get_weights(line, colx):
                    nbr, benf, pcnt = weight
                    support_list.append((nbr, benf, pcnt))
    return support_list


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
    newtxt = newtxt.replace('..', '.')
    return newtxt

def isvow(l):
    return l in vowels
def iscons(l):
    return l not in vowels


def plot_mappings(pred_list, chapno, blkno, text, lpos, length):
    vwins, hwins = 16, 2 # small plot windows for single failed mappings

    fig = plt.figure(figsize=(18, 36), dpi=200, tight_layout=True)
    ax = fig.subplots(vwins, hwins, gridspec_kw={})
    plt.subplots_adjust(top=2.8, wspace=0.3)
    rpos = lpos+length
    fig.suptitle(f'xkey {chapno:03d}_{blkno:03d}  pos {lpos} - {rpos} ', fontsize=20)
    chart_name = f"prediction spot {chapno:03d}_{blkno:03d}_pos_{lpos}_{rpos}.png"


    for itx, (itno, mode, avg_tab) in enumerate(pred_list):
        if itx >= vwins*hwins:
            break
        # avg_tab is a dictionary, where all letters come with their probability vector over the length of the block
        vax = itx // hwins
        hax = itx %  hwins

        itwin = ax[vax, hax]  # each iteration gets its own little plot window
        plotlen = rpos - lpos
        xvect = np.linspace(lpos, rpos, plotlen)
        itwin.set_ylim(-0.1, 1.5)
        itwin.set_xlim(lpos, rpos)

        for predltr, seq in avg_tab.items():
            #print(f"itwin for {chapno}_{blkno}  [{ltr}]*{repeat} ({lndx}) ==> {itltr} {itx} ax:{vax, hax}")
            cor = G.colors.get(predltr, "grey")

            #print(f"pred seq: {predltr} {cor} l={len(seq)}, seq:{seq[500:510]}")

            plotv = np.copy(seq[lpos:rpos])
            tvect = xvect
            if len(xvect) > len(plotv):
                tvect = xvect[:len(plotv)]
            plotv[plotv < 0.05] = np.nan
            if np.nanmax(plotv) > 0.01:
                lp = np.argmax(seq[lpos:rpos])
                yp = 1 + 0.3*plotv[lp]  # take the probaility at the strongest prediction to midify the y pos of the letter
                xp = lp + lpos
                #print(f"plot text {lpos} {predltr}")
                itwin.text(xp, yp, predltr, color=cor, fontsize=14)

                itwin.plot(tvect, plotv, color=cor)

    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print("saved as:", chart)

    #  plt.show()  # does not work as expected

def interpolate(vect_5ms, yfact=1, move=0):
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
    #print("colormix, found:", len(cormix))
    cortab = {l : c for l, c in zip(G.lc.ltrs, cormix)}
    return cortab

main()
