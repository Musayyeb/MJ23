# python3
'''

    Try to get highly accurate letter positions from only evaluating predictions
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus1h"
    chap = ''
    blk = ''
    model = ''
    plot = False
    avgwin = 20

    layout = """
    title map_nbr - Add neighborly support for certain letters
    text     recording  recording id, example: 'hus1h'
    text     chap    Chapter or a list of chapter n:m
    text     blk     Block or list of blocks 3:8 or '*'
    bool     plot    plot to file
    text     model   model name
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
from machine_learning.ml_tools import PredLoader, get_training_specs
from dataclasses import dataclass
from collections import Counter
import json
import random
import time
import traceback
import math

class G:
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None  # colormix stored here
    mapped_letters = []
    results = []  # ratings for blocks and iterations
    friends = {}
    support_list = []
    all_data = {}  # collection of block related results

vowels = 'aiuAYNW*'




def main():
    stt = tbx.Statistics()
    G.colors = colormix()

    recd, chap, blk = dialog.recording, dialog.chap, dialog.blk

    model_name = dialog.model


    ltr_seq = "ḍḏḥḫṣṭṯẓzfhkstġšǧiALqbdlmuayʻʼ*NWYnrw"  # preferred sequence for search in predictions

    for chapno, blkno in tbx.chap_blk_iterator(recd, chap, blk):

        xdim, predictions = load_predictions(recd, chapno, blkno, model_name)
        xdim5 = int(xdim / 5)

        text, letter_length = prepare_text(recd, chapno, blkno, xdim)
        text = '.' + text + '.'
        print("prepared text:", text)

        ltr_count = dict(Counter(text))
        missing_ltrs = [ltr for ltr in ltr_seq if ltr in ltr_count]

        print("initial missing letters", ' '.join([f"{ltr_count[ltr]}{ltr}" for ltr in missing_ltrs]))

        ltrmap = [[0, '.', 20, 0], [len(text), '.', xdim5-20, 0]]  # [[ltr, lndx, pos, peak}, ...]

        probtab, probdict = select_predictions(predictions, missing_ltrs, ltr_count)


        print("   - - -")
        for minprob in (0.5, 0.3, 0.2, 0.1, 0.0):
            for ndx, ltr in enumerate(missing_ltrs):
                if ltr_count[ltr] > 0:

                    # this is the mapping routine
                    mapct = map_ltr(ltr, text, ltrmap, probdict[ltr], minprob)

                    if mapct:
                        ltr_count[ltr] -= mapct
                        # print(f"iter {minprob} missing letters", ' '.join([f"{ltr_count[ltr]}{ltr}" for ltr in missing_ltrs]))


            missing_ltrs = [ltr for ltr in missing_ltrs if ltr_count[ltr]]
            print(f"iter {minprob} missing letters", [(ltr, ltr_count[ltr]) for ltr in missing_ltrs])

        for x in sorted(ltrmap):
            print("     ", x)
        print()

        if dialog.plot:
            chart_name = f"next_{chapno:03d}_{blkno:03d}_{model_name}.png"
            plot(probtab, xdim, text, ltrmap, chart_name)



def map_ltr(ltr, text, ltrmap, prob_avg, minprob):
    # find this one letter in the probabilities
    mapct = 0
    print()
    for beg, end, ltrct in get_map_segments_2(ltrmap, ltr, text):  # the letter may occur in more than 1 segment
        lpos, rpos = beg[1], end[1]
        if lpos >= rpos:
            print(f"bad position range {beg}:{end} - ignored")
            continue
        ltrpos = lpos + prob_avg[lpos:rpos].argmax()  # get the index of the highest value in the vector
        peak = prob_avg[ltrpos]
        if peak < minprob:
            print(f"peak {peak:5.3f} < {minprob:5.3f}")
            continue
        print(f"found letter '{ltr}', at position {ltrpos} scanned prob[{lpos}:{rpos}], strength={peak:5.3f}")
        lxrange = confirm_letter(text, ltr, ltrpos, beg, end)
        # the letter is actually there
        if not lxrange:
            continue
        print(f"found letter '{ltr}' at index {lxrange}")

        insert_found_letter(ltrmap, ltr, ltrpos, lxrange, peak)
        mapct += len(lxrange)

    return mapct

def insert_found_letter(ltrmap, ltr, ltrpos, lxrange, peak):
    # in case of repeated letters, the actual positions are only assumptions
    tspan = 20 * len(lxrange)
    fst = ltrpos - (tspan + 20) / 2
    for relpos, lndx in enumerate(lxrange):
        pos = int(fst + relpos * 20)
        ltrmap.append([lndx, ltr, pos, peak])

def get_map_segments_2(ltrmap, ltr, text):
    # yield the gaps between two mapped letters
    sortmap = sorted(ltrmap)
    for (lndx1, _, pos1, _), (lndx2, _, pos2, _) in zip(sortmap, sortmap[1:]):
        if lndx2 - lndx1 == 1:
            continue
        ltrct = text.count(ltr, lndx1, lndx2)  # use a built-in  string method
        if ltrct: # there is something
            print(f"get_segments: found {ltrct} * '{ltr}' in text {lndx1}:{lndx2}")
            yield (lndx1+1, pos1+5), (lndx2, pos2-5), ltrct


def confirm_letter(text, ltr, ltrpos, beg, end):
    # search in a certain range
    # return the (all) letter indexes, where this letter was found


    """
    # we r searching for letter with lndxn, where lndx1 < lndxn < lndx2
    lndx1, pos1 = str
    lndx2, pos2 = end
    txtlen = lndx2 - lndx1
    ltxtlen, rtxtlen = ((lndx2 - lndxn) - 1), ((lndx1 - lndxn) -1)
    hipos =       (((lndxn+repeat-1) * ltr_time) + str
    lopos = end - (((lndxn-1) * ltr_time)
    lopos < relpos < hipos
    """


    lndx1, pos1 = beg

    lndx2, pos2 = end
    # calc the relative position of ltrpos between pos1 and pos2
    relpos = (ltrpos - pos1) / (pos2-pos1)
    print(f"ltr position {pos1} < {ltrpos} < {pos2} ==> {relpos:4.2f} relpos")
    lopos = relpos * 0.90
    hipos = relpos * 1.10
    txtlen = lndx2 - lndx1
    londx = max(lndx1, lndx1 + math.floor(lopos * txtlen) - 2)
    hindx = min( lndx1 + math.ceil(hipos * txtlen) + 2, lndx2)
    print(f"ltr index {lndx1} < {londx} < '{text[londx:hindx]}' < {hindx} < {lndx2}")

    # verify, that there is one and only one ltr(sequuence)
    teststr = text[londx:hindx]
    ltrndx = teststr.find(ltr)
    if ltrndx == -1:
        print(f"cnfrm ltr '{ltr}' not found")
        return []
    if not teststr.count(ltr) == teststr.rfind(ltr) - ltrndx + 1:
        print(f"cnfrm ltr '{ltr}' more than one occurence")
        return []
    lx = ltrndx + londx
    found = []
    tx = lx
    while tx < len(text) and text[tx] == ltr:
        found.append(tx)
        tx += 1

    tx = lx - 1
    while tx >= 0 and text[tx] == ltr:
        found.append(tx)
        tx -= 1

    return found


def get_map_segments(ltrmap):
    # yield the gaps between two mapped letters
    sortmap = sorted(ltrmap)
    for (lndx1, _, pos1), (lndx2, _, pos2) in zip(sortmap, sortmap[1:]):
        if lndx2 - lndx1 == 1:
            continue
        yield (lndx1+1, pos1+50), (lndx2, pos2-50)

def get_single_segment(ltrmap, ltrpos):
    # yield the gaps between two mapped letters
    sortmap = sorted(ltrmap)
    for (lndx1, _, pos1), (lndx2, _, pos2) in zip(sortmap, sortmap[1:]):
        if lndx2 - lndx1 == 1:
            continue
        if pos1+50 < ltrpos < pos2-50:
            return (lndx1+1, pos1+50), (lndx2, pos2-50)

def select_predictions(predictions, missing_ltrs, ltr_count):
    probtab = []
    probdict = {}

    for ltr in missing_ltrs:  # needed predictions for this block
        ct = ltr_count[ltr]
        prob = np.array(predictions[ltr])

        prob_cor = calculate_predictions_average(prob, dialog.avgwin)  # average over time (floating window)
        prob_avg = np.average(prob_cor, axis=0)  # average over the iterations

        probtab.append((ltr, ct, prob_cor, prob_avg))  # this is for plotting

        probdict[ltr] = prob_avg  # this is for mapping

    return probtab, probdict


def load_predictions(recd, chapno, blkno, model_name):
    specs = get_training_specs(recd, model_name)
    itno_list = range(specs.iter)
    pred_loader = PredLoader(recd, chapno, blkno)
    vect_loader = att.VectLoader(recd, chapno, blkno)
    xdim = vect_loader.get_xdim()  # duration of block determins the x dimension of chart

    predictions = {}
    for letter in G.lc.ltrs:
        predictions[letter] = []

    # ltr_seq_ms = interpolate(ltr_seq)
    # avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
    # avg_tab[letter] = avg

    # for each block, mapping is done with different iterations (ML models)
    for itno in itno_list:
        # print(f"chapno: {chapno:03d}, blkno: {blkno:03d}, iteration: {itno:2d}")

        ltr_rows = pred_loader.get_calculated_predictions(model_name, itno)
        # the predictions are for 5ms intervals - to avoid confusion, convert the data into 1ms
        # ltr_rows = np.array([interpolate(row) for row in ltr_rows])
        preds = ltr_rows
        preds[:, 0:70] = 0  # clean the first and last 350 ms of each block
        preds[:, -70:] = 0
        # print("preds shape", preds.shape)
        for ndx in range(preds.shape[0]):
            ltr = G.lc.label[ndx]
            predictions[ltr].append(preds[ndx])

    return xdim, predictions

def plot(probtab, xdim, text, ltrmap, chart_name):

    xdim5 = int(xdim / 5)

    xinch = 2 + xdim5/100
    yinch = 2 + len(probtab)*0.4

    fig, ax = plt.subplots(1, 1, figsize=(xinch, yinch), dpi=160, gridspec_kw={})
    fig.tight_layout()
    stripe = plt # ax[0]
    ydim = len(probtab) *0.8

    stripe.ylim(0,ydim)
    stripe.xlim(-100, xdim5+20)

    # ltrmap.append([lndx, ltr, pos])

    for vndx, (ltr, ct, _, prob_avg) in enumerate(probtab):

        ypos = (ydim - 3) / len(probtab) * vndx + 0.5
        stripe.text(-90, ypos, f"{ltr} ({ct})", fontsize=20, color="blue")

        stripe.plot(prob_avg[20:] * 1.0 + ypos)

    ltrxpos = np.arange(0, xdim5-10, xdim5 / len(text))
    # print("ltrxpos", ltrxpos, len(ltrxpos))

    ypos_text, ypos_map, ypos_peak = ydim-0.7, ydim-1.5, ydim-2.5
    for x, c in zip(ltrxpos, text):
        stripe.text(x, ypos_text, c, fontsize=20, color="black")

    for lndx, ltr, ltrpos, peak in ltrmap:
        stripe.text(ltrpos, ypos_map, ltr, fontsize=20, horizontalalignment='center', color="red")

        stripe.vlines(x=ltrpos, ymin=ypos_peak, ymax=ypos_peak + peak * 1.0,
                   colors='purple')

    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print(f"chart is saved as {chart}")

def plot_stripe(stripe, xdim, ltr, ct, prob_cor, prob_avg):
    init_stripe(stripe, xdim)

    pltcor = np.swapaxes(prob_cor, 0, 1)
    stripe.plot(pltcor - 1)  # put the colorful probabilities below the zero line
    stripe.plot(prob_avg * 3)
    stripe.text(-180, 0, f"{ltr} ({ct})", fontsize=12)

def init_stripe(stripe, xdim):
    start, stop = 0, xdim
    ticks = np.arange(start, stop, 400)
    tlabels = [f"{int(n * 5)}" for n in ticks]
    stripe.set_xticks(ticks, labels=tlabels)
    print(f"xdim {xdim} ticks {len(ticks)} {ticks[:3]} labels {tlabels[:3]}")



def __plot(probtab, max_stripes, xdim, text, ltrmap, chart_name):

    stripes = min(max_stripes, len(probtab))

    fig, ax = plt.subplots(stripes + 1, 1, figsize=(6 + xdim/1000, stripes * 1.2), dpi=160, gridspec_kw={})
    fig.tight_layout()
    xdim5 = int(xdim / 5)

    for axx in range(0, stripes):
        if axx == 0:
            stripe0 = ax[0]

            init_stripe(stripe0, xdim5)
            continue
        # leave ax[0] empty
        ltr, ct, prob_cor, prob_avg = probtab[axx - 1]

        plot_stripe(ax[axx], xdim5, ltr, ct, prob_cor, prob_avg)

    ltrxpos = np.arange(10, xdim-10, xdim5 / len(text))
    print("ltrxpos", ltrxpos, len(ltrxpos))
    for x, c in zip(ltrxpos, text):
        stripe0.text(x, 0.3, c)
    for lndx, ltr, ltrpos in ltrmap:
        stripe0.text(ltrpos, -0.9, ltr)


    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print(f"chart is saved as {chart}")


def prepare_text(recd, chapno, blkno, xdim):
    text = tt.get_block_text(recd, chapno, blkno)  # , spec="ml")
    print(f"db_text: [{text}] len: {len(text)}")

    text = text.strip('.').strip()  # remove '.', also remove space
    text = adjust_timing(text)  # make text more linear in time

    # calulate relation between text and audio length
    letter_length = (xdim - 800) / len(text)  # milliseconds per letter
    print(f"adjusted: [{text}] len:{len(text)}")
    # print(f"letter_length: {int(letter_length)} ms")
    return text, letter_length


def find_position(recd, chapno, blkno, xdim):
    text, letter_length = prepare_text(recd, chapno, blkno, xdim)
    max_len, max_ltr, max_ind = longest_sequence(text)   
    
    mid_pos = max_ind + max_len // 2
    avg_len = (xdim - 800) // len(text)

    rec_pos = mid_pos * avg_len

    return max_len, rec_pos, max_ltr, text

def scan_left(ltr, seq, pos, level):
    # call scan_left only if you are inside the letter
    ipos = pos
    while seq[pos] >= level:  # go left for the left boundary
        pos -= 1
    # print(f"      scan left '{ltr}' from {ipos}, found pl: {pos}, lvl={level:4.2f}")
    return pos

def scan_right(ltr, seq, pos, endpos, level):
    # call scan_right only if you are before the letter start
    ipos = pos
    while pos < endpos:  # go right for the left boundary
        if seq[pos] >= level:
            break
        pos += 1
    else:
        return 0
    # print(f"      scan right '{ltr}' from {ipos}, found pl: {pos}, endpos={endpos}")
    return pos

def scan_for_the_end(ltr, seq, pos, endpos, level):
    # call scan_4_end to find the end of the letter
    # while minlen
    ipos = pos # initial position
    pr = 0
    highest = 0
    highpos = 0
    while True:
        prob = seq[pos]
        if prob >= level or pos < endpos:
            if prob >= level:
                pr = pos # there is actually some probability here
            if prob > highest:
                highest = prob
                highpos = pos
            pos += 1

        else:
            break
    if pr:
        pass # print(f"      scan for the end '{ltr}'  ipos:{ipos}, endpos:{endpos} pos:{pos}")
    else:
        pass # print(f"         scan for the end '{ltr}' failed: ipos:{ipos}, endpos:{endpos} pos:{pos}")
    return pr, highest, highpos

def scan_predictions(ltr, seq, pos, span, level):
    # seq = prodiction vector for the given letter
    # pos = position in vector where to search
    # minlen = minimum length of scanning - even when the predictions go (temporarily) down
    # level = minimum probability of the predictions to be observed
    highest = 0
    highpos = 0

    limit = pos + span
    # if predictions overlap, after one letter we may be in the next already
    # so eventually go back and search the beginning
    if seq[pos] >= level:
        pl = scan_left(ltr, seq, pos, level)
    else:
        endpos = min(limit, len(seq)-1)
        pl = scan_right(ltr, seq, pos, endpos, level)

    if pl == 0:
        return None  # letter not found

    # pl is the position of the left boundary
    # now go for the right boundary 'pr'
    endpos = min(pl + span, len(seq)-1)
    pr, highest, highpos = scan_for_the_end(ltr, seq, pl, endpos, level)

    # the idea is to find better boundaries - mostly works well
    # better boundaries is for high probability predictions increase the level for the boundaries
    #if newlevel := highest * 0.3 > level:
    #    lscan_pos = int(min((pl+pr)/2, endpos, highpos))  # dont start too far away
    #    pl = scan_left(ltr, seq, lscan_pos, newlevel)
    #    pr, highest, highpos = scan_for_the_end(ltr, seq, pl, endpos, newlevel)

    return pl, pr, highest, highpos


def check_letter(recd, chapno, blkno, xdim):
    max_len, rec_pos, max_ltr, text = find_position(recd, chapno, blkno, xdim)

    pl, pr, highest, highpos = scan_predictions(max_ltr, text, rec_pos, 0, 0.07)
    
    avg_len = (xdim - 800) // len(text)
    pred_len = (pr - pl) / avg_len 

    if max_len+ 1  > pred_len > (max_len - 1):
        return True
    return False

def calculate_predictions_average(np_mat, winsize):
    # calculate a smoothed curve for the predictions
    # also turn the data into a letter-based dictionary
    avg_tab = []  # smoothed single letter probabilities

    for ndx, ltr_seq in enumerate(np_mat):
        # put the averaged probability curves into a new matrix
        letter = G.lc.ltrs[ndx]
        # ltr_seq_ms = interpolate(ltr_seq)
        avg = st.running_mean(ltr_seq, winsize=winsize, iterations=3)
        avg_tab.append(avg)

    return np.array(avg_tab)


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


def unique_letters(text):
    # A generator - yield repeated letters only once, but with repeat counter
    skip = 0
    for ndx, ltr in enumerate(text):
        if skip:
            # skip over previously processed letters
            skip -= 1
            continue

        # double letters are detected only once, but are counted
        dndx = ndx
        repeat = 1
        while True:
            dndx += 1
            if dndx < len(text) and text[dndx] == ltr:
                repeat += 1
                skip += 1
                continue
            else:
                break
        yield ndx, ltr, repeat


def colormix():
    cormix = []
    val = [200, 150, 100, 50]
    for r,g,b in product(val, val, val):
        if 350 <= (r+g+b) <= 500:
            cormix.append((r/255,g/255,b/255))
    #print("colormix, found:", len(cormix))
    cortab = {l : c for l, c in zip(G.lc.ltrs, cormix)}
    return cortab

def longest_sequence(s):
    max_len = 0
    max_ltr = ''
    max_ind = 0

    while True:
        try:
            ndx, ltr, repeat = unique_letters(s)
            if repeat > max_len:
                max_len = repeat
                max_ltr = ltr
                max_ind = ndx
        except:
            break

    return max_len, max_ltr, max_ind


main()



