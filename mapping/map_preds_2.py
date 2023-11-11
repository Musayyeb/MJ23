# python3
'''

    Try to get highly accurate letter positions from only evaluating predictions
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus9h"
    chap = ''
    blk = ''
    model = ''
    problvl = 0.0
    plot = False
    avgwin=20

    layout = """
    title map_nbr - Add neighborly support for certain letters
    text     recording  recording id, example: 'hus1h'
    text     chap    Chapter or a list of chapter n:m
    text     blk     Block or list of blocks 3:8 or '*'
    text     model   model name
    float    problvl probability level (% of peak) for determining length    
    bool     plot    plot to file
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
from collections import Counter
from itertools import product

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

#ltr_seq = "ʼAafṣšsNWYḍḏḥḫṭṯẓzhktġǧiLqbdlmuyʻ*nrw"  # preferred sequence for search in predictions
ltr_seq = "ṣḥfḏrʻkzybdLhqNnʼAašsWYḍḫṭṯẓtġǧilmu*w"

def main():
    stt = tbx.Statistics()
    G.colors = colormix()

    recd, chap, blk = dialog.recording, dialog.chap, dialog.blk

    model_name = dialog.model

    for chapno, blkno in tbx.chap_blk_iterator(recd, chap, blk):

        xdim, predictions = load_predictions(recd, chapno, blkno, model_name)
        xdim5 = int(xdim / 5)
        print(predictions)
        print(predictions['A'])

        text, letter_length = prepare_text(recd, chapno, blkno, xdim)
        print("prepared text:", text)

        ltrmap, probtab = do_block(text, predictions, xdim5)


        ndx, ltr, repeat = unique_letters(text)                       # This added
        print(f"letter ndx {ndx}  ltr {ltr}  repeat {repeat}  letter length {repeat*letter_length}")        # This added

        if dialog.plot:
            chart_name = f"next_{chapno:03d}_{blkno:03d}_{model_name}.png"
            plot(probtab, xdim, text, ltrmap, chart_name)

def unique_letters(text):       # A generator - yield repeated letters once with repeat counter

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


        return ndx, ltr, repeat      # yeild ndx, ltr, repeat       # This changed

def do_block(text, predictions, xdim):
    ltr_count = dict(Counter(text))         # shows how many times we have each letter
    missing_ltrs = [ltr for ltr in ltr_seq if ltr in ltr_count]

    print("initial missing letters", ' '.join([f"{ltr_count[ltr]}{ltr}" for ltr in missing_ltrs]))
    probtab, probdict = select_predictions(predictions, missing_ltrs, ltr_count)

    ltrmap = []   # [[ltr, lndx, pos, peak}, ...]
    oldsegs = [(0, len(text), 0, xdim)]  # this is the initial segment: the full block

    while oldsegs:
        newsegs = []
        for oseg in oldsegs:
            newsegs.extend( process(oseg, ltrmap, text, probdict) )
        oldsegs = newsegs
        print(f"\nnewsegs:", newsegs)


    verify_results(ltrmap, text, probdict)



    return ltrmap, probtab

def verify_results(ltrmap, text, probdict):
    """ the result may have missing letters and (few cases) letters in a wrong sequence
        it can not have extra letters
    """
    # sortmap = sorted(ltrmap, key=lambda x: x[1])  # sort by lower letter index
    sortmap = sorted(ltrmap, reverse=True)  # sort by higher letter index

    print("final result:")
    for unsrtd, srtd in zip(ltrmap, sortmap):
        print("     ", unsrtd)
    print()

    print("text ", text)
    print()

    # check the final result
    tndx = 0
    prev_hi = 0
    errpos = []
    for ltr, londx, hindx, lopos, ltrpos, hipos, peak in sortmap:
        if londx != prev_hi:
            print(f"letter index gap {prev_hi}:{londx} miss: [{text[prev_hi:londx]}] ")
            errpos.append(londx)
        prev_hi = hindx


    prev_rpos = 0
    prev_ltr = '.'
    for ltr, londx, hindx, lopos, ltrpos, hipos, peak in sortmap:
        if lopos < prev_rpos:
            print(f"ltrndx {londx} letter {ltr} overlaps prev ltr {prev_ltr}  range {lopos-prev_rpos}")
        prev_ltr, prev_rpos = ltr, hipos
    print()

    return
    # rescan area between londx -2  and hindx + 1
    oldsegs = []
    for err in errpos:
        beg = end = 0  # rescan segment
        for ndx, (ltr, londx, hindx, lopos, ltrpos, hipos, peak) in enumerate(ltrmap):

            if not beg and (err - londx) < 10:
                beg = ndx
            if not end and (hindx -err) > 3:
                end = ndx
            print(err, ndx, londx, hindx, beg, end)

        if beg and not end:
            end = len(ltrmap) - 1

        print("beg mapping", ltrmap[beg])
        print("end mapping", ltrmap[end])
        _, _, hindx, _, _, hipos, _ = ltrmap[beg]
        _, londx, _, lopos, _, _, _ = ltrmap[end]
        ltrmap[beg+1:end] = []
        oldsegs.append((hindx, londx,hipos, lopos))

    print("\n\nrestart\n\noldsegs:", oldsegs)

    while oldsegs:
        newsegs = []
        for oseg in oldsegs:
            newsegs.extend( process_2(oseg, ltrmap, text, probdict) )
        oldsegs = newsegs
        print(f"\nnewsegs:", newsegs)

    print()

    errpos = []
    for ltr, londx, hindx, lopos, ltrpos, hipos, peak in ltrmap:
        if londx != prev_hi:
            print(f"letter index gap {prev_hi}:{londx} miss: [{text[prev_hi:londx]}] ")
            errpos.append(londx)
        prev_hi = hindx


def process(segm, ltrmap, text, probdict):
    mapped_item = (None, "initial")
    print()
    lndx, rndx, lpos, rpos = segm
    ltrlst = set(text[lndx:rndx])
    missing = [ltr for ltr in ltr_seq if ltr in ltrlst]  # the missing letters in the "right" sequence

    for ltr, minprob in product( missing, (0.5, 0.3, 0.2, 0.1, 0.01)):
        mapped_item = map_ltr(ltr, segm, text, probdict[ltr], minprob)
        print(f"map_ltr result:", mapped_item)
        if not mapped_item[0] is None:
            break
    if mapped_item[0] is None:
        return []
    # insert mapped_item into lettermap
    ltrmap.append(mapped_item)

    # each mapped item splits the segment and creates 0..2 new segments
    newsegs = []
    ltr, londx, hindx, lopos, ltrpos, hipos, peak = mapped_item
    if londx > lndx:
        newsegs.append((lndx, londx, lpos, lopos))
    if hindx < rndx:
        newsegs.append((hindx, rndx, hipos, rpos))
    return newsegs

def process_2(segm, ltrmap, text, probdict):
    mapped_item = (None, "initial")
    print()
    lndx, rndx, lpos, rpos = segm
    ltrlst = set(text[lndx:rndx])
    missing = [ltr for ltr in ltr_seq if ltr in ltrlst]  # the missing letters in the "right" sequence

    for minprob, ltr in product( (0.1,  0.01), missing):
        mapped_item = map_ltr(ltr, segm, text, probdict[ltr], minprob)
        print(f"map_ltr result:", mapped_item)
        if not mapped_item[0] is None:
            break
    if mapped_item[0] is None:
        return []
    # insert mapped_item into lettermap
    ltrmap.append(mapped_item)

    # each mapped item splits the segment and creates 0..2 new segments
    newsegs = []
    ltr, londx, hindx, lopos, ltrpos, hipos, peak = mapped_item
    if londx > lndx:
        newsegs.append((lndx, londx, lpos, lopos))
    if hindx < rndx:
        newsegs.append((hindx, rndx, hipos, rpos))
    return newsegs

def map_ltr(ltr, segm, text, prob_avg, minprob):
    # return a mapping item, or None + a message
    print(f"map_ltr {segm} ==> {ltr}")
    lndx, rndx, lpos, rpos = segm

    if lpos >= rpos:
        return None, f"map_ltr - no space left for ltr '{ltr}': {lpos}:{rpos}"

    ltrpos = lpos + prob_avg[lpos:rpos].argmax()  # get the index of the highest value in the vector
    peak = round(prob_avg[ltrpos], 3)
    if peak < minprob:
        return None, f"peak too low: {peak:5.3f} < {minprob:5.3f}"

    print(f"found letter '{ltr}', at position {ltrpos} scanned prob[{lpos}:{rpos}], strength={peak:5.3f}")

    lxrange = confirm_letter(segm, text, ltr, ltrpos)
    if lxrange[0] is None:
        return lxrange
    # the letter is actually in the text
    londx, hindx = lxrange

    lopos, hipos = prob_range(prob_avg, ltrpos)

    print(f"found letter '{ltr}' at index {lxrange}")

    return ltr, londx, hindx, lopos, ltrpos, hipos, peak


def confirm_letter(segm, text, ltr, ltrpos):
    # search in a certain range
    # return an index range, or None + message
    lndx, rndx, lpos, rpos = segm
    # calc the relative position of ltrpos between pos1 and pos2
    relpos = (ltrpos - lpos) / (rpos-lpos)
    print(f"ltr position {lpos} < {ltrpos} < {rpos} ==> {relpos:4.2f} relpos")
    lopos = relpos * 0.90
    hipos = relpos * 1.10
    txtlen = rndx - lndx
    lx = max(lndx, lndx + math.floor(lopos * txtlen) - 2)
    rx = min(lndx + math.ceil(hipos * txtlen) + 2, rndx)

    teststr = text[lx:rx]
    print(f"ltr index {lndx} < {lx} < '{teststr}' < {rx} < {rndx}")

    # verify, that there is one and only one ltr(sequuence)
    findx = teststr.find(ltr)
    if findx == -1:
        return None, f"cnfrm ltr '{ltr}' not found"
    if not teststr.count(ltr) == teststr.rfind(ltr) - findx + 1:
        return None,  f"cnfrm ltr '{ltr}' more than one occurence"

    hindx = findx + lx  # text index is where the letter is found, there may be repeated letters

    while hindx < len(text) and text[hindx] == ltr:
        hindx += 1

    londx = findx + lx
    while londx >= 0 and text[londx] == ltr:
        londx -= 1

    return londx + 1, hindx


def prob_range(prob_avg, ltrpos):
    # scan the probabilities left and right, where they are above a certain level
    peak = prob_avg[ltrpos]
    min_lvl = peak  * dialog.problvl
    lx = rx = ltrpos

    while prob_avg[lx] > min_lvl:
        lx -= 1
    while prob_avg[rx] > min_lvl:
        rx += 1
    return lx, rx

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

    xinch = 2 + xdim5/90
    yinch = 2 + len(probtab)*0.4

    fig, ax = plt.subplots(1, 1, figsize=(xinch, yinch), dpi=160, gridspec_kw={})
    fig.tight_layout()
    stripe = plt # ax[0]
    ydim = len(probtab) *0.8

    stripe.ylim(0,ydim)
    stripe.xlim(-100, xdim5+20)

    ticks = np.arange(0, xdim5, 100)
    tlabels = [f"{int(n)}" for n in ticks]
    stripe.xticks(ticks, labels=tlabels)

    # ltrmap.append([lndx, ltr, pos])

    for vndx, (ltr, ct, _, prob_avg) in enumerate(probtab):
        cor = G.colors[ltr]
        ypos = (ydim - 3) / len(probtab) * vndx + 3
        stripe.text(-90, ypos, f"{ltr} ({ct})", fontsize=20, color="blue")

        stripe.plot(prob_avg * 1.0 + ypos, color=cor)

    ltrxpos = np.arange(0, xdim5-10, xdim5 / len(text))
    # print("ltrxpos", ltrxpos, len(ltrxpos))

    ypos_text, ypos_map, ypos_mx, ypos_peak = 0.2, 0.8, 1.5, 1.9

    for x, c in zip(ltrxpos, text):
        stripe.text(x, ypos_text, c, fontsize=20, color="black")

    for mapx, (ltr, londx, hindx, lopos, ltrpos, hipos, peak) in enumerate(ltrmap):
        cor = G.colors[ltr]
        name = ltr * (hindx - londx)
        stripe.text(ltrpos, ypos_map, name, fontsize=20, horizontalalignment='center', color=cor)

        stripe.vlines(x=ltrpos, ymin=ypos_peak, ymax=ypos_peak + peak * 1.0 + 0.1,
                   colors=cor, linewidth=3)

        # draw small diagonals to show the length of the letter
        x1 = np.array([lopos, hipos])
        yoffs = (hipos-lopos)*0.002
        y1 = np.array([ypos_peak-yoffs-0.1, ypos_peak+yoffs-0.1])

        plt.plot(x1, y1, color=cor, linewidth=3)

        stripe.text(ltrpos, ypos_mx, mapx, fontsize=10, horizontalalignment='center', color='black')

    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print(f"chart is saved as {chart}")



def prepare_text(recd, chapno, blkno, xdim):
    text = tt.get_block_text(recd, chapno, blkno)  # , spec="ml")
    print(f"db_text: [{text}] len: {len(text)}")

    text = text.strip('.').strip()  # remove '.', also remove space
    text = adjust_timing(text)  # make text more linear in time
    G.tempo = int((xdim - 800)/len(text))        # tempo = average length of letters in block


    # calulate relation between text and audio length
    letter_length = (xdim - 800) / len(text)  # milliseconds per letter
    print(f"adjusted: [{text}] len:{len(text)}")
    # print(f"letter_length: {int(letter_length)} ms")
    return text, letter_length


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





def colormix():
    cormix = []
    val = [220, 90, 20, 160]
    for r,g,b in product(val, val, val):
        cormix.append((r/255,g/255,b/255))
    #print("colormix, found:", len(cormix))
    print(f"colormix seq={len(ltr_seq)}  cor={len(cormix)}")
    cortab = {l : c for l, c in zip(sorted(ltr_seq), cormix)}
    print(cortab.keys())
    return cortab

main()

