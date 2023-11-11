# python3
'''

    start mapping of predictions

    plot the predictions along with some audio attributes

    Now: get a rating for the mapping
    run more than one model and compare the ratings
'''

from splib.cute_dialog import start_dialog
import sys

class dialog:
    recording = "hus9h"
    chapno = 0
    blkno = ''
    name = ''
    iters = 0
    avgwin = 5
    shushu = 99

    layout = """
    title Nullmap - start mapping
    text     recording  recording id, example: 'hus1h'
    int      chapno    Chapter
    text     blkno     Block or list of blocks 3:8
    text     name      model name
    int      iters     model iterations

    int      avgwin   window size for average probability (slices)
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
from machine_learning.ml_tools import PredLoader
import pickle
import random
import time
import traceback

class G:
    lc = att.LabelCategories()
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None
    mapping_tab = []  # for plotting [(ltr, pos), ...]
    mapped_letters = []
    results = []  # ratings for blocks and iterations

plot_layout = {
    'm': (-50, "red", 16),   # mapped
    's': (-10, "green", 12),     # start search
    'f': (-10, "red", 12),     # failed to find letter
    'p': (-23, "darkviolet", 12),     # prediction (start/peak/end)
    #'k': (-36, "blue", 12),      # peak
    'r': ( 50, "maroon", 16)  # iter + rating
}
vowels = 'aiuAYNW*'


def main():
    colormix()

    stt = tbx.Statistics()
    recd, chapno, blkno = dialog.recording, dialog.chapno, dialog.blkno
    if ':' in blkno:
        blk1, blk2 = blkno.split(':')
        blk_range = range(int(blk1), int(blk2))
    else:
        blk_range = range(int(blkno), int(blkno)+1)

    model_name, iterations = dialog.name, int(dialog.iters)


    for blkno in blk_range:
        pred_loader = PredLoader(recd, chapno, blkno)

        # prepare stripe for text, loudness and frequency
        freq_vect, pars_ampl, rosa_ampl = att.load_freq_ampl(recd, chapno, blkno)
        freq_vect[freq_vect > 450] = 450  # limit max value to 300 Hz (numpy)

        pyaa_ampl = interpolate(pars_ampl, 600, 0)
        pyaa_ampl[pyaa_ampl < 0] = 0
        pyaa_ampl = pyaa_ampl / 200
        freq_vect = interpolate(freq_vect, 1, 0)

        xdim = len(freq_vect)  # duration of block determins the x dimension of chart
        xinch = xdim * 0.010  # zoom factor for x-axis (in inches)

        text = tt.get_block_text(recd, chapno, blkno)  # , spec="ml")
        print(f"db_text: [{text}] len: {len(text)}")

        text = text.strip('.').strip()  # remove '.', also remove space

        # letter_length = int((xdim - 800) / len(text))  # milliseconds per letter
        # print("letter_length", int(letter_length))

        text = adjust_timing(text)  # make text more linear in time

        # calulate relation between text and audio length

        letter_length = (xdim - 800) / len(text)  # milliseconds per letter
        print(f"adjusted: [{text}] len:{len(text)}")
        print(f"letter_length: {int(letter_length)} ms")

        texttab = []  # for plotting
        textpos = np.linspace(430, int(xdim - 450), len(text))
        for l, p in zip(text, textpos):
            texttab.append((l, p))

        # start plotting

        stripes = iterations + 1  # text and probabilities
        dimy = stripes * 3 + 2
        fig, ax = plt.subplots(stripes, 1, figsize=(xinch, dimy), dpi=100, gridspec_kw={})
        fig.tight_layout()
        G.colors = colormix()

        text_stripe = ax[0]  # stripe for loudness frequeny and text

        text_stripe.plot(pyaa_ampl, color="purple", linewidth=0.8)  # librosa rms

        text_stripe.plot(freq_vect, color="green", linewidth=0.8)
        text_stripe.set_xlim(0, xdim)
        for l, p in texttab:
            text_stripe.text(p, 20, l, color="black", fontsize=20)


        for iter in range(iterations):
            print(f"blkno: {blkno:03d}, iteration: {iter:2d}")
            #avg_pred = load_pred_data(recd, chapno, blkno)
            ltr_rows = pred_loader.get_calculated_predictions(model_name, iter)
            #print(ltr_rows)
            ltr_rows = np.array([interpolate(row) for row in ltr_rows])
            #print(ltr_rows)

            avg_tab = {}  # smoothed single letter probabilities

            for ndx, ltr_seq in enumerate(ltr_rows):
                # put the averaged probability curves into a new matrix
                letter = G.lc.ltrs[ndx]
                #ltr_seq_ms = interpolate(ltr_seq)
                avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
                avg_tab[letter] = avg



            avg_stripe = ax[iter+1]             # second stripe for the average of the remaining stripes
            avg_stripe.set_ylim(-70, 140)
            avg_stripe.set_xlim(0, xdim)
            start, stop = ax[1].get_xlim()
            ticks = np.arange(start, stop, 100)
            tlabels = [f"{int(n/100)}" for n in ticks]
            avg_stripe.set_xticks(ticks, labels=tlabels)


            try:
                ratg = map(avg_tab, text, letter_length, iter)
                G.results.append((blkno, iter, ratg))
            except Exception as excp:
                print("===== exception ====")
                traceback.print_exc()
                pass


            # print(f"average probabilities: {ltr_rows.shape}")

            plot_pred(ltr_rows, avg_stripe)      # plot the average
            for l, p in texttab:
                avg_stripe.text(p, -65, l, color="black", fontsize=20)

            # save final chart

        chart_name = f"{chapno:03d}_{blkno:03d}_{model_name}.png"
        chart = cfg.work / dialog.recording / 'charts' / chart_name
        plt.savefig(chart)
        print(f"chart is saved as {chart}")

    # end of all blocks
    write_results_file(chapno)

def write_results_file(chapno):
    # write rating results as a matrix
    blkdir = {}
    blks = set()
    iters = set()
    for blk, iter, ratg in G.results:
        blks.add(blk)
        iters.add(iter)
        if not blk in blkdir:
            blkdir[blk] = {}
        blkdir[blk][iter] = ratg
    itersum = {k:0 for k in iters}
    blksum = 0
    lines = []
    itstr = "  ".join([f"{it:6d}" for it in sorted(iters)])
    lines.append(f'blkno    avg  {itstr}')

    for blk in sorted(blks):

        for it in sorted(iters):
            ratg = blkdir[blk][it]
            blksum += ratg
            itersum[it] += ratg
        iterline = ' '.join([f"{blkdir[blk][it]:5.2f}  " for it in sorted(iters)])
        lines.append(f"  {blk:03d}  {blksum/len(iters):5.2f}    {iterline}")
    iterline = ' '.join([f" {itersum[it]/len(blks):5.2f} " for it in iters])
    lines.append(f"  avg:        {iterline}")

    result_fn = cfg.work / dialog.recording / f"results_{chapno}_{dialog.blkno}.txt"
    with open(result_fn, mode='w') as fo:
        for line in lines:
            fo.write(f"{line}\n")


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


def load_pred_data(recd, chapno, blkno, iter):
    # load the saved predicition data, which was calculated from average predictions
    full = cfg.data / recd / 'probs' / f"{chapno:03d}_{blkno:03d}.npy"
    data = np.load(full)
    print(f"loaded numpy data: {data.shape}")
    return data


def map(avg_tab, text, letter_length, iter):
    # avg_tab is a dictionary with one vector per letter
    # letter_length is approx. ms per letter
    # adjust = time_factor / 250

    G.mapping_tab = []
    ratg = 0
    ratg_found = 10
    ratg_notf  = -10
    ratg_good_len = 1
    ratg_len_limit = 0.7

    after_found  = int(60) # * adjust)

    limit_search = int(1.20 * letter_length)  # search is limited from curpos to some end position in ms
    not_found    = int(0.8 * letter_length)   # after not finding a letter, go forward to search the next letter
    restart      = int(-10) # * adjust)

    minlen       = int(0.4 * letter_length)  # after finding a letter, adjust the curpos for the next letter search
    #                        # this length is muiltiplied for repeated letters

    level = 0.07  # minimum probability, optimal is at 0.07


    print(f"text: [{text}]")
    curpos = 400   # Position is in ms

    ltr_start = 0  # Position, where found letter starts

    for ndx, ltr, repeat in unique_letters(text):

        print(f"from {curpos:5d} go for '{ltr}' ")
        G.mapping_tab.append((f"{ltr}>", curpos, 's'))
        seq = avg_tab[ltr]  # select right probabilitry vector for the letter


        span = repeat*minlen*1.3
        limit = curpos + span

        if curpos >= len(seq):
            result = None
        else:
            result = scan_predictions(seq, curpos, span, level)

        if result is None:
            G.mapping_tab.append((f"?{ltr}", limit, 'f'))
            curpos += not_found
            ratg += ratg_notf
            continue  # go to the next letter


        pl, pr, highest, highpos = result  # found letter positions

        G.mapping_tab.append((f"^", highpos, 'p'))
        ratg += ratg_found
        G.mapping_tab.append((f"({ltr}", pl , 'p'))
        ltr_len = pr - pl + 1  # add 1 to avoid divion by zero

        accuracy = (ltr_len / (minlen))  #(repeat*minlen))
        accuracy = accuracy if accuracy > 1 else 1 / accuracy
        if accuracy > ratg_len_limit:
            ratg += ratg_good_len


        ltrpos = highpos # (pr + pl)/2      corrected
        # print(f"end at {p: 5d} final pos: {ltrpos} repeat: {repeat}, at_least: {at_least}")
        G.mapping_tab.append((f"{ltr})", pr-20, 'p'))
        G.mapping_tab.append((ltr, ltrpos, 'm'))

        # after the end of the letter is found, where to continue next search?
        # pl is where this letter started, pr is where it ended
        pos1 = int(pl + (repeat-1) * letter_length)
        pos2 = int(pr - 50)
        curpos = max(pos1, pos2) # need tuning

    ratg_final = ratg / len(text)
    print(f"\nratg: {ratg}, text: {len(text)} - final rating: {ratg_final}")
    G.mapping_tab.append((f"iter:{iter}, ratg:{ratg_final:5.2}", 20, "r"))

    return ratg_final

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


def scan_predictions(seq, pos, span, level):
    # seq = prodiction vector for the given letter
    # pos = position in vector where to search
    # minlen = minimum length of scanning - even when the predictions go (temporarily) down
    # level = minimum probability of the predictions to be observed
    highest = 0
    highpos = 0

    limit = pos + span
    if seq[pos] >= level:
        pl = scan_left(seq, pos, level)
        print(f"scan left from {pos}, found pl: {pl}")
    else:
        endpos = min(limit, len(seq)-1)
        pl = scan_right(seq, pos, endpos, level)

    if pl == 0:
        return None  # letter not found

    # pl is the position of the left boundary
    # now go for the right boundary 'pr'
    endpos = min(pl + span, len(seq)-1)
    pr, highest, highpos = scan_for_the_end(seq, pl, endpos, level)

    newlevel = highest * 0.3
    pl = scan_left(seq, highpos, newlevel)
    pr, highest, highpos = scan_for_the_end(seq, pl, endpos, newlevel)

    return pl, pr, highest, highpos


def scan_left(seq, pos, level):
    # call scan_left only if you are inside the letter
    while seq[pos] >= level:  # go left for the left boundary
        pos -= 1
    return pos


def scan_right(seq, pos, endpos, level):
    # call scan_right only if you are before the letter start
    while pos < endpos:  # go right for the left boundary
        if seq[pos] >= level:
            break
        pos += 1
    else:
        return 0
    return pos

def scan_for_the_end(seq, pos, endpos, level):
    # call scan_4_end to find the end of the letter
    # while minlen
    pr = 0
    highest = 0
    highpos = 0
    while True:
        prob = seq[pos]
        if prob > level or pos < endpos:
            pos += 1
            if prob > level:
                pr = pos # there is actually some probability here
            if prob > highest:
                highest = prob
                highpos = pos

        else:
            break
    return pr, highest, highpos


def plot_pred(ltr_rows, stripe, mapped=True):
    lines = []
    # ltr_rows = np.swapaxes(preds, 0, 1)
    for ndx, ltr_seq in enumerate(ltr_rows):
        letter = G.lc.ltrs[ndx]

        # print(letter, ltr_seq.shape, ltr_seq)

        avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)

        seq = []
        mapshow = True
        for pos, n in enumerate(avg):  # the probabilities for a given letter
            if n > 0.05:  #  append this probability
                seq.append((pos, int(n*100)))
            else:

                if len(seq) > 10:
                    # if this prediction segment is mapped, show it, else ignore it
                    # the mapped_letters come as [(ltr, start, end), ...]
                    seq_start = (pos - len(seq)) * 5  # begin of the sequence (probability curve)
                    seq_end = pos
                    # print(f"pred-sequence start: {seq_start}, end: {seq_end}")
                    for ltr, start, end in G.mapped_letters:  # the positions are in ms
                        if ltr != letter:
                            continue
                        middle = (start + end ) / 2
                        # print(f"mapped? start: {start}, end: {end}")
                        if seq_start <= start <= seq_end or seq_start <= end <= seq_end or seq_start <= middle <= seq_end:
                            # the sequences overlap, so this is a "mapped" sequence
                            mapshow = True
                            # print("    yesss")
                    if mapped:
                        if mapshow:
                            lines.append((letter, seq))
                    else:
                        if not mapshow:
                            lines.append((letter, seq))
                    mapshow = True

                seq = []

    plines = []
    for ltr, seq in lines:
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

    # print(f"plot mapped letters: {G.mapping_tab}")

    for ltr, pos, layout in G.mapping_tab:
        ypos, cor, size = plot_layout[layout]
        stripe.text(pos, ypos, ltr, color=cor, size=size)

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
