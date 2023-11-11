# python3
'''

    This mapping approach is based on onemap.py

    plot the predictions along with some audio attributes

    Now: get a rating for the mapping

    run more than one model and compare the ratings

    Create one artificial rating, where "friendly letters" add up their probabilities
    This neighborly support is for "one" iteration
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus1h"
    chapno = ''
    blkno = ''
    name = ''
    iter = '0'
    avgwin = 20
    plot = False

    layout = """
    title map_nbr - Add neighborly support for certain letters
    text     recording  recording id, example: 'hus1h'
    text     chapno    Chapter or a list of chapter n:m
    text     blkno     Block or list of blocks 3:8 or '*'
    text     name      model name
    text     iter      one or more specific iterations
    bool     plot      flag for plotting

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
from machine_learning.ml_tools import PredLoader, get_training_specs
from dataclasses import dataclass
import json
import random
import time
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

vowels = 'aiuAYNW*'

# all ltrs: '.abdfhiklmnoqrstuwyzġšǧʻʼḍḏḥḫṣṭṯẓ*ALNWY'
support_weights = {0:3, 1:5, 2:8, 3:11, 4:15, 5:20, 6:30, 7:40, 8:50, 9:60}
friends = [ # the x-axis supports the y-axis
'''
_ b d k q r t ǧ ṭ l L
b _ 3 . 3 . . 3 3 . .
d 3 _ . 3 . 3 3 3 . .
k . . _ 3 . . . . . .
q 3 3 3 _ . . 3 3 . .
r . . . . _ . . . 2 2
t . 3 . . . _ . 3 . .
ǧ 3 3 . 3 . . _ 3 . .
ṭ 3 3 . 3 . 3 3 _ . .
l . . . . 2 . . . _ 4
L . . . . 2 . . . 4 _
''',
'''
_ a A * i y Y u w W
a _ 5 4 . . . . . .
A 5 _ 4 . . . . . .
* 4 4 _ . . . . . .
i . . . _ 3 3 . . .
y . . . 3 _ 5 . . .
Y . . . 3 5 _ . . .
u . . . . . . _ 3 3
w . . . . . . 3 _ 4
W . . . . . . 3 4 _
''',
'''
_ m n N
m _ 3 2
n 3 _ 4
N 2 4 _
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
    stt = tbx.Statistics()
    G.colors = colormix()
    results = []
    G.support_list = parse_support()  # collect all supporting neighbor relations into one list
    print("support list: ", G.support_list)


    recd, chap_no, blk_no = dialog.recording, dialog.chapno, dialog.blkno

    G.blk_counter = tbx.get_block_counters(recd)
    if ':' in chap_no:  # chap_no is either a single number or a range of n:m
        chp1, chp2 = chap_no.split(':')
        chp_range = range(int(chp1), int(chp2))
    else:
        chp_range = range(int(chap_no), int(chap_no)+1)

    for chapno in chp_range:

        if blk_no == '*':
            blk_range = range(1, G.blk_counter[chapno]+1)
        elif ':' in blk_no:  # blockno is either a single number or a range of n:m
            blk1, blk2 = blk_no.split(':')
            blk_range = range(int(blk1), int(blk2))
        else:
            blk_range = range(int(blk_no), int(blk_no)+1)

        model_name, iters = dialog.name, dialog.iter

        if iters == '*':
            specs = get_training_specs(recd, model_name)
            itno_list = range(specs.iter)
        else:
            tok = iters.split()  # iters is either a single number or a sequence of numbers (split by space)
            itno_list = [int(iters)] if len(tok) == 1 else [int(x) for x in tok]


        # this program processes n blocks in sequence
        for blkno in blk_range:

            blk_results = []
            pred_loader = PredLoader(recd, chapno, blkno)
            vect_loader = att.VectLoader(recd, chapno, blkno)
            xdim = vect_loader.get_xdim()  # duration of block determins the x dimension of chart

            text, letter_length = prepare_text(recd, chapno, blkno, xdim)

            # for each block, mapping is done with different iterations (ML models)
            for itno in itno_list:

                ltr_rows = pred_loader.get_calculated_predictions(model_name, itno)
                # the predictions are for 5ms intervals - to avoid confusion, convert the data into 1ms
                ltr_rows = np.array([interpolate(row) for row in ltr_rows])
                for row in ltr_rows:   # each row has some noise: flatten 350 ms at both ends
                    row[0:350] = 0
                    row[-350:] = 0

                sup_rows = add_nbr_support(ltr_rows)
                # reversed_rows = np.flip(new_rows, axis=1)  # dont use the reverse approach

                # for each iteration of the model,
                #     try a mapping on the original predictions (ltr_rows)
                #     and on the supported predictions  (sup_rows)
                alt_mapping = (("orig",ltr_rows), ("supp", sup_rows))
                for mode, preds in alt_mapping:
                    print(f"blkno: {blkno:03d}, iteration: {itno:2d}, mode: {mode} predid:{id(preds)}")

                    avg_tab = calculate_predictions_average(preds)
                    try:
                        mappings, ratg = map(avg_tab, text, letter_length, xdim)
                    except Exception as excp:
                        print("===== exception ====")
                        traceback.print_exc()
                        continue

                    block_info = AD(chapno=chapno, blkno=blkno, itno=itno, mode=mode,
                                    ratg=ratg, txtlen=len(text), mslen=xdim)
                    blk_results.append((block_info, avg_tab, mappings))

            if dialog.plot:
                plot(blk_results, text, xdim, vect_loader, model_name)

            fn = cfg.data / recd / 'mapping_results' /f"pred_results {chapno:03d}.{blkno:03d}.json"
            json_results = [[blk,map] for blk, pred, map in blk_results]
            with open(fn, mode='w') as fo:
                json.dump(json_results, fo)

            results += blk_results

    print_mapping_results(results)



def print_mapping_results(results):
    # there are mappings from more than one pass (stripes)
    print()
    print("Mapping Results:")

    sorted_map = []
    for blk_info, preds, mappings in results:
        bi = blk_info
        # {'chapno': 20, 'blkno': 5, 'itno': 2, 'mode': 'orig', 'ratg': 3.3159522024253163,
        # 'txtlen': 34, 'mslen': 7990}
        # [{'ltr': 'ʼ', 'lndx': 0, 'repeat': 1, 'search': 350, 'lpos': 341, 'rpos': 364,
        # 'ratg': 5.113207547169812, 'limit': 562, 'level': 0.11723357597684653, 'hipos': 353}, ...]

        #for chapno, blkno, itno, mode, ratg, mapping in results:
        sorted_map.append((bi.chapno, bi.blkno, "A", bi.itno, bi.mode, bi.ratg))

        print()
        print(bi.chapno, bi.blkno, bi.itno, bi.mode, bi.ratg)
        print("    ltr rpt ndx   lpos  rpos  rating")
        for x in mappings:
            print(f"     {x.ltr}   {x.repeat}   {x.lndx:2d}  {x.lpos:5d} {x.rpos:5d}  {x.ratg:5.2f}")
            sorted_map.append((bi.chapno, bi.blkno, 'B', x.lndx, x.ltr, x.repeat, bi.itno,
                               bi.mode, x.lpos, x.rpos))

    sorted_map.sort()
    print("\n--------------------------------------------------------")
    for x in sorted_map:
        if x[2] == 'A':
            chapno, blkno, _, itno, mode, ratg = x
            print(f"ch:{chapno:3d} bl:{blkno:3d} it:{iter} {mode} ratg: {ratg:5.2f}")

        if x[2] == 'B':
            _, _, _, lndx, ltr, repeat, itno, mode, lpos, rpos = x
            llen = rpos - lpos
            print(f"'{ltr}'({repeat}) ndx:{lndx:2d}  it:{itno} {mode}  pos: {lpos:5d},{rpos:5d}  l:{llen:4d}")

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

def map(avg_tab, text, letter_length, xdim):
    # avg_tab is a dictionary with one vector per letter
    # letter_length is approx. ms per letter
    # adjust = time_factor / 250

    mapping_tab = []
    ratg = 0  # total rating for this block
    # points for the rating observations
    ratg_found = 5
    ratg_notf  = 0

    after_found  = int(60) # * adjust)
    limit_search = int(1.0 * (letter_length+1))  # search is limited from curpos to some end position in ms
    not_found    = int(1.0 * (letter_length+1))   # after not finding a letter, go forward to search the next letter
    restart      = int(-10) # * adjust)
    minlen       = int(1.0 * (letter_length+1))  # after finding a letter, adjust the curpos for the next letter search
    #                                        # this length is multiplied for repeated letters

    level = 0.07  # minimum probability, optimal is at 0.07

    print(f"text: [{text}]")
    curpos = 400   # Position is in ms


    for ndx, ltr, repeat in unique_letters(text):
        seq = avg_tab[ltr]  # select right probability vector for the letter

        ltrspec = AD(ltr=ltr, lndx=ndx, repeat=repeat, search=curpos, lpos=0, rpos=0, ratg=ratg_notf)
        mapping_tab.append(ltrspec)

        span = int((repeat-1)*letter_length+limit_search)
        ltrspec.limit = int(curpos + span)

        result = None if curpos >= len(seq) else scan_predictions(ltr, seq, curpos, span, level)

        if result is None:
            curpos += not_found
            ratg += ratg_notf
            continue  # go to the next letter


        pl, pr, highest, hipos = result  # found letter positions
        ltrspec.lpos = pl
        ltrspec.rpos = pr
        ltrspec.level = highest
        ltrspec.hipos = hipos

        # adjust the rating
        accuracy = 0
        if pr > 0:
            ltr_len = pr - pl + 1  # add 1 to avoid divion by zero
            accuracy = (ltr_len / (repeat*minlen))
            accuracy = accuracy if accuracy < 1 else 1 / accuracy
        r = ratg_found + accuracy
        ratg += r
        ltrspec.ratg = r


        # after the end of the letter is found, where to continue next search?
        # pl is where this letter started, pr is where it ended
        pos1 = int(pl + ((repeat-1) * letter_length))
        #pos2 = (pos1 + letter_length)
        pos3 = pr - 20
        curpos =  max(pos1, pos3) # pos1 + int(minlen/2)    # int((pr + pl)/2)  # max(pos1, pos2) # need tuning

    ratg_final = ratg / len(text)

    return mapping_tab, ratg_final


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


def calculate_predictions_average(np_mat):
    # calculate a smoothed curve for the predictions
    # also turn the data into a letter-based dictionary
    avg_tab = {}  # smoothed single letter probabilities

    for ndx, ltr_seq in enumerate(np_mat):
        # put the averaged probability curves into a new matrix
        letter = G.lc.ltrs[ndx]
        # ltr_seq_ms = interpolate(ltr_seq)
        avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
        avg_tab[letter] = avg

    return avg_tab

plot_layout = {
    'm': (-50, "red", 16),   # mapped
    's': (-10, "green", 12),     # start search
    'f': (-10, "red", 12),     # failed to find letter
    'p': (-23, "darkviolet", 12),     # prediction (start/peak)
    #'k': (-36, "blue", 12),      # peak
    'e': (-36, "magenta", 12),     # prediction (end)
    #'k': (-36, "blue", 12),      # peak
    'r': ( 50, "maroon", 16)  # iter + rating
}

def plot(results, text, xdim, vect_loader, model_name):
    xinch = xdim * 0.010  # zoom factor for x-axis (in inches)

    texttab = []  # for plotting
    textpos = np.linspace(430, int(xdim - 430), len(text))
    for l, p in zip(text, textpos):
        texttab.append((l, p))   # original text letters

    stripes = 1 + len(results)  # (text & amplitudes) + n * (original predictions, supported predictions)
    dimy = stripes * 3 + 2
    fig, ax = plt.subplots(stripes, 1, figsize=(xinch, dimy), dpi=100, gridspec_kw={})
    fig.tight_layout()

    stripe = ax[0]

    stripe.set_xlim(0, xdim)
    start, stop = stripe.get_xlim()
    ticks = np.arange(start, stop, 100)
    tlabels = [f"{int(n / 100)}" for n in ticks]
    stripe.set_xticks(ticks, labels=tlabels)
    plot_vectors(vect_loader, stripe, texttab)  # plot loudness, freq

    for rx, (blkinfo, preds, mappings) in enumerate(results):
        bi = blkinfo
        stripe_x = rx + 1

        stripe = ax[stripe_x]

        # now we have predictions, stripe and the mapping

        stripe.set_ylim(-70, 140)
        stripe.set_xlim(0, xdim)
        stripe.set_xticks(ticks, labels=tlabels)

        print(f"plot predid:{id(preds)} stripe: {stripe_x} iter {bi.itno} mode {bi.mode}")
        plot_pred(preds, stripe)  # plot the average

        for l, p in texttab:
            stripe.text(p, -65, l, color="black", fontsize=20)

        ypos, cor, size = plot_layout['r']  # ratings
        text = f" iter:{bi.itno} mode:{bi.mode} ratg:{bi.ratg:5.2}"
        stripe.text(20, ypos, text, color=cor, size=size)

        # single mapping {'ltr': 'u', 'lndx': 5, 'repeat': 1, 'search': 991, 'lpos': 1195, 'rpos': 1280,
        # 'ratg': 5.44559585492228, 'limit': 1242, 'level': 0.1887024748041657, 'hipos': 1250}
        for m in mappings:  # get the mapping values for each letter


            #for ltr, pos, layout in tab:

            ypos, cor, size = plot_layout['s']
            pos = m.search
            text = f"{m.ltr}>"
            stripe.text(pos, ypos, text, color=cor, size=size)

            if m.lpos:
                ypos, cor, size = plot_layout['p']
                pos = m.lpos
                text = f"({m.ltr}"
                stripe.text(pos, ypos, text, color=cor, size=size)

            if m.rpos:
                ypos, cor, size = plot_layout['e']
                pos = m.rpos
                text = f"{m.ltr})"
                stripe.text(pos, ypos, text, color=cor, size=size)

            if m.lpos and m.rpos:
                ypos, cor, size = plot_layout['m']
                pos = int((m.lpos + m.rpos) / 2)
                text = f"{m.ltr}"
                stripe.text(pos, ypos, text, color=cor, size=size)




    chart_name = f"{bi.chapno:03d}_{bi.blkno:03d}_{model_name}.png"
    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print(f"chart is saved as {chart}")


def plot_vectors(vector_loader, stripe, texttab):
    pyaa, rosa, freq = vector_loader.get_vectors()
    stripe.plot(pyaa, color="purple", linewidth=1.1)  # pyaa amplitude
    stripe.plot(rosa, color="black", linewidth=1.1)  # librosa rms

    stripe.plot(freq, color="green", linewidth=1.1)
    xdim = vector_loader.get_xdim()

    stripe.set_ylim(0, 300)
    stripe.set_xlim(0, xdim)
    for l, p in texttab:
        stripe.text(p, 20, l, color="black", fontsize=20)


def add_nbr_support(ltr_rows):
    new_rows = np.copy(ltr_rows)
    print("new_rows:", new_rows.shape)
    for nbr, benf, pcnt in G.support_list:
        nbrx = G.lc.categ[nbr]
        benx = G.lc.categ[benf]
        # print(f"nbr_supp: {nbr}:{nbrx}, {benf}:{benx}")
        new_rows[benx] += ltr_rows[nbrx]*pcnt
    return new_rows


def write_results_file(chapno):
    # write rating results as a matrix
    blkdir = {}
    blks = set()
    itnos = set()
    for blk, itno, ratg, mapping in G.results:
        blks.add(blk)
        itnos.add(itno)
        if not blk in blkdir:
            blkdir[blk] = {}
        blkdir[blk][itno] = ratg
    itersum = {k:0 for k in itnos}
    blksum = 0
    lines = []
    itstr = "  ".join([f"{it:6d}" for it in sorted(itnos)])
    lines.append(f'blkno    avg  {itstr}')

    for blk in sorted(blks):

        for itno in sorted(itnos):
            ratg = blkdir[blk][itno]
            blksum += ratg
            itersum[itno] += ratg
        iterline = ' '.join([f"{blkdir[blk][it]:5.2f}  " for it in sorted(itnos)])
        lines.append(f"  {blk:03d}  {blksum/len(itnos):5.2f}    {iterline}")
    iterline = ' '.join([f" {itersum[it]/len(blks):5.2f} " for it in itnos])
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


def plot_pred(ltr_rows, stripe, mapped=True):
    lines = []
    # ltr_rows = np.swapaxes(preds, 0, 1)
    for ndx, (ltr, ltr_seq) in enumerate(ltr_rows.items()):
        letter = G.lc.ltrs[ndx]

        # print(letter, ltr_seq.shape, ltr_seq)

        #avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)

        seq = []
        mapshow = True
        for pos, n in enumerate(ltr_seq):  # the probabilities for a given letter
            if n > 0.10:  #  append this probability
                seq.append((pos, int(n*100)))
            else:

                if len(seq) > 30:
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

