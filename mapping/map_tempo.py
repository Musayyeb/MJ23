# python3
'''
    Lets go back to timing: units, peaks and pits

'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus1h"
    chapno = ''
    blkno = ''
    name = ''
    iters = '0'
    db_write = False
    avgwin = 20
    pnpwin = 30
    level = 0
    bandwidth = 0
    plot = False

    layout = """
    title map_multi - while mapping, observe multiple iterations
    text     recording  recording id, example: 'hus1h'
    text     chapno    Chapter or a list of chapter n:m
    text     blkno     Block or list of blocks 3:8 or '*'
    text     name      model name
    text     iters     one or more specific iterations
    label    the code does not overwrite existing database entries
    bool     db_write  write the database - uncheck for testing 
    bool     plot      generate plots (expensive)

    int      avgwin   window size for average probability (slices)
    int      pnpwin   window size for peak'n'pit
    int      level    level for calc'n peak'n'pit
    int      bandwidth hysteresis for peak'n'pit
     
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
from dataclasses import dataclass, field
from typing import List
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
_ b d k q r t ǧ ṭ l L ġ
b _ 3 . 3 . . 3 3 . . 1
d 3 _ . 3 . 3 3 3 . . 1
k . . _ 3 . . . . . . 3
q 3 3 3 _ . . 3 3 . . 3
r . . . . _ . . . 2 2 .
t . 3 . . . _ . 3 . . .
ǧ 3 3 . 3 . . _ 3 . . .
ṭ 3 3 . 3 . 3 3 _ . . .
l . . . . 2 . . . _ 4 .
L . . . . 2 . . . 4 _ .
ġ . . 3 3 1 1 . . . . _
''',
'''
_ a A * i y Y u w W
a _ 5 4 . . . . . .
A 7 _ 4 . . . . . .
* 4 4 _ . . . . . .
i . . . _ 3 3 . . .
y . . . 3 _ 5 . . .
Y . . . 3 5 _ . . .
u . . . . . . _ 3 3
w . . . . . . 5 _ 4
W . . . . . . 3 2 _
''',
'''
_ m n N w h ʼ ḥ
m _ 3 2 . . . .
n 3 _ 4 . . . .
N 2 4 _ . . . .
w . . . _ 3 3 .
h . . . 3 _ 3 3
ʼ . . . 3 3 _ .
ḥ . . . . 3 . _
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
    #mm = Mimax(9999)
    #for v in [0.2, 0.1, 0.3, 0.2, 0.4, 0.6, 0.3]:
    #    mm.add(v)
    #print(mm.show())
    #return

    G.colors = colormix()
    G.runtoken = tbx.RunToken('map_multi.run_token')
    G.support_list = parse_support()  # collect all supporting neighbor relations into one list
    print("support list: ", G.support_list)

    pdb.db_connector(db_worker)

def db_worker(dbman):

    stt = tbx.Statistics()

    dbref = dbman.connect(dialog.recording, 'proj')
    conn = dbref.conn

    results = []

    recd, chap_seq, blk_seq = dialog.recording, dialog.chapno, dialog.blkno

    model_name, iters = dialog.name, dialog.iters
    if iters == '*':
        specs = get_training_specs(recd, model_name)
        itno_list = range(specs.iter)
    else:
        tok = iters.split()  # iters is either a single number or a sequence of numbers (split by space)
        itno_list = [int(iters)] if len(tok) == 1 else [int(x) for x in tok]

    for chapno, blkno in tbx.chap_blk_iterator(recd, chap_seq, blk_seq):

        if G.runtoken.check_break():
            break

        # if block is already done - execute anyway, if we dont plan to write the databse
        if check_block(conn, recd, chapno, blkno) and dialog.db_write:
            print(f"skipped prosessing for existing database data {chapno:03d}_{blkno:03d}")
            continue

        print(f"block: {chapno:03d}_{blkno:03d}")
        blk_results = []
        pred_loader = PredLoader(recd, chapno, blkno)


        # get the loudness curve and the peaks'n'pits
        vect_loader = att.VectLoader(recd, chapno, blkno)
        xdim = vect_loader.get_xdim()  # duration of block determins the x dimension of chart
        pyaa, orosa, freq = vect_loader.get_vectors()  # we use the librosa rms
        rosa = st.running_mean(orosa, dialog.pnpwin, 3)  #

        text, letter_length = prepare_text(recd, chapno, blkno, xdim)

        pnp = get_peaks_and_pits(rosa, level=dialog.level, bandwidth=dialog.bandwidth)
        # print(pnp)
        # pnp = reduce(pnp)
        # print("new:", pnp)

        pred_list = calc_pred_list(pred_loader, model_name, itno_list)


        mapping = map_pnp(text, pnp, pred_list, xdim)

        plottxt = []

        for pos, ltr, maxprob, avgprob in mapping:
            plottxt.append((pos, 530, ltr, "red"))
            if maxprob > 0.01:
                plottxt.append((pos, maxprob*480, 'x', "maroon"))
            if avgprob > 0.01:
                plottxt.append((pos, avgprob*480, 'o', "blue"))

        for pos, lbl, amp in pnp:
            plottxt.append((pos, -30, lbl, 'black'))
        lsp = np.linspace(300, xdim-300, len(text))

        for ltr, pos in zip(text, lsp):
            plottxt.append((pos, 600, ltr, "black"))

        pnp_plot(chapno, blkno, rosa, plottxt, xdim)

    return


def pnp_plot(chapno, blkno, seq, text, xdim):
    dimx = int(xdim/ 500)
    plt.figure(figsize=(dimx, 5))
    plt.title(f"{chapno:03d}_{blkno:03d} pnpwin: {dialog.pnpwin} level:{dialog.level} bandwidth: {dialog.bandwidth}")
    plt.tight_layout()
    plt.ylim(-50, 660)

    plt.plot(seq)
    for xpos, ypos, lbl, cor in text:
        plt.text(xpos,ypos,lbl, size=12, color=cor)
    # plt.show()
    chart_name = f"{chapno:03d}_{blkno:03d}_{dialog.pnpwin}_{dialog.level}_{dialog.bandwidth}.png"
    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print(f"chart is saved as {chart}")

def map_pnp(text, pnp, pred_list, xdim):
    # map letter acording to peaks, pits and timing
    mapping = []
    letter_length = (xdim - 800) / len(text)  # milliseconds per letter
    print(f"map_text: ({len(text)}) {text}")
    print(f"xdim: {xdim} ltrlen: {letter_length}")
    curpos = 350
    for lndx, ltr in enumerate(text):
        span = letter_length * 1.3
        if lndx == 0:
            continue
        if isvow(ltr):
            start, end = curpos , curpos + span
            pos, amp = hipeak(pnp, start, end)
            if not pos:
                print(f"not found {ltr}, {lndx}, st:{start} end:{end}")
                mapping.append((curpos+letter_length*0.5, '?', 0, 0))
                curpos += letter_length
                continue

        if iscons(ltr):
            pos, amp = lopit(pnp, curpos , curpos + span)
            if not pos:
                print(f"not found {ltr}, {lndx}, st:{start} end:{end}")
                mapping.append((curpos+letter_length*0.5, '?', 0, 0))
                curpos += letter_length
                continue

        maxprob, avgprob = get_probs(pred_list, ltr, pos)
        mapping.append((pos, ltr, maxprob, avgprob))

        curpos = pos + 10

    return mapping

def get_probs(pred_list, ltr, pos):
    # get probabilities for letter at position
    maxprob = 0
    sumprob = 0
    for itx, (itno, mode, avg_tab) in enumerate(pred_list):
        seq = avg_tab[ltr]  # select right probability vector for the letter
        prob = seq[pos]
        sumprob += prob
        maxprob = max(maxprob, prob)
    avgprob = sumprob / len(pred_list)
    return maxprob, avgprob


def hipeak(pnp, start, end):
    hipos = 0
    hival = 0
    for pos, lbl, amp in pnp:
        if pos > end:
            break
        if pos < start:
            continue
        if lbl == 'k':
            if amp > hival:
                hipos = pos
                hival = amp
    return hipos, hival


def lopit(pnp, start, end):
    lopos = 0
    loval = 999
    for pos, lbl, amp in pnp:
        if pos > end:
            break
        if pos < start:
            continue
        if lbl == 't':
            if amp < loval:
                lopos = pos
                loval = amp
    return lopos, loval



    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def calc_pred_list(pred_loader, model_name, itno_list):
    # get the predictions
    pred_list = []
    print(f"use model {model_name} with [{itno_list}] iterations")
    for itno in itno_list:

        ltr_rows = pred_loader.get_calculated_predictions(model_name, itno)
        # the predictions are for 5ms intervals - to avoid confusion, convert the data into 1ms
        ltr_rows = np.array([interpolate(row) for row in ltr_rows])
        for row in ltr_rows:  # each row has some noise: flatten 350 ms at both ends
            row[0:350] = 0
            row[-350:] = 0

        # sup_rows = add_nbr_support(ltr_rows)
        # reversed_rows = np.flip(new_rows, axis=1)  # dont use the reverse approach

        # for each iteration of the model,
        #     try a mapping on the original predictions (ltr_rows)
        #     and on the supported predictions  (sup_rows)
        alt_mapping = (("orig", ltr_rows),)  # ("supp", sup_rows))
        for mode, preds in alt_mapping:
            #print(f"chapno: {chapno:03d}, blkno: {blkno:03d}, iteration: {itno:2d},  mode: {mode}")

            avg_tab = calculate_predictions_average(preds)
            pred_list.append((itno, mode, avg_tab))

    return pred_list


def old():
    while False:
        # for each block, mapping is done with different iterations (ML models)
        pred_list = []
        print(f"use model {model_name} with [{itno_list}] iterations")
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
                print(f"chapno: {chapno:03d}, blkno: {blkno:03d}, iteration: {itno:2d},  mode: {mode}")

                avg_tab = calculate_predictions_average(preds)
                pred_list.append((itno, mode, avg_tab))


        # all types of predictions are now lined up in one sequence
        try:
            mappings, ratg, plot_targets = map(pred_list, text, letter_length, xdim)
        except Exception as excp:
            print("===== exception ====")
            traceback.print_exc()
            continue

        if plot_targets and dialog.plot:
            plot_failed_mappings(plot_targets, pred_list, chapno, blkno)


        block_info = AD(chapno=chapno, blkno=blkno, ratg=ratg, txtlen=len(text), mslen=xdim)
        blk_results.append((block_info, mappings))

        fn = cfg.data / recd / 'mapping_results' /f"pred_results {chapno:03d}.{blkno:03d}.json"
        with open(fn, mode='w') as fo:
            json.dump(blk_results, fo)

        results += blk_results

        if dialog.db_write:
            confirm_block(conn, recd, chapno, blkno)
            conn.commit()
            write_letterbase(conn, recd, blk_results)
        else:
            print(blk_results)

    # print_mapping_results(results)
    results = []

    print(f"runtime: {stt.runtime()}")


def get_peaks_and_pits(loud, level=8, bandwidth=8):
    # This method only works for a loudness curve (all values positive) !
    # create a vector of the same length as the input loudness curve.
    # use this as input for charts!
    # insert the high turns as positive, and the low turns as negative peaks
    # the level indicates a minimum loudness for the high peaks
    # the bandwidth gives a percentage of the minimum change to recognize
    #       a new peak or pit

    print("the speech_aps algo")
    peaks = [0 for x in range(len(loud))]
    up = True
    pcnt_up   = (100 + bandwidth)/100
    pcnt_down = (100 - bandwidth)/100

    pnp = []
    band_high = level
    band_low = level
    for ndx, l in enumerate(loud):
        if l > 10:
            pnp.append((ndx, 't', l))
            break

    for ndx, (l1, l2) in enumerate(zip(loud, loud[1:])):
        if up and l1 > l2:
            up = False
            if l1 > level:
                if l1 > band_high:
                    # peaks[ndx] = l1
                    pnp.append((ndx, 'k', l1))
                    band_high = l1 * pcnt_up
                    band_low = l1 * pcnt_down
        elif l2 > l1:
            up = True
            if l1 < band_low:
                # peaks[ndx] = -l1
                pnp.append((ndx, 't', l1))
                band_high = l1 * pcnt_up
                band_low = l1 * pcnt_down

    lastpos = 0
    for ndx, l in enumerate(loud[-500:]):
        if l > 10:
            lastpos = ndx
    pos = len(loud) - 500 + lastpos
    pnp.append((pos, 't', 10))

    return pnp # peaks


def ___get_peaks_and_pits(seq, hyst=5):
    # seq is a list of amplitude values
    # hyst is a percentage of max peak as in indicator of a phase change
    pnp = []
    up = False
    for ndx, (a1, a2) in enumerate( zip(seq, seq[1:])):
        diff = abs(a1-a2)
        if up:
            if a1 > a2 and diff > 0.2:  # a1 is a peak
                pnp.append((ndx, 'k', round(a1, 2)))
                up = False
        else: # down
            if a1 < a2 and diff > 0.2:  # a1 is a pit
                pnp.append((ndx, 't', round(a1, 2)))
                up = True
    return pnp

def reduce(pnp):
    dpnp = []
    for (pos1, lbl1, a1), (pos2, lbl2, a2) in zip(pnp, pnp[1:]):
        d = pos2-pos1 + abs(a2-a1)
        dpnp.append((d, pos1, lbl1, a1))
    dpnp.append((9999, pos2, lbl2, a2))

    postab = [x[1] for x in dpnp]

    while True:
        mind = sorted(dpnp)
        print("mind", mind[0])
        diff, pos, lbl, a = mind[0]  # get the smallest difference
        if diff > 30:
            break
        ndx = postab.index(pos)
        print("index", ndx, 'for', dpnp[ndx])
        if ndx > 0:
            pdiff, ppos, plbl, pa = dpnp[ndx-1]
            dpnp[ndx-1] = (pdiff+diff, ppos, plbl, pa)
        del dpnp[ndx]
        del postab[ndx]

    newpnp = []
    for diff, pos, lbl, a in dpnp:
        newpnp.append((pos, lbl, a))

    lbltab = [x[1] for x in newpnp]
    for ndx in enumerate(lbltab):
        pass

    return newpnp


def __get_peaks_and_pits(seq, hyst=2):
    # seq is a list of amplitude values
    # hyst is a percentage of max peak as in indicator of a phase change
    maxpeak = np.max(seq)
    hyst = hyst /100 * maxpeak
    pnp = []
    curr = 'k'   # k = peak, t = pit
    prev = 0
    up = False
    reflvl = hyst
    for ndx, (a1, a2) in enumerate( zip(seq, seq[1:])):
        if up:
            if a1 > a2:
                print(ndx, "down",int(a1), int(a2), int(reflvl))
            if a1 > a2 and a1 > reflvl:  # a1 is a peak
                print("found", (ndx, 'k', int(a1)))
                pnp.append((ndx, 'k', int(a1)))
                up = False
                reflvl = a2 - hyst
        else: # down
            if a1 < a2:
                print(ndx, "up", int(a1), int(a2), int(reflvl))
            if a1 < a2 and a1 < reflvl:  # a1 is a pit
                pnp.append((ndx, 't', int(a1)))
                print("found", (ndx, 't', int(a1)))
                up = True
                reflvl = a2 + hyst
    return pnp



def write_letterbase(conn, recd, results):
    for blkinfo, mappings in results:
        print(f"write db {blkinfo.chapno}-{blkinfo.blkno} mappings: {len(mappings)}")
        cbkey = f"{blkinfo.chapno:03d}_{blkinfo.blkno:03d}"
        sql = 'DELETE FROM lettermap WHERE recd == ? and cbkey == ?'
        conn.execute(sql, (recd, cbkey))
        mappings[0].lgap = 0
        mappings[-1].rgap = 0
        for l1, l2 in zip(mappings, mappings[1:]):
            l1.rgap = l2.lgap = l2.lpos - l1.rpos
        sql = """INSERT INTO lettermap (ltr, recd, cbkey, lndx, rept, ratg, lpos, rpos, lgap, rgap)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        for ltr in mappings                                                         :
            conn.execute(sql, (ltr.ltr, recd, cbkey, ltr.lndx, ltr.repeat, ltr.ratg,
                               ltr.lpos, ltr.rpos, ltr.lgap, ltr.rgap))
    return
'''
    ltr    text  nn  # individual letter
    recd   text  nn  # recording
    cbkey  text  nn
    lndx   int   nn  # letter index of the first letter 
    rept   int   nn  # repetition: add this to the current lndx to get the next lndx
    ratg   float     # some value, which represents the 'quality' of the letter, higher is better
    lpos   int   nn  # left position (ms)
    rpos   int   nn  # right position (ms)
    lgap   int   nn  # gap to the previous letter (negative if overlap)
    rgap   int   nn  # gap to the next letter (negative if overlap)
    lampl  int       # amplitude (loudness) at the left boundary
    rampl  int       # amplitude (loudness) at the right boundary
    lfreq  int       # freq at the left boundary
    rfreq  int       # freq at the right boundary
    melody text      # text represents a number of intermediate frequencies

'''

def check_block(conn, recd, chapno, blkno):
    # return true if block is already in this database table
    sql = "SELECT blkno from mapping_done WHERE recd == ? and chapno == ? and blkno == ?"
    csr = conn.execute(sql, (recd, chapno, blkno))
    rows = csr.fetchall()
    return len(rows) > 0

def confirm_block(conn, recd, chapno, blkno):
    # insert this block to confirm, it is already done
    sql = "INSERT INTO mapping_done (recd, chapno,blkno) VALUES (?, ?, ?)"
    conn.execute(sql, (recd, chapno, blkno))

def __print_mapping_results(results):
    # there are mappings from more than one pass (stripes)
    print()
    print("Mapping Results:", len(results))

    sorted_map = []
    for blk_info, mappings in results:
        bi = blk_info
        # {'chapno': 20, 'blkno': 5, 'itno': 2, 'mode': 'orig', 'ratg': 3.3159522024253163,
        # 'txtlen': 34, 'mslen': 7990}
        # [{'ltr': 'ʼ', 'lndx': 0, 'repeat': 1, 'search': 350, 'lpos': 341, 'rpos': 364,
        # 'ratg': 5.113207547169812, 'limit': 562, 'level': 0.11723357597684653, 'hipos': 353}, ...]

        #for chapno, blkno, itno, mode, ratg, mapping in results:
        # print(f"bi: {bi}")
        sorted_map.append((bi.chapno, bi.blkno, "A", bi.ratg))

        # print(bi.chapno, bi.blkno, bi.ratg)
        # print("    ltr rpt ndx   lpos  rpos  rating")
        for x in mappings:
            #print(f"     {x.ltr}   {x.repeat}   {x.lndx:2d}  {x.lpos:5d} {x.rpos:5d}  {x.ratg:5.2f}")
            sorted_map.append((bi.chapno, bi.blkno, 'B', x.lndx, x.ltr, x.repeat, x.itno, x.mode, x.lpos, x.rpos))

    sorted_map.sort()
    print("\n--------------------------------------------------------")
    prev_end = 0
    for x in sorted_map:
        if x[2] == 'A':
            chapno, blkno, _, ratg = x
            print(f"ch:{chapno:3d} bl:{blkno:3d} {ratg:5.2f}")

        if x[2] == 'B':
            _, _, _, lndx, ltr, repeat, itno, mode,  lpos, rpos = x
            llen = rpos - lpos
            gap = lpos - prev_end
            go = "GP" if gap > 0 else "OV"

            gostr = f"{go}: {abs(gap)}" if prev_end > 0 and rpos > 0 else ''
            prev_end = rpos

            print(f"'{ltr}'({repeat}) ndx:{lndx:2d} it:{itno:2d} mode:{mode[0]}  pos: {lpos:5d},{rpos:5d}  l:{llen:4d}  {gostr}")

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


def map(pred_list, text, letter_length, xdim):
    # pred list is a list of tuples: (itno, mode, avg_tab)
    # avg_tab is a dictionary with one vector per letter
    # letter_length is approx. ms per letter
    # xdim = total length in ms

    mapping_tab = []
    ratg = 0  # total rating for this block
    # points for the rating observations
    ratg_found = 5
    ratg_notf  = 0

    after_found  = int(60) # * adjust)
    limit_search = int(1.1 * (letter_length+1))  # search is limited from curpos to some end position in ms
    not_found    = int(1.0 * (letter_length+1))   # after not finding a letter, go forward to search the next letter
    restart      = int(-10) # * adjust)
    minlen       = int(1.0 * (letter_length+1))  # after finding a letter, adjust the curpos for the next letter search
    #                                        # this length is multiplied for repeated letters

    level = 0.01  # minimum probability, optimal is at 0.07

    print(f"text: [{text}]")
    curpos = 320  # Position is in ms - this is supposed to be the start of a letter

    plot_targets = []
    for lndx, ltr, repeat in unique_letters(text):  # text is linear text with repetitions of certain vowels and mosaics

        if curpos >= xdim: # xdim is the total time of the block
            break

        # calculate expected letter length by dividing the remaining time by the remaining letters
        ltrs = len(text) - lndx   # remaining letters to map
        remaining = xdim - 350 - curpos  # remaining time (ms) in the block
        ltr_time = remaining / ltrs

        ltrspec = AD(ltr=ltr, lndx=lndx, repeat=repeat, search=curpos, lpos=0, rpos=0,
                     ratg=ratg_notf, itno=-1, mode='x')
        mapping_tab.append(ltrspec)

        span = int(repeat * ltr_time + ltr_time * 0.1) # ooptimal is 0.4
        ltrspec.limit = int(curpos + span)
        targ_pos = curpos + 20 # + int(ltr_time / 2)

        #print(f"map ltr {ltr} *{repeat} ({lndx}) ltrs:{ltrs}  remain:{remaining} time={ltr_time:5.1f}  curpos:{curpos}  targ_pos:{targ_pos}")

        rate_it = PredictionRating(targ_pos, ltr_time, pos_weight=3, len_weight=5, prob_weight=8)

        # the avg_tab is a dictionary of all letters with their probability vectors
        for itx, (itno, mode, avg_tab) in enumerate(pred_list):
            seq = avg_tab[ltr]  # select right probability vector for the letter

            # find a given letter in these predictions (of one iteration)
            it_result = scan_predictions(itx, ltr, lndx, seq, curpos, span, level)

            if it_result:  # may be None
                # got this: AD(ltr, ltr_ndx, pl, pr, hiprob, hipos, avg_prob)
                it_result.pos = int((it_result.pr + it_result.pl) / 2)  # something like middle pos
                it_result.len = it_result.pr - it_result.pl
                it_result.itno = itno
                it_result.mode = mode

                rate_it.add_object(it_result)
                #ltr_preds.append((itno, mode, it_result))

        best, num_of_ratgs = rate_it.get_best_object()

        # if number of results is below some expectation, plot the actual predictions of all iterations
        # or at least of the 10 or 12 weakest iterations, to learn, where the models fail
        # here we just collect the places, which should go to the plot. The pred_list contains the data
        if num_of_ratgs < 3:
            plot_targets.append((ltr, repeat, lndx, curpos))

        if num_of_ratgs:
            ratg_str = f"pl:{best.pl:5d} pr:{best.pr:5d} "\
                      f"mpos:{best.pos:5d} len:{best.len:4d}  itno:{best.itno:2d} mode:{best.mode}  "\
                      f"ltr_ratg:{best.ltr_ratg:5.2f}  "\
                      f"pos/len/prob: {best.pos_ratg:5.2f}/{best.len_ratg:5.2f}/{best.prob_ratg:5.2f}  "\
                      f"avg_prob:{best.avg_prob:4.2f}  hi_prob:{best.hiprob:4.2f} "\
                      f"hipos:{best.hipos:5d}"
        else:
            ratg_str = ""
        print(f"best: {num_of_ratgs:2d}*ratgs ==> "
              f"ltr: {best.ltr} ({best.lndx:3d})  {ratg_str} ")


        if not num_of_ratgs:  # if we find no prediction at all!
            oldpos = curpos
            curpos += int(ltr_time * repeat)
            print(f"none - advance curpos {oldpos} by {ltr_time*repeat}  --> {curpos}")
            ratg += ratg_notf
            continue  # go to the next letter

        # {'pl', 'pr', 'hiprob', 'hipos', 'avg_prob', 'pos', 'len', 'itno', 'mode'}
        ltrspec.lpos = pl = best.pl
        ltrspec.rpos = pr = best.pr
        ltrspec.ltr_ratg = best.ltr_ratg
        ltrspec.itno = best.itno
        ltrspec.mode = best.mode

        # print(f"best: {pl, pr}, m:{int((pr+pl)/2)}  ratg:{best.ltr_ratg:5.2f}  it:{itno},{mode}")

        r = ratg_found + best.ltr_ratg

        ratg += r
        ltrspec.ratg = r


        # after the end of the letter is found, where to continue next search?
        # pl is where this letter started, pr is where it ended
        pos1 = int(pl + ((repeat-1) * letter_length))
        #pos2 = (pos1 + letter_length)
        pos3 = pr - 20
        curpos = max(pos1, pos3)    # int((pr + pl)/2)  # max(pos1, pos2) # need tuning

    ratg_final = ratg / len(text)

    return mapping_tab, ratg_final, plot_targets


class PredictionRating:
    def __init__(self, pref_pos, pref_len, pos_weight, len_weight, prob_weight):
        self.pref_pos = pref_pos  # this is, where the middle of the letter should be found
        self.pref_len = pref_len
        self.pos_weight = pos_weight
        self.len_weight = len_weight
        self.prob_weight = prob_weight
        self.obj_list = []

    def add_object(self, obj):
        # AD(pl, pr, hiprob, hipos, avg_prob, pos, len, itno, mode)
        self.obj_list.append(obj)
        return

    def rate_object(self, obj):
        # high rating is better
        pos_diff = max(10, abs(obj.pos-self.pref_pos))
        pos_ratg = 10 / pos_diff

        len_q = obj.len  / self.pref_len
        len_ratg = len_q if len_q < 1 else 1 / len_q  # best rating is close to 1.0

        prob_ratg = obj.avg_prob
        pos_score = pos_ratg * self.pos_weight
        len_score = len_ratg * self.len_weight
        prob_score = prob_ratg * self.prob_weight
        # best rating is high, up to 23 (depends on the weights)
        return pos_score, len_score, prob_score

    def get_best_object(self):
        if len(self.obj_list) == 0:
            return None, 0
        valid_objs = 0
        for obj in self.obj_list:
            if obj.pl:
                valid_objs += 1
                rtpos, rtlen, rtprob = self.rate_object(obj)
            else:
                rtpos, rtlen, rtprob = 0, 0, 0
            obj.pos_ratg = rtpos
            obj.len_ratg = rtlen
            obj.prob_ratg = rtprob
            obj.ltr_ratg = rtpos + rtlen + rtprob
        obj_rank = sorted(self.obj_list, reverse=True, key=lambda obj: obj.ltr_ratg)

        return obj_rank[0], valid_objs


def scan_predictions(itx, ltr, lndx, seq, pos, span, level):
    # seq = prediction vector for the given letter
    # pos = position in vector where to start the search
    # minlen = minimum length of scanning - even when the predictions go (temporarily) down
    # level = minimum probability of the predictions to be observed
    highest = 0
    highpos = 0

    limit = pos + span
    # if predictions overlap, after one letter we may be in(!) the next letter already
    # so eventually go back and search the beginning of this letter
    if seq[pos] >= level:
        pl = scan_left(ltr, seq, pos, level)
        if pl < (pos-20):
            pl = pos
    else:
        endpos = min(limit, len(seq)-1)  # don't run into the end of prediction vector
        pl = scan_right(ltr, seq, pos, endpos, level)

    if pl == 0:
        return AD(ltr=ltr, lndx=lndx, pl=0, pr=0, hiprob=0, hipos=0, avg_prob=0.0   )
        return None  # letter not found

    # pl is the position of the left boundary
    # now go for the right boundary 'pr'
    endpos = min(pl + span, len(seq)-1)
    pr, highest, highpos, avgprob = scan_for_the_end(ltr, seq, pl, endpos, level)

    return AD(ltr=ltr, lndx=lndx, pl=pl, pr=pr, hiprob=highest, hipos=highpos, avg_prob=avgprob)


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
    pl = pos # initial position = left boundary
    pr = 0
    highest = 0  # highest probability
    highpos = 0  # pos of the highest prob
    sumprob = 0  # add all probabilities to get the average
    while True:
        prob = seq[pos]
        if prob >= level:
            sumprob += prob
            pr = pos # there is actually some probability here
            if prob > highest:
                highest = prob
                highpos = pos
        pos += 1  # we continue through the sequence until endpos, even if the probabilities went down
        if pos > endpos:
            break
    length = pr-pl
    avgprob = sumprob/length if length else 0
    return pr, highest, highpos, avgprob


def calculate_predictions_average(np_mat):
    # calculate a smoothed curve for the predictions
    # also turn the data into a letter-based dictionary
    avg_tab = {}  # smoothed single letter probabilities

    for ndx, ltr_seq in enumerate(np_mat):
        # put the averaged probability curves into a new matrix
        letter = G.lc.ltrs[ndx]
        avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
        avg_tab[letter] = avg

    return avg_tab


def add_nbr_support(ltr_rows):
    new_rows = np.copy(ltr_rows)
    print("new_rows:", new_rows.shape)
    for nbr, benf, pcnt in G.support_list:
        nbrx = G.lc.categ[nbr]
        benx = G.lc.categ[benf]
        # print(f"nbr_supp: {nbr}:{nbrx}, {benf}:{benx}")
        new_rows[benx] += new_rows[nbrx]*pcnt
    return new_rows


def write_results_file(chapno):
    # write rating results as a matrix
    blkdir = {}
    blks = set()
    iters = set()
    for blk, itno, ratg, mapping in G.results:
        blks.add(blk)
        iters.add(itno)
        if not blk in blkdir:
            blkdir[blk] = {}
        blkdir[blk][itno] = ratg
    itersum = {k:0 for k in iters}
    blksum = 0
    lines = []
    itstr = "  ".join([f"{itno:6d}" for itno in sorted(iters)])
    lines.append(f'blkno    avg  {itstr}')

    for blk in sorted(blks):

        for itno in sorted(iters):
            ratg = blkdir[blk][itno]
            blksum += ratg
            itersum[itno] += ratg
        iterline = ' '.join([f"{blkdir[blk][itno]:5.2f}  " for it in sorted(iters)])
        lines.append(f"  {blk:03d}  {blksum/len(iters):5.2f}    {iterline}")
    iterline = ' '.join([f" {itersum[itno]/len(blks):5.2f} " for itno in iters])
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

def plot_failed_mappings(plot_targets, pred_list, chapno, blkno):
    vwins, hwins = 4, 3 # small plot windows for single failed mappings

    for trg in plot_targets:
        # (ltr, repeat, lndx, curpos)
        ltr, repeat, lndx, curpos = trg

        fig, ax = plt.subplots(vwins, hwins, figsize=(20, 12), dpi=100, gridspec_kw={})
        fig.tight_layout()

        for itx, (itno, mode, avg_tab) in enumerate(pred_list):
            if itx >= vwins*hwins:
                break
            # avg_tab is a dictionary, where all letters come with their probability vector over the length of the block
            vax = itx // hwins
            hax = itx %  hwins

            itwin = ax[vax, hax]  # each iteration gets its own little plot window
            ms_from, ms_to = curpos - 100, curpos + 600
            plotlen = ms_to - ms_from
            xvect = np.linspace(ms_from, ms_to, plotlen)
            itwin.set_ylim(-0.1, 1.4)
            itwin.set_xlim(ms_from, ms_to)

            for predltr, seq in avg_tab.items():
                #print(f"itwin for {chapno}_{blkno}  [{ltr}]*{repeat} ({lndx}) ==> {itltr} {itx} ax:{vax, hax}")
                cor = G.colors.get(predltr, "grey")

                #print(f"pred seq: {predltr} {cor} l={len(seq)}, seq:{seq[500:510]}")

                plotv = np.copy(seq[ms_from:ms_to])
                plotv[plotv < 0.05] = np.nan
                if np.nanmax(plotv) > 0.01:
                    lpos = np.argmax(seq[ms_from:ms_to])
                    ypos = 1 + 0.3*plotv[lpos]  # take the probaility at the strongest prediction to midify the y pos of the letter
                    xpos = lpos + ms_from
                    #print(f"plot text {lpos} {predltr}")
                    itwin.text(xpos, ypos, predltr, color=cor, fontsize=10)

                    itwin.plot(xvect, plotv, color=cor)

        #plt.show()
        chart_name = f"missing_maps {chapno:03d}_{blkno:03d}_{ltr}_{lndx}.png"
        chart = cfg.work / dialog.recording / 'charts' / chart_name
        print("saved as:", chart)
        plt.savefig(chart)

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
