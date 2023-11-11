# python3
'''

    start mapping of predictions

    plot the predictions along with some audio attributes
'''

from splib.cute_dialog import start_dialog
import sys

class dialog:
    recording = "hus9h"
    chapno = 0
    blkno = 0
    name = ''
    iter = 0
    avgwin = 5

    layout = """
    title Nullmap - start mapping
    text     recording  recording id, example: 'hus1h'
    int      chapno    Chapter
    int      blkno     Block
    text     name      model name
    int      iter      model iteration

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
import traceback

class G:
    lc = att.LabelCategories()
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None
    mapping_tab = []  # for plotting [(ltr, pos), ...]
    mapped_letters = []

plot_layout = {
    'm': (-40, "red", 16),   # mapped
    's': (-10, "green", 10),     # start search
    'p': (-20, "purple", 10),     # prediction
    'e': (-30, "blue", 10),  # end of letter
}
vowels = 'aiuAYNW*'


def main():
    colormix()

    stt = tbx.Statistics()
    recd, chapno, blkno = dialog.recording, dialog.chapno, dialog.blkno
    model_name, iter = dialog.name, dialog.iter
    pred_loader = PredLoader(recd, chapno, blkno)

    # prediction data selection parameters:

    #avg_pred = load_pred_data(recd, chapno, blkno)
    print("model name", model_name)
    ltr_rows = pred_loader.get_calculated_predictions(model_name, iter)
    print(ltr_rows)
    ltr_rows = np.array([interpolate(row) for row in ltr_rows])
    print(ltr_rows)

    avg_tab = {}
    # ltr_rows = np.swapaxes(avg_pred, 0, 1)

    for ndx, ltr_seq in enumerate(ltr_rows):
        # put the averaged probability curves into a new matrix
        letter = G.lc.ltrs[ndx]
        #ltr_seq_ms = interpolate(ltr_seq)
        avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
        avg_tab[letter] = avg


    # prepare stripe for text, loudness and frequency
    freq_vect, pars_ampl, rosa_ampl = att.load_freq_ampl(recd, chapno, blkno)
    freq_vect[freq_vect > 450] = 450  # limit max value to 300 Hz (numpy)

    pyaa_ampl = interpolate(pars_ampl, 600, 0)
    pyaa_ampl[pyaa_ampl < 0] = 0
    pyaa_ampl = pyaa_ampl / 200
    freq_vect = interpolate(freq_vect, 1, 0)

    xdim = len(freq_vect)  # duration of block determins the x dimension of chart
    xinch = xdim * 0.010   # zoom factor for x-axis (in inches)

    text = tt.get_block_text(recd, chapno, blkno) #, spec="ml")
    print("db_text:", text)

    text = text.strip('.')

    # letter_length = int((xdim - 800) / len(text))  # milliseconds per letter
    # print("letter_length", int(letter_length))

    text = adjust_timing(text)  # make text more linear in time

    # calulate relation between text and audio length

    letter_length = (xdim - 800) / len(text)  # milliseconds per letter
    print("adjusted:", text)
    print("letter_length", int(letter_length))


    texttab = []  # for plotting
    textpos = np.linspace(430, int(xdim-450), len(text))
    for l, p in zip(text, textpos):
        texttab.append((l, p))

    # start plotting

    stripes = 3  # text and probabilities
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


    avg_stripe = ax[1]             # second stripe for the average of the remaining stripes
    avg_stripe.set_ylim(-45, 140)
    avg_stripe.set_xlim(0, xdim)



    #chart_name = f"{chapno:03d}_{blkno:03d}.png"
    #chart = cfg.work / dialog.recording / 'charts' / chart_name
    #plt.savefig(chart)
    #print(f"chart is saved as {chart}")

    try:
        map(avg_tab, text, letter_length)
    except Exception as excp:
        print("===== exception ====")
        traceback.print_exc()
        pass

    print("db_text:", text)

    print(f"average probabilities: {ltr_rows.shape}")

    plot_pred(ltr_rows, avg_stripe, mapped=True)      # plot the average

    avg_stripe = ax[2]             # second stripe for the average of the remaining stripes
    avg_stripe.set_ylim(-40, 140)
    avg_stripe.set_xlim(0, xdim)

    plot_pred(ltr_rows, avg_stripe, mapped=False)      # plot the average

    # save final chart
    start, stop = ax[1].get_xlim()
    ticks = np.arange(start, stop, 100)
    tlabels = [f"{int(n/100)}" for n in ticks]
    ax[0].set_xticks(ticks, labels=tlabels)
    ax[1].set_xticks(ticks, labels=tlabels)

    chart_name = f"{chapno:03d}_{blkno:03d}_{model_name}_{iter:02d}.png"
    chart = cfg.work / dialog.recording / 'charts' / chart_name
    plt.savefig(chart)
    print(f"chart is saved as {chart}")

    # plt.show()


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
    print("old:", text)
    print("new:", newtxt)
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


def count_identical_letters(text, ndx):
    this_letter = text[ndx]
    count = 0
    pos = ndx
    while pos < len(text) and text[pos] == this_letter:
        pos += 1
        count += 1
    return count

def map(avg_tab, text, letter_length):
    # avg_tab is a dictionary with one vector per letter
    # letter_length is approx. ms per letter
    # adjust = time_factor / 250

    ratg = 0
    ratg_found = 10
    ratg_notf  = -10
    ratg_good_len = 1
    ratg_len_limit = 0.7

    after_found  = int(60) # * adjust)
    limit_search = int(1.20 * letter_length)  # search is limited from curpos to some end position in ms
    not_found    = int(0.15 * letter_length)   # after not finding a letter, go forward to search the next letter
    restart      = int(-10) # * adjust)
    skip_dot     = int(5) # * adjust)
    double_ltr   = int(35) # * adjust)
    minlen       = int(0.85 * letter_length)  # after finding a letter, adjust the curpos for the next letter search
    #                        # this length is muiltiplied for repeated letters

    trigger = 0.07  # minimum probability, optimal is at 0.07

    #  take the original prediction, smooth out the probability value

    print(text)
    curpos = 300   # Position is in ms
    prev_ltr = ''
    repeat = 0

    ltr_start = 0  # Position, where found letter starts
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


        print(f"from {curpos:5d} ms scan for ltr '{ltr}' ", end=" ")
        G.mapping_tab.append((f"<{ltr}", curpos, 's'))
        seq = avg_tab[ltr]
        for p in range(curpos, min(curpos+limit_search, len(seq)-1)):
            #print(f"start probability for {ltr} @{p}: {seq[p]:4.2f}")

            if seq[p] < trigger:   # probability
                continue   # too low, repeat

            print(f"found at {p:5d}", end=" ")
            ratg += ratg_found
            G.mapping_tab.append((f"({ltr}", p , 'p'))
            ltr_start = p
            curpos = p  # current position is where the probability started
            break
        else:  # not found
            G.mapping_tab.append((f"?{ltr}", p, 'p'))
            curpos += not_found
            ratg += ratg_notf
            print()
            continue

        # search end of current letter

        at_least = repeat * minlen  # this is the expected length
        # at_least: the letter can not end before this
        for p in range(curpos, curpos + at_least*2):
            #print(f"end probability for {ltr} @{p}: {seq[p]:4.2f}")
            if seq[p] < trigger:
                break

        # probability fell below the trigger threshold
        ltr_len = p - ltr_start
        accuracy = (ltr_len / (repeat*minlen))
        accuracy = accuracy if accuracy > 1 else 1 / accuracy
        if accuracy > ratg_len_limit:
            ratg += ratg_good_len

        ltrpos = (p + ltr_start)/2
        print(f"end at {p: 5d} final pos: {ltrpos} repeat: {repeat}, at_least: {at_least}")
        G.mapping_tab.append((f"{ltr}>", p, 'e'))
        G.mapping_tab.append((ltr, ltrpos, 'm'))
        G.mapped_letters.append((ltr, ltr_start, p))

        # after the end of the letter is found, where to continue next search?
        # curpos is where this letter started, p is where it ended
        curpos += at_least

        repeat = 0

    ratg_final = ratg / len(text)
    print(f"ratg: {ratg}, text: {len(text)} - final rating: {ratg_final}")


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
    print("colormix, found:", len(cormix))
    cortab = {l : c for l, c in zip(G.lc.ltrs, cormix)}
    return cortab

main()
