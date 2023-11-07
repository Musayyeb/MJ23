# python3
'''
    functions needed for synthesis
'''

from functools import reduce
import numpy as np


def get_sample_text(ldata, name):
    # the name refers to an entry in the sample text file
    full = ldata / 'synth' / 'sample.txt'
    with open(full, mode='r') as fi:
        data = fi.read()
    for line in data.splitlines():
        line = line.split('#')[0].strip()
        if line == '':
            continue
        short, what, text = line.split(None, 2)
        if name == short:
            text = text.strip('"')
            return text
    return None

def freq_transitions(vect):
    # generator gives places, where frequencies are switched on and off
    pfreq = 0
    for pos, f in enumerate(vect):
        if f > 0:
            if pfreq == 0:
                yield pos, "on "
                pfreq = f
        else:
            if pfreq > 0:
                yield pos, "off"
                pfreq = f
    yield 99999, "off"


def change_ampl(snip, fpms, lfact, rfact, rampsize, slopesize):
    # snippet is a numpy array. It has the wav frames
    # including the ramp size (half on each side)
    # slopesize (ms) is the part on both ends, where the adjustement happens
    total_length = snip.shape[0]
    rs2 = int(rampsize * fpms / 2)  # half of the ramp in frames
    slopeframes = int(slopesize * fpms)
    lramp = np.ones(rs2) * lfact

    lslope = np.linspace(lfact, 1.0, num=slopeframes)
    rslope = np.linspace(1.0, rfact, num=slopeframes)
    rramp = np.ones(rs2) * rfact
    print(f"change_ampl - bad length: {total_length} - {len(lramp)} - {len(lslope)} - {len(rslope)} - {len(rramp)} ")
    flat = np.ones(total_length - len(lramp) - len(lslope) - len(rslope) - len(rramp))

    #print(f"lengths: tot:{total_length} lr:{len(lramp)}, lsl:{len(lslope)}, flat:{len(flat)}, "
    #      f"rsl:{len(rslope)}, rr:{len(rramp)} ")

    change_vect = np.concatenate((lramp, lslope, flat, rslope, rramp))
    # print("change_vect", change_vect)
    new_vect = snip * change_vect
    new_vect = new_vect.astype(np.int16)

    return new_vect


def myrange(size, fr, to, endlim=False):
    # go linear
    delta = to - fr
    adjust = 1 if endlim else 0
    fact = delta / (size - adjust)
    r = [fr + x * fact for x in range(size)]
    return r



def limit_wave_amplitude(wav):
    # the values of wave frames may not exceed the amplitude limit of a 16-bit integer
    # here we check,how many frames are above/below and limit the final
    # frames to the allowed min/max
    # we also return the frame count, so the caller may decide, how to react
    lo, hi = -0x7fff - 1,  0x7fff

    count_hi = reduce(lambda x, y: x + 1 if y > hi else x, wav, 0)
    if count_hi:
        wav = [a if a <= hi else hi for a in wav]

    count_lo = reduce(lambda x, y: x + 1 if y < lo else x, wav, 0)
    if count_lo:
        wav = [a if a >= lo else lo for a in wav]

    return wav, count_hi + count_lo
