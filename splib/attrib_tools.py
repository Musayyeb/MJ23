# python3
"""
    sound_tools.py

    Collection of functions to process audio files
"""

from config import get_config
cfg = get_config()
import numpy as np
from functools import cache
import os


@cache
def get_cached_vector(recd, chapno, blkno, vect_id):
    # prepare stripe for text, loudness and frequency
    freq_vect, pars_ampl, rosa_ampl = load_freq_ampl(recd, chapno, blkno)
    freq_vect[freq_vect > 450] = 450  # limit max value to 300 Hz (numpy)
    if vect_id == "pyaa":
        pyaa_ampl = interpolate(pars_ampl, 600, 0)
        pyaa_ampl[pyaa_ampl < 0] = 0
        vect = pyaa_ampl / 200
    elif vect_id == "rosa":
        vect = interpolate(rosa_ampl, 1200, 0)
    elif vect_id == 'freq':
        vect = interpolate(freq_vect, 1, 0)

    return vect



class VectLoader:
    # load the amplitude and frequency vectors
    # using a class makes sense, because the xdim is needed befor the vectors are needed
    def __init__(self, recd, chapno, blkno):
        # prepare stripe for text, loudness and frequency
        freq_vect, pars_ampl, rosa_ampl = load_freq_ampl(recd, chapno, blkno)
        freq_vect[freq_vect > 450] = 450  # limit max value to 300 Hz (numpy)

        pyaa_ampl = interpolate(pars_ampl, 600, 0)
        pyaa_ampl[pyaa_ampl < 0] = 0
        self.pyaa_ampl = pyaa_ampl / 200
        self.rosa_ampl = interpolate(rosa_ampl, 1200, 0)
        self.freq_vect = interpolate(freq_vect, 1, 0)

        xdim = len(freq_vect)  # duration of block determins the x dimension of chart

    def get_xdim(self):
        return len(self.freq_vect)

    def get_vectors(self):
        return self.pyaa_ampl, self.rosa_ampl, self.freq_vect

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


def save_freq_ampl(recd, chap, blkno, freq_vect, ampl_vect):

    # combine the 2 arrays
    data = np.stack((freq_vect, ampl_vect))
    print("hstack", data.shape)

    fn = f'freq_ampl_{chap:03d}_{blkno:03d}.npy'
    path = str(cfg.attribs / "freq_ampl").format(recd=recd)
    full = os.path.join(path, fn)

    np.save(full, data)

class AllVectors():
    def __init__(self, recd, chapno, blkno):
        self.vectlist = []
        self.namelist = []

        data = load_freq_ampl(recd, chapno, blkno)
        self.vectlist.extend(data)
        names = 'pars_freq pars_ampl rosa_ampl'.split()
        self.namelist.extend(names)

        data = load_pyaa_other(recd, chapno, blkno)
        self.vectlist.extend(data)
        names = '''zcr enrg enrg_entr spec_cent spec_sprd
                   spec_entr spec_flux spec_rlof'''.split()
        names = ['pyaa_'+s for s in names]
        self.namelist.extend(names)

        data = load_pyaa_mfcc(recd, chapno, blkno)
        self.vectlist.extend(data)
        names = [f'pyaa_mfcc_{n}' for n in range(1,14)]
        self.namelist.extend(names)

        data = load_pyaa_chrom(recd, chapno, blkno)
        self.vectlist.extend(data)
        names = [f'pyaa_chrm_{n}' for n in range(1,13)]
        names.append('pyaa_chrm_std')
        self.namelist.extend(names)

        data = load_pars_fmnts(recd, chapno, blkno)
        self.vectlist.extend(data)
        names = [f'pars_fmnt_{n}' for n in range(1,8)]
        self.namelist.extend(names)

    def get_values(self, mspos):
        slice = round(mspos/5)
        values = [v[slice] for v in self.vectlist]
        return values

    def get_names(self):
        return self.namelist

    def get_vectors(self):
        return self.vectlist


def load_freq_ampl(recd, chap, blkno):
    fn = f'freq_ampl_{chap:03d}_{blkno:03d}.npy'
    path = str(cfg.attribs / "freq_ampl").format(recd=recd)
    full = os.path.join(path, fn)
    data = np.load(full)
    freq_vect, pars_ampl, rosa_ampl = data

    return freq_vect, pars_ampl, rosa_ampl

def load_pyaa_other(recd, chap, blkno):
    fn = f'other_{chap:03d}_{blkno:03d}.npy'
    path = str(cfg.attribs / "pyaa_other").format(recd=recd)
    full = os.path.join(path, fn)
    data = np.load(full)
    # zcr, energy, energy_entropy, spectral_centroid, spectral_spread,
    # spectral_entropy, spectral_flux, spectral_rolloff
    all_vect = data

    return all_vect

def load_pyaa_mfcc(recd, chap, blkno):
    fn = f'mfcc_{chap:03d}_{blkno:03d}.npy'
    path = str(cfg.attribs / "pyaa_mfcc").format(recd=recd)
    full = os.path.join(path, fn)
    data = np.load(full)
    # pyaa_mfcc_{n} - n = 1..13
    all_vect = data

    return all_vect

def load_pyaa_chrom(recd, chap, blkno):
    fn = f'chrom_{chap:03d}_{blkno:03d}.npy'
    path = str(cfg.attribs / "pyaa_chrom").format(recd=recd)
    full = os.path.join(path, fn)
    data = np.load(full)
    # pyaa_chrm_{n} - n = 1..12, pyaa_chrm_std
    all_vect = data

    return all_vect

def load_pars_fmnts(recd, chap, blkno):
    fn = f'fmnts_{chap:03d}_{blkno:03d}.npy'
    path = str(cfg.attribs / "pars_fmnts").format(recd=recd)
    full = os.path.join(path, fn)
    data = np.load(full)
    # pars_fmnt_{n} - n = 1..7
    all_vect = data

    return all_vect


class LabelCategories():
    ltrs = ".abdfhiklmnoqrstuwyzġšǧʻʼḍḏḥḫṣṭṯẓ*ALNWY"
    # put "." as unused category '0', just because
    categ = {l:c for c,l in enumerate(ltrs)}   # have the letter, get the index
    label = {c:l for l,c in categ.items()}     # have the index, get the letter


def test_categories():
    print("\nrunning test_categories")
    lc = LabelCategories()
    test = "Yabdfhiklmnoqrst"
    categ = [lc.categ[x] for x in test]
    label = [lc.label[x] for x in categ]
    print(test)
    print(categ)
    print(label)

if __name__ == "__main__":
    print(f"this module {__file__} is for importing")

    test_categories()