from config import get_config
cfg = get_config()

import numpy as np
import os


def main():
    for recd in 'hus1h hus9h'.split():
        print("recording", recd)
        for folder in "freq_ampl pars_fmnts pyaa_chrom pyaa_mfcc pyaa_other".split():
            print('   folder:', folder)
            replace = 0
            totcount = 0
            path = cfg.data / recd / "attribs" / folder
            fl = os.listdir(path)
            for fn in fl:
                full = path / fn
                np_mat = np.load(full)
                #cond = np.NaN in np_mat
                count = np.sum(np.isnan(np_mat))
                if count:
                    totcount += count
                    replace += 1
                    np.nan_to_num(np_mat, copy=False, nan=0.0)
                    np.save(full, np_mat)
            print("nan counter:", replace, totcount)

main()