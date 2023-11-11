# python3
"""
    There seems to be a mismatch of data columns between
    -   data, which we get for training from the database
    -   data, which we get for predictions from the numpy files

"""

from config import get_config
cfg = get_config()

# from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from splib.cute_dialog import start_dialog
from prepare_model import get_model
from ml_data import get_pred_data
from ml_data import get_training_data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
import numpy as np
import splib.toolbox as tbx
AD = tbx.AttrDict
import splib.attrib_tools as att
import pickle
import random
import time

class G:
    lc = att.LabelCategories()
    ml_result = None  # json writing oject
    lc = att.LabelCategories

class dialog:
    recording = "hus1h"
    attrs = 'bfpcm'
    chapno = 0
    blkno = 0
    frpos = 0
    topos = 0
    span = 0
    savename = ''

    layout = """
    title Predictions and plotting
    text     recording  recording id, example: 'hus1h'
    int      chapno    Chapter
    int      blkno     Block
    int      frpos   from position (ms)    
    int      topos   to position (ms)
        
    label specs for input data
    text     attrs    which attribs? (bcfmp)
    int      span     span (+/- ms range)
    text     savename  model name (for scaler)
    """


def main():
    if not start_dialog(__file__, dialog):
        return

    attr_sel = dialog.attrs
    recd = dialog.recording
    chapno = dialog.chapno
    blkno = dialog.blkno
    frpos = dialog.frpos
    topos = dialog.topos
    name = dialog.savename
    span = int(dialog.span)


    #fn = f'{name}.scale'
    #full = cfg.work / dialog.recording / 'saved_models' / fn
    #scaler = pickle.load(open(full, 'rb'))


    data_spec = AD(attr_sel=attr_sel, span=span)
    # get the attribute from the training database
    x_values, r_values, y_labels = \
        get_training_data(dialog.recording, data_spec)

    scaler = MinMaxScaler()
    scaler.fit(x_values)
    #x_values = scaler.transform(x_values)

    # get the attributes from the numpy vector data
    v_values = get_pred_data(dialog.recording, chapno, blkno, data_spec)
    #v_values = scaler.transform(v_values)

    for xv, rv in zip(x_values, r_values):
        ltr, chap_blk, pos = rv
        chap = int(chap_blk[0:3])
        blk = int(chap_blk[4:7])
        if not (chap==chapno and blk == blkno):
            continue
        print(chap, blk, pos, ltr)
        traindata = [round(n,4) for n in xv]
        print("training",traindata)
        v_ndx = int(pos/5)
        predictdata = [round(n,4) for n in v_values[v_ndx]]
        print("predictn",predictdata)
main()