# python3
'''
    Machine predictions - plot the result

    plot the predictions along with some audio attributes
'''

from splib.cute_dialog import start_dialog
import sys

class dialog:
    recording = "hus9h"
    chapno = 0
    attr = 'bfpcm'
    span = 0

    iters = 0
    avgwin = 5
    savename = ''

    layout = """
    title Save predictions as numpy data files
    text     recording  recording id, example: 'hus1h'
    text     chapno    Chapter (range n:m)

    label specs for input data
    text     attrs    which attribs? (bcfmp)
    text     span     side datapoints (0/5/10) ms
    int      iters    model iterations
    text     savename saved model name
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config
cfg = get_config()

# from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from prepare_model import get_model
from ml_data import get_pred_data
from matplotlib import pyplot as plt
import numpy as np
import splib.toolbox as tbx
from splib.toolbox import get_block_counters
import splib.sound_tools as st
from itertools import product
AD = tbx.AttrDict
import splib.attrib_tools as att
import splib.text_tools as tt
import pickle
import random
import time

class G:
    lc = att.LabelCategories()
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None

    scaler = None
    models = []
    data_spec = None

def get_range(text):
    toks = text.split(':')
    if len(toks) == 1:
        return (int(text), )
    return range(int(toks[0]), int(toks[1]))

def main():
    stt = tbx.Statistics()
    recd, chaptxt = dialog.recording, dialog.chapno
    block_counters = tbx.get_block_counters(recd)

    # prediction data selection parameters:
    span = int(dialog.span)
    attr_sel = dialog.attrs

    G.data_spec = AD(attr_sel=attr_sel, span=span)
    # load the scaler
    name = dialog.savename
    fn = f'{name}.scale'
    full = cfg.work / dialog.recording / 'saved_models' / fn
    G.scaler = pickle.load(open(full, 'rb'))
    print("loaded scaler", full)

    for iter in range(dialog.iters):

        fn = f'{name}_{iter:02d}.model'
        full = cfg.work / dialog.recording / 'saved_models' / fn

        G.models.append(keras.models.load_model(full))
        print("loaded model", full)

    for chapno in get_range(chaptxt):
        blocks = block_counters[chapno]
        chapno = int(chapno)

        print(f"\n\nProcess chapter {chapno} with {blocks} blocks")
        for blkno in range(1, blocks+1):

            load_block(recd, chapno, blkno)

def load_block(recd, chapno, blkno):

    x_values = get_pred_data(dialog.recording, chapno, blkno, G.data_spec)
    x_values = G.scaler.transform(x_values)
    print(f"x_values:, {x_values.shape} for block {blkno}")

    allpreds = []  # collect predictions from all model instances

    for iter in range(dialog.iters):

        # this is the prediction
        preds = G.models[iter].predict(x_values, verbose=0)

        allpreds.append(preds)

        if iter == 0:
            print("preds", preds.shape)
            # print("preds_v:", preds[:10])


    avg_pred = np.mean(allpreds, axis=0)  # average predictions

    print(f"average probabilities: {avg_pred.shape}")

    full = cfg.data / dialog.recording / 'probs' / f"{chapno:03d}_{blkno:03d}.npy"
    np.save(full, avg_pred)

main()
