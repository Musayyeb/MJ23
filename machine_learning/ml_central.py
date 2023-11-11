# python3
'''
    Machine learning central

    from here we start all(!) machine learning operations
    This module helps to document
    -   which data goes into training
    -   which ml-model is used
    -   which are the results (quality and time)
'''

import sys
from splib.cute_dialog import start_dialog

class dialog:
    recording = "hus1h hus9h"
    attrs = 'bcfmp'
    span = 0
    reduced = True

    layers = '8 16 32 16 8'
    iters = 0
    full_iters = 0
    epochs = 10
    batch = 32
    savename = ''

    layout = """
    title Machine learning central
    label  Select the automap training data, none means manumap only
    text     recording  recording list none / hus1h / hus9h / hus1h hus9h
    label specs for input data
    text     attrs    which attribs? (bcfmp)
    text     span     side datapoints (0/5/10) ms
    bool   reduced   reduced set of consonants
    label specs for ml-model training
    text    layers   list of hidden layers (sizes)
    int     iters    iterations
    int     full_iters iterations with full data (<= iterations)
    int     epochs   epochs
    int     batch    batch size
    text    savename name to save model
    """

if not start_dialog(__file__, dialog):
    sys.exit()

import numpy as np

from config import get_config
cfg = get_config()

# from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from prepare_model import get_model
from ml_data import get_training_data, data_r_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import splib.toolbox as tbx
AD = tbx.AttrDict
import splib.attrib_tools as att
import pickle
import json
import random
import time
import os, glob, shutil

class G:
    lc = att.LabelCategories()
    ml_result = None  # json writing oject
    model_path = None

def main():
    stt = tbx.Statistics()
    ml_result_fn = cfg.work / 'ml_results.txt'
    G.ml_result = tbx.UJson(ml_result_fn)

    mlrec = AD(time=tbx.now())  # keep all information about the training

    mlrec.attr_sel = attr_sel = dialog.attrs
    mlrec.span = span = int(dialog.span)
    mlrec.batch = batchsz = dialog.batch
    mlrec.epochs = epochs = dialog.epochs
    mlrec.iters = iterations = dialog.iters
    mlrec.full_iters = full_data_iters = dialog.full_iters # how many iterations with full (no test) data
    mlrec.savename = savename = dialog.savename.strip()

    G.model_path = cfg.work / 'saved_models'

    # data selection parameters:
    data_spec = AD(recd= dialog.recording, attr_sel=attr_sel, span=span, reduced=dialog.reduced)

    random.seed()

    recd = dialog.recording  # there is no training data in hus1h

    x_values, r_values, y_labels = get_training_data(data_spec)

    scaler = MinMaxScaler()
    scaler.fit(x_values)
    x_values = scaler.transform(x_values)

    print('x_values',x_values.shape)
    # as our labels are categories, we must encode them as "one-hot"
    # that means, each category is a column with the value 1 and
    # value 0 for all other columns

    y_1hot = to_categorical(y_labels)
    # print("y_labels", y_labels[:10])
    # print("y_1hot", y_1hot[:10])


    categories = len(y_1hot[0])  # this must match the size of the output layer
    print("to_categorical returns classes: ", categories)

    # model parameters
    model_spec = tbx.AD(nfeatures= len(x_values[0]), output=categories,
                        layers= dialog.layers)
    # mlrec.model_spec = model_spec
    mlrec.features = len(x_values[0])
    mlrec.layers = dialog.layers


    #collect results from train/evaluate iterations
    stt.reset()  # clear all keys

    if savename:
        # prepare folder to store models, scalers and spec data
        prepare_model_folder(savename)


    for iter in range(iterations):  # iterate over training and evaluation
        print(f"Iteration: {iter}")
        model = get_model(model_spec)

        random.seed()

        tsize = 0.20 if iter >= full_data_iters else 0.0  # The first iteration(s) gets all the data

        x_train, x_test, y_train, y_test, r_test = data_r_split(x_values, y_1hot,
                                                   r_values, test_size=tsize)
        #y_train = np.asarray(y_train)
        #y_test = np.asarray(y_test)
        mlrec.x_count = len(x_train)

        print(f"data sizes: xy_train {x_train.shape}, {y_train.shape}"
                      f"  -  xy_test {x_test.shape}, {y_test.shape}")

        # this is the training
        history = model.fit(x_train, y_train, batch_size=batchsz, epochs=epochs, verbose=0) #,
        #          validation_data=[x_test, y_test])

        hkeys = history.history.keys()
        print("hkeys",hkeys)
        for k in hkeys:
            vlist = [f"{round(n,3)}" for n in history.history[k]]
            print(f"{k:10} : {vlist}")

        if savename:
            # save the model
            full = G.model_path / savename / f"mod_{iter:02d}.model"
            model.save(full)
            print("saved model file:", full)

        if iter >= full_data_iters:
            score = model.evaluate(x_test, y_test, verbose=0)
            print("model score:", score[0], score[1])

            preds = model.predict(x_test)

            stt.collect("iters", 1)
            for row, test_1hot, ref in zip(preds, y_test, r_test):
                # print(hi_probabilities(row, test_1hot, ref))

                right, prob = calc_probabilities(row, test_1hot)
                stt.collect('right_ltrs', right)  # number of correct mappings
                stt.collect('ltr_prob', prob)    # actual probability

            statis = stt.evaluate('iters')
            statis = stt.evaluate("right_ltrs")
            right = statis.avg
            #mlrec.right_ltrs = f"ct:{statis.sum} avg:{statis.avg:5.3f} std:{statis.stdd:5.3f}"
            mlrec.right_ltrs = round(statis.avg, 3)
            statis = stt.evaluate("ltr_prob")
            prob = statis.avg
            #mlrec.ltr_prob = f"ct:{statis.count} avg:{statis.avg:5.3f}  std:{statis.stdd:5.3f}"
            mlrec.ltr_prob = round(statis.avg,3)
            print(f"pred result: right_ltrs={right}, ltr_prob={prob}")

        mlrec.runtime = stt.runtime()
        G.ml_result.dump(mlrec)

    # end of the training loop

    if savename:
        # save the scaler
        full = G.model_path / savename / f'{savename}.scale'
        pickle.dump(scaler, open(full, 'wb'))
        print(f"saved scaler {full}")

        # save some of the specifications of the training
        trainspec = AD(attr_sel=mlrec.attr_sel, span=mlrec.span, iter=mlrec.iters,
                       reduced=dialog.reduced, layers=dialog.layers, epochs=dialog.epochs,
                       recording=dialog.recording)
        print(trainspec)
        full = G.model_path / savename / 'model_specs.json'
        json.dump(trainspec, open(full, mode='w'))



def prepare_model_folder(savename):
    print("delete previous files")
    # if the model shall be saved, first delete existing model files

    full = G.model_path / savename
    if os.path.exists(full):
        shutil.rmtree(full)
    os.mkdir(full)


def __delete_saved_model(savename):
    print("delete previous files")
    # if the model shall be saved, first delete existing model files

    full = G.model_path / f'{savename}.scale'
    if os.path.exists(full):
        os.remove(full)

    full = G.model_path / f'{savename}_specs.json'
    if os.path.exists(full):
        os.remove(full)

    generic = str(G.model_path / f"{savename}_*.model")
    for filename in glob.glob(generic):
        shutil.rmtree(filename)


def calc_probabilities(row, test_1hot):
    categ  = np.argmax(test_1hot)  # get the letter from the one-hot format
    hi_prob = sorted([(n,x) for x, n in enumerate(row) ], reverse=True)
    for ndx, (n, cat) in enumerate(hi_prob):
        if cat == categ:
            if ndx == 0:
                return 1, n
            else:
                return 0, n


def hi_probabilities(row, test_1hot, ref):
    letter, block, mspos = ref

    assert letter == G.lc.label[np.argmax(test_1hot)]  # get the letter from the one-hot format

    hi_probs = sorted([f"{int(99.9*n):2d} {G.lc.label[ndx]}" for ndx, n in enumerate(row)], reverse=True)
    good = '^' if hi_probs[0][-1] == letter else ' '
    this_ltr = next (s for s in hi_probs if s[-1] == letter)
    other = [s for s in hi_probs if s[-1] != letter and s[:2] > " 4"]
    res = f" {letter} {block} {mspos:6d}   {this_ltr} {good}   {'   '.join(other)}"
    return res


main()