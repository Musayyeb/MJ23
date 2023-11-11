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
    recording = "hus1h"

    layout = """
    title Machine learning central
    text     recording  recording id, example: 'hus1h'
    """

if not start_dialog(__file__, dialog):
    sys.exit()


from config import get_config
cfg = get_config()

# from sklearn.model_selection import train_test_split
import splib.attrib_tools as att
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from prepare_model import get_model
from ml_data import get_training_data, data_r_split
import splib.toolbox as tbx
AD = tbx.AttrDict
from itertools import product
import numpy as np
import pickle
import random
import time

class G:
    lc = att.LabelCategories()
    ml_result = None  # json writing oject

    """
class dialog:
    recording = "hus1h"
    attr = 'bcfmp'
    span = 0

    layers = '40'        # '8 16 32 16 8'
    iters = 1
    epochs = 10
    batch = 32
    savename = ''

    layout = 
    title Machine learning central
    text     recording  recording id, example: 'hus1h'
    """


def main():
    ml_result_fn = cfg.work / 'ml_results.txt'
    G.ml_result = tbx.UJson(ml_result_fn)
    mlrec = AD(time=tbx.now())

    # i_attr = 'bp bf bfp bcf bcfp bcmp bcfmp'.split() # more is better
    i_attr = ['bcfmp',]
    i_span = [10,]
    i_batch = [4, 8, 16, 24, 32,]  # smaller batch size ==> better results, much slower
    i_epochs = [20,]
    i_iters = [15,]
    # i_layers = ['8 16 8', '8 16 32 16 8', '8 16 32 64 32 16 8'] # the smallest is the best and fastest
    # i_layers = ['4', '8', '4 4', '8 8', '4 4 4', '4 8 4', '4 4 4 4','4 8 8 4']
    i_layers = ['4', ]

    for attr, span, batch, epochs, iters, layers in product(
                i_attr, i_span, i_batch, i_epochs, i_iters, i_layers):
        mlrec.attr_sel = attr
        mlrec.span = span
        mlrec.batch = batch
        mlrec.epochs = epochs
        mlrec.iters = iters
        mlrec.layers = layers

        print("product:", mlrec)

        random.seed()

        train_and_predict(mlrec)

    # end of test_loop




def train_and_predict(mlrec):
    # collect results from train/evaluate iterations
    stt = tbx.Statistics()

    # data selection parameters:
    data_spec = AD(attr_sel=mlrec.attr_sel, span=mlrec.span)

    # prepare data
    x_values, r_values, y_labels = get_training_data(dialog.recording, data_spec)

    scaler = MinMaxScaler()
    scaler.fit(x_values)
    x_values = scaler.transform(x_values)

    print('x_values',x_values.shape)

    mlrec.x_count = len(x_values)
    mlrec.y_count = len(y_labels)
    # as our labels are categories, we must encode them as "one-hot"
    # that means, each category is a column with the value 1 and
    # value 0 for all other columns
    y_1hot = to_categorical(y_labels)

    mlrec.categories = len(y_1hot[0])  # this must match the size of the output layer
    mlrec.features = len(x_values[0])

    for testloop in range(mlrec.iters):  # iterate over training and evaluation
        print(f"Iteration: {testloop}")
        model_spec = tbx.AD(nfeatures=mlrec.features, output=mlrec.categories,
                            layers=mlrec.layers)

        model = get_model(model_spec)

        x_train, x_test, y_train, y_test, r_test = \
            data_r_split(x_values, y_1hot, r_values, test_size=0.20)

        print(f"data sizes: train x={x_train.shape}, y={y_train.shape}"
              f"  -  test x={x_test.shape}, y={y_test.shape}")

        # this is the training
        print(f"Training with {mlrec.epochs} epochs, specs: {model_spec}")

        model.fit(x_train, y_train, batch_size=mlrec.batch, epochs=mlrec.epochs, verbose=0,
                  validation_data=[x_test, y_test])

        # this is the prediction
        preds = model.predict(x_test)

        for row, test_1hot, ref in zip(preds, y_test, r_test):
            # print(hi_probabilities(row, test_1hot, ref))

            right, prob = calc_probabilities(row, test_1hot)
            stt.collect('right_ltrs', right)  # number of correct mappings
            stt.collect('ltr_prob', prob)    # actual probability

        statis = stt.evaluate("right_ltrs")
        #mlrec.right_ltrs = f"ct:{statis.sum} avg:{statis.avg:5.3f} std:{statis.stdd:5.3f}"
        mlrec.right_ltrs = round(statis.avg, 3)

        statis = stt.evaluate("ltr_prob")
        # mlrec.ltr_prob = f"ct:{statis.count} avg:{statis.avg:5.3f}  std:{statis.stdd:5.3f}"
        mlrec.ltr_prob = round(statis.avg,3)

    mlrec.runtime = stt.runtime()
    G.ml_result.dump(mlrec)


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