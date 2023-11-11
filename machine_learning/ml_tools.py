# python3
"""
    functions and classes,which support the machine learning and predictions
"""
from config import get_config
cfg = get_config()
from machine_learning.ml_data import get_pred_data
import splib.toolbox as tbx
AD = tbx.AttrDict
import keras
import numpy as np
import json, pickle
from functools import cache
import os


class PredLoader:
    # get prediction data fresh from a saved model
    # or get the data from saved numpy datasets
    def __init__(self, recd, chapno, blkno):
        self.recd = recd
        self.chapno = chapno
        self.blkno = blkno
        self.model_path = cfg.work / 'saved_models'

        # these values depend on the model
        self.model_name = ''
        self.specs = None
        self.x_values = None
        self.scaler = None

    def get_calculated_predictions(self, model_name, itno):
        # load scaler and (a specific) model to calculate predictions
        self.refresh_data(model_name)
        avail = self.specs.iter
        if itno > avail:
            raise RuntimeError(
                f"requested model iteration {itno} is not available {avail}")

        # eventually use cached predictions
        fn = f'predictions_{self.chapno:03d}_{self.blkno:03d}_{itno:02d}.npy'
        path = str(cfg.pred_cache).format(recd=self.recd)
        path = os.path.join(path, model_name)
        os.makedirs(path, exist_ok=True)

        full = os.path.join(path, fn)

        if os.path.exists(full):
            # print(f"reuse cached preds: {full}")
            preds = np.load(full)
        else:
            # print(f"calcuclate preds: {full}")

            model = self.get_model(model_name, itno)
            preds = model.predict(self.x_values)

            #print(f"predictions from the model:", preds.shape)
            preds = np.swapaxes(preds, 0, 1)
            #print(f"predictions as we need them:", preds.shape)
            np.save(full, preds)

        return preds


    def refresh_data(self, model_name):
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.specs = self.get_training_specs(model_name)
        self.scaler = self.get_scaler(model_name)
        self.x_values = self.get_training_data(model_name, self.specs, self.scaler)

    def get_scaler(self, model_name):
        full = self.model_path / model_name / f'{model_name}.scale'
        return get_scaler(full)

    def get_model(self, model_name, itno):
        fn = f'mod_{itno:02d}.model'
        full = self.model_path / model_name /  fn
        return get_model(full)


    def get_training_data(self, model_name, specs, scaler):
        x_values = get_pred_data(self.recd, self.chapno, self.blkno, specs)
        # print(f"ml_tools - x values shape:{x_values.shape}")
        x_values = scaler.transform(x_values)
        return x_values

    def get_training_specs(self, model_name):
        # load the specifications of the training
        return get_training_specs(self.recd, model_name)
        '''
        model_path = cfg.work / self.recd / 'saved_models'
        full = model_path / f"{model_name}_specs.json"
        trainspec = json.load(open(full, mode='r'))
        return AD(**trainspec) '''


@cache
def get_scaler(full):
    scaler = pickle.load(open(full, 'rb'))
    return scaler

@cache
def get_model(full):
    #print(f"load model: {full}")
    model = keras.models.load_model(full)
    return model


def get_training_specs(recd, model_name):
    # load the specifications of the training
    model_path = cfg.work / 'saved_models'
    full = model_path / model_name / 'model_specs.json'
    trainspec = json.load(open(full, mode='r'))
    return AD(**trainspec)



def test():
    pred_loader = PredLoader('hus9h', 100, 3)

    pred = pred_loader.get_calculated_predictions('sibel', 2)
    print(f"pred data: {pred.shape}")
    pred = pred_loader.get_calculated_predictions('elise', 0)
    print(f"pred data: {pred.shape}")
    pred = pred_loader.get_calculated_predictions('sibel', 1)
    print(f"pred data: {pred.shape}")
    pred = pred_loader.get_calculated_predictions('sibel', 9)
    print(f"pred data: {pred.shape}")



if __name__ == "__main__":
    test()