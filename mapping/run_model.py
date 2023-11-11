                                                                                                                                                                                                                                                   # python3
'''
    When we run mapping, we need predictions. These predictions can be run
    just in time, therefor the class ml-tools/PredictionLoader runs the predictions.
    The PredictionLoader now saves the prediction data as numpy files. 
    When the PredictionLoader finds saved predictions, it returns these immediately
    and is much faster on a second run.
    
    This code just runs the PredictionLoader for a given model to predict and save the
    data for later usage by the mapping programs
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus9h"
    chapno = ''
    name = ''

    layout = """
    title    Run Model to precalculate and save the predictions
    text     recording  recording id, example: 'hus9h'
    text     chapno    Chapter or a list of chapter n:m
    text     name      model name
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config, AD
cfg = get_config()

import splib.toolbox as tbx
from machine_learning.ml_tools import PredLoader, get_training_specs

class G:
    pass

def main():
    G.runtoken = tbx.RunToken('run_model.run_token')

    recd, chap_seq, = dialog.recording, dialog.chapno

    model_name = dialog.name

    specs = get_training_specs(recd, model_name)
    itno_list = range(specs.iter)

    for chapno, blkno in tbx.chap_blk_iterator(recd, chap_seq, "*"):
        if G.runtoken.check_break():
            print("run token termination")
            break

        pred_loader = PredLoader(recd, chapno, blkno)

        for itno in itno_list:
            pred_loader.get_calculated_predictions(model_name, itno)

        print(f"block {chapno:03d}_{blkno:03d} done")

main()
