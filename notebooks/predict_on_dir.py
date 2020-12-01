import sys
import os
from ISR.predict import Predictor

from ISR.models import RDN

model = RDN(weights='noise-cancel')

def predict_on_dir(input_dir):
    output_dir = input_dir + '_x2'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    predictor = Predictor(input_dir=input_dir, output_dir=output_dir)
    predictor.get_predictions(model=model, weights_path='rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')

if __name__ == "__main__":
    input_dir = sys.argv[1]
    predict_on_dir(input_dir)