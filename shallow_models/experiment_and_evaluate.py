import utils
import shallow_classifiers as sc
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--outcome', type=str, required=True)
parser.add_argument('--path', type=str, required=True)

args = parser.parse_args()

model_type = args.model_type
outcome = args.outcome
path = args.path

classifier = sc.ShallowClassifier(model_type=model_type, outcome=outcome)

classifier.experiment_and_set_hyperparameters()
classifier.train()
classifier.save_predictions()