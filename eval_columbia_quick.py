import sys, os, glob, pickle, torch, tqdm, argparse, python_speech_features
import numpy as np
from scipy.io import wavfile
from scipy import signal
from ASD import ASD

parser = argparse.ArgumentParser()
parser.add_argument('--pretrainModel', type=str, required=True)
parser.add_argument('--colSavePath', type=str, default="/usershome/cs671_user6/asd_project/ColData")
args = parser.parse_args()

args.pyworkPath = os.path.join(args.colSavePath, 'col', 'pywork')
args.pycropPath = os.path.join(args.colSavePath, 'col', 'pycrop')

# Load existing tracks
with open(os.path.join(args.pyworkPath, 'tracks.pckl'), 'rb') as f:
    vidTracks = pickle.load(f)
print("Loaded %d tracks" % len(vidTracks))

# Load model
s = ASD()
s.loadParameters(args.pretrainModel)
s.eval()
print("Model loaded")

# Load col_labels for evaluation
col_labels_path = os.path.join(args.colSavePath, 'col_labels')

# Run evaluate_col_ASD directly from scores.pckl if exists, else error
scores_path = os.path.join(args.pyworkPath, 'scores.pckl')
if os.path.exists(scores_path):
    with open(scores_path, 'rb') as f:
        scores = pickle.load(f)
    print("Loaded existing scores")
else:
    print("scores.pckl not found, need to run full eval")
    sys.exit(1)

# Evaluate per speaker
from sklearn.metrics import f1_score
names = ['bell','boll','lieb','long','sick']
for name in names:
    label_path = os.path.join(col_labels_path, name)
    if not os.path.exists(label_path):
        continue
    print(f"Speaker: {name}")

