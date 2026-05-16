import os, glob, pickle, numpy, tqdm, math, cv2, argparse
import python_speech_features
from scipy.io import wavfile
from ASD_adaptive import AdaptiveASD
from utils.repo_paths import col_pycrop, col_pywork, exps_path

parser = argparse.ArgumentParser(description="Generate Columbia scores with FCAI routing")
parser.add_argument("--pycropPath", type=str, default=None)
parser.add_argument("--pyworkPath", type=str, default=None)
parser.add_argument("--pyworkFcai", type=str, default=None, help="Output dir for FCAI scores")
parser.add_argument("--baselineModel", type=str, default=None)
parser.add_argument("--transformerModel", type=str, default=None)
args = parser.parse_args()

PYCROP = args.pycropPath or col_pycrop()
PYWORK = args.pyworkPath or col_pywork()
PYWORK_FCAI = args.pyworkFcai or col_pywork("pywork_fcai")
BASELINE = args.baselineModel or os.path.join(
    exps_path("exp_baseline", "model"), "model_0022.model"
)
TRANSF = args.transformerModel or os.path.join(
    exps_path("ablation1_transformer", "model"), "model_0039.model"
)

tracks = pickle.load(open(os.path.join(PYWORK, "tracks.pckl"), "rb"))


def count_concurrent_faces(tracks):
    from collections import defaultdict

    frame_to_tracks = defaultdict(list)
    for tidx, track in enumerate(tracks):
        for frame in track["track"]["frame"].tolist():
            frame_to_tracks[frame].append(tidx)

    track_max_faces = {}
    for tidx, track in enumerate(tracks):
        max_concurrent = 1
        for frame in track["track"]["frame"].tolist():
            concurrent = len(frame_to_tracks[frame])
            max_concurrent = max(max_concurrent, concurrent)
        track_max_faces[tidx] = max_concurrent
    return track_max_faces


track_face_counts = count_concurrent_faces(tracks)
print(
    "Face count distribution:",
    {k: sum(1 for v in track_face_counts.values() if v == k) for k in [1, 2, 3, 4]},
)

model = AdaptiveASD(BASELINE, TRANSF)

files = sorted(glob.glob(os.path.join(PYCROP, "*.avi")))
allScores = []

for tidx, file in enumerate(tqdm.tqdm(files)):
    fileName = os.path.splitext(os.path.basename(file))[0]
    _, audio = wavfile.read(file.replace(".avi", ".wav"))
    audioFeature = python_speech_features.mfcc(
        audio, 16000, numcep=13, winlen=0.025, winstep=0.010
    )
    video = cv2.VideoCapture(file)
    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[56:168, 56:168]
            videoFeature.append(face)
        else:
            break
    video.release()
    videoFeature = numpy.array(videoFeature)

    num_faces = track_face_counts.get(tidx, 1)
    score = model.score_clip(audioFeature, videoFeature, num_faces)
    allScores.append(score)

os.makedirs(PYWORK_FCAI, exist_ok=True)
import shutil

shutil.copy(os.path.join(PYWORK, "tracks.pckl"), os.path.join(PYWORK_FCAI, "tracks.pckl"))

with open(os.path.join(PYWORK_FCAI, "scores.pckl"), "wb") as f:
    pickle.dump(allScores, f)
print(f"FCAI scores saved to {PYWORK_FCAI}/scores.pckl")
