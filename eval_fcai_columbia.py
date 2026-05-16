import os, glob, pickle, numpy, tqdm, math, cv2
import python_speech_features
from scipy.io import wavfile
from ASD_adaptive import AdaptiveASD

PYCROP   = '/usershome/cs671_user6/asd_project/ColData/col/pycrop'
PYWORK   = '/usershome/cs671_user6/asd_project/ColData/col/pywork'
BASELINE = '/usershome/cs671_user6/asd_project/LR-ASD/exps/exp_baseline/model/model_0022.model'
TRANSF   = '/usershome/cs671_user6/asd_project/LR-ASD/exps/ablation1_transformer/model/model_0039.model'

# Load face count per track from tracks.pckl
tracks = pickle.load(open(f'{PYWORK}/tracks.pckl','rb'))

# Count how many other tracks overlap with each track
def count_concurrent_faces(tracks):
    from collections import defaultdict
    frame_to_tracks = defaultdict(list)
    for tidx, track in enumerate(tracks):
        for frame in track['track']['frame'].tolist():
            frame_to_tracks[frame].append(tidx)
    
    track_max_faces = {}
    for tidx, track in enumerate(tracks):
        max_concurrent = 1
        for frame in track['track']['frame'].tolist():
            concurrent = len(frame_to_tracks[frame])
            max_concurrent = max(max_concurrent, concurrent)
        track_max_faces[tidx] = max_concurrent
    return track_max_faces

track_face_counts = count_concurrent_faces(tracks)
print("Face count distribution:", 
      {k: sum(1 for v in track_face_counts.values() if v==k) for k in [1,2,3,4]})

model = AdaptiveASD(BASELINE, TRANSF)

files = sorted(glob.glob(f'{PYCROP}/*.avi'))
allScores = []

for tidx, file in enumerate(tqdm.tqdm(files)):
    fileName = os.path.splitext(file.split('/')[-1])[0]
    _, audio = wavfile.read(file.replace('.avi','.wav'))
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13,
                                                winlen=0.025, winstep=0.010)
    video = cv2.VideoCapture(file)
    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[56:168, 56:168]
            videoFeature.append(face)
        else: break
    video.release()
    videoFeature = numpy.array(videoFeature)
    
    num_faces = track_face_counts.get(tidx, 1)
    score = model.score_clip(audioFeature, videoFeature, num_faces)
    allScores.append(score)

os.makedirs('/usershome/cs671_user6/asd_project/ColData/col/pywork_fcai', exist_ok=True)
import shutil
shutil.copy(f'{PYWORK}/tracks.pckl',
            '/usershome/cs671_user6/asd_project/ColData/col/pywork_fcai/tracks.pckl')

with open('/usershome/cs671_user6/asd_project/ColData/col/pywork_fcai/scores.pckl','wb') as f:
    pickle.dump(allScores, f)
print("FCAI scores saved!")
