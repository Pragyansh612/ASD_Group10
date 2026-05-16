import sys, os, glob, pickle, torch, tqdm, math, cv2, argparse
import numpy, python_speech_features
from scipy.io import wavfile
from ASD_camib import ASD
from utils.repo_paths import col_pycrop, col_pywork

parser = argparse.ArgumentParser()
parser.add_argument('--pretrainModel', type=str, required=True)
parser.add_argument('--pycropPath', type=str, default=None)
parser.add_argument('--pyworkPath', type=str, default=None)
args = parser.parse_args()
args.pycropPath = args.pycropPath or col_pycrop()
args.pyworkPath = args.pyworkPath or col_pywork("pywork_camib")

s = ASD()
s.loadParameters(args.pretrainModel)
s.eval()
print("Model loaded: %s" % args.pretrainModel)

files = sorted(glob.glob("%s/*.avi" % args.pycropPath))
print("Found %d face clips" % len(files))

durationSet = {1,1,1,2,2,2,3,3,4,5,6}
allScores = []
for file in tqdm.tqdm(files):
    fileName = os.path.splitext(file.split('/')[-1])[0]
    _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
    video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret == True:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            videoFeature.append(face)
        else:
            break
    video.release()
    videoFeature = numpy.array(videoFeature)
    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
    audioFeature = audioFeature[:int(round(length * 100)),:]
    videoFeature = videoFeature[:int(round(length * 25)),:,:]
    allScore = []
    for duration in durationSet:
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                inputA = torch.FloatTensor(audioFeature[i*duration*100:(i+1)*duration*100,:]).unsqueeze(0).cuda()
                inputV = torch.FloatTensor(videoFeature[i*duration*25:(i+1)*duration*25,:,:]).unsqueeze(0).cuda()
                embedA = s.model.forward_audio_frontend(inputA)
                embedV = s.model.forward_visual_frontend(inputV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                score = s.lossAV.forward(out, labels=None)
                scores.extend(score)
        allScore.append(scores)
    allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
    allScores.append(allScore)

with open(os.path.join(args.pyworkPath, 'scores.pckl'), 'wb') as f:
    pickle.dump(allScores, f)
print("Scores saved to %s/scores.pckl" % args.pyworkPath)
