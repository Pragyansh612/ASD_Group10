import os, glob, pickle, numpy, tqdm, argparse
from sklearn.metrics import f1_score, accuracy_score

parser = argparse.ArgumentParser()
from utils.repo_paths import col_data_root

parser.add_argument('--colSavePath', type=str, default=None)
parser.add_argument('--scoresPath', type=str, default=None, help='Override scores.pckl path')
args = parser.parse_args()
args.colSavePath = args.colSavePath or col_data_root()

pyworkPath   = os.path.join(args.colSavePath, 'col', 'pywork')
pyframesPath = os.path.join(args.colSavePath, 'col', 'pyframes')
scoresPath   = args.scoresPath or os.path.join(pyworkPath, 'scores.pckl')

vidTracks = pickle.load(open(os.path.join(pyworkPath, 'tracks.pckl'), 'rb'))
scores    = pickle.load(open(scoresPath, 'rb'))
print("Loaded %d tracks, %d scores" % (len(vidTracks), len(scores)))

# Load GT labels
txtPath = os.path.join(args.colSavePath, 'col_labels', 'fusion', '*.txt')
predictionSet = {}
for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
    predictionSet[name] = [[], []]

dictGT = {}
for file in glob.glob(txtPath):
    lines = open(file).read().splitlines()
    idName = file.split('/')[-1][:-4]
    for line in lines:
        data = line.split('\t')
        frame = int(int(data[0]) / 29.97 * 25)
        x1, y1 = int(data[1]), int(data[2])
        x2, y2 = x1 + int(data[3]), y1 + int(data[3])
        gt = int(data[4])
        if frame in dictGT:
            dictGT[frame].append([x1, y1, x2, y2, gt, idName])
        else:
            dictGT[frame] = [[x1, y1, x2, y2, gt, idName]]

# Build face timeline
flist = sorted(glob.glob(os.path.join(pyframesPath, '*.jpg')))
faces = [[] for _ in range(len(flist))]
for tidx, track in enumerate(vidTracks):
    score = scores[tidx]
    for fidx, frame in enumerate(track['track']['frame'].tolist()):
        s = numpy.mean(score[max(fidx-2, 0): min(fidx+3, len(score)-1)])
        faces[frame].append({
            'score': float(s),
            's': track['proc_track']['s'][fidx],
            'x': track['proc_track']['x'][fidx],
            'y': track['proc_track']['y'][fidx]
        })

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# IOU matching
for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
    if fidx not in dictGT:
        continue
    for gtThisFrame in dictGT[fidx]:
        faceGT = gtThisFrame[0:4]
        labelGT = gtThisFrame[4]
        idGT = gtThisFrame[5]
        ious = []
        for face in faces[fidx]:
            faceLocation = [
                int((face["x"]-face["s"])*4), int((face["y"]-face["s"])*4),
                int((face["x"]+face["s"])*4), int((face["y"]+face["s"])*4)
            ]
            iou = bb_iou(faceLocation, faceGT)
            if iou > 0.5:
                ious.append([iou, round(face['score'], 2)])
        labelPredict = ious[-1][1] if len(ious) > 0 else 0
        predictionSet[idGT][0].append(labelPredict)
        predictionSet[idGT][1].append(labelGT)

# Evaluate
print("\n=== COLUMBIA IOU F1 RESULTS ===")
F1s = 0
names = sorted(['long', 'bell', 'boll', 'lieb', 'sick'])
for name in names:
    s = numpy.int64(numpy.array(predictionSet[name][0]) > 0)
    l = numpy.array(predictionSet[name][1])
    if len(l) == 0:
        print("%s: no predictions" % name)
        continue
    F1  = f1_score(l, s, zero_division=0)
    ACC = accuracy_score(l, s)
    F1s += F1
    print("%s: ACC=%.2f%%  F1=%.2f%%" % (name, 100*ACC, 100*F1))
print("Average F1: %.2f%%" % (100 * F1s / 5))
