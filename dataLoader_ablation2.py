import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        audio_path = os.path.join(dataPath, videoName, dataName + '.wav')
        if not os.path.isfile(audio_path):
            audio = numpy.zeros(16000, dtype=numpy.int16)
        else:
            _, audio = wavfile.read(audio_path)
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):
    candidates = list(set(audioSet.keys()) - {dataName})
    if len(candidates) == 0:
        return audio
    noiseName = random.choice(candidates)
    noiseAudio = audioSet[noiseName]
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio
    return audio.astype(numpy.int16)

def load_audio(data, dataPath, numFrames, audioAug, audioSet=None):
    dataName = data[0]
    fps = float(data[2])
    audio = audioSet[dataName]
    if audioAug == True:
        augType = random.randint(0, 1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
    audio = python_speech_features.mfcc(audio, 16000, numcep=13,
                winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage = maxAudio - audio.shape[0]
        audio = numpy.pad(audio, ((0, shortage), (0, 0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)), :]
    return audio

def load_visual(data, dataPath, numFrames, visualAug):
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg" % faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False)
    faces = []
    H = 112
    if visualAug == True:
        new = int(H * random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H / 2, H / 2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H, H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y + new, x:x + new], (H, H)))
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H, H)))
    faces = numpy.array(faces)
    return faces

def load_visual_other(dataName, visualPath, numFrames, allLines, visualAug):
    """
    Pick a second face track from the SAME clip as dataName.
    Same clip = same prefix before ':' in track ID.
    Returns zeros (H,W,T) if no other track exists in this clip.
    """
    H = 112
    clip_id = dataName.rsplit(':', 1)[0]   # e.g. _mAfwH6i90E_1080_1140

    # collect all track IDs from same clip (excluding target)
    candidates = []
    for line in allLines:
        other_name = line.split('\t')[0].strip()
        if other_name == dataName:
            continue
        if other_name.startswith(clip_id + ':'):
            candidates.append(other_name)

    if len(candidates) == 0:
        # fallback: pick random track from dataset (simulate multi-face)
        random_line = random.choice(allLines)
        other_name = random_line.split('\t')[0]
    else:
        other_name = random.choice(candidates)

    videoName  = other_name[:11]
    faceFolderPath = os.path.join(visualPath, videoName, other_name)
    faceFiles = glob.glob("%s/*.jpg" % faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda d: float(d.split('/')[-1][:-4]))

    if len(sortedFaceFiles) == 0:
        return numpy.zeros((numFrames, H, H), dtype=numpy.float32)

    faces = []
    augType = 'orig' if not visualAug else random.choice(['orig', 'flip'])
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        if face is None:
            faces.append(numpy.zeros((H, H), dtype=numpy.uint8))
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H, H))
        if augType == 'flip':
            face = cv2.flip(face, 1)
        faces.append(face)

    # pad if other track is shorter than numFrames
    while len(faces) < numFrames:
        faces.append(faces[-1] if faces else numpy.zeros((H, H), dtype=numpy.uint8))

    return numpy.array(faces[:numFrames])

def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch  = []
        self.allLines   = open(trialFileName).read().splitlines()  # kept for other-face lookup
        mixLst = self.allLines
        sortedMixLst = sorted(mixLst,
            key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)
        start = 0
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end

    def __getitem__(self, index):
        batchList  = self.miniBatch[index]
        numFrames  = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, visualFeatures2, labels = [], [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList)
        two_face = 0
        for line in batchList:
            data = line.split('\t')
            audioFeatures.append(load_audio(data, self.audioPath, numFrames,
                                            audioAug=True, audioSet=audioSet))
            visualFeatures.append(load_visual(data, self.visualPath, numFrames, visualAug=True))

            # --- second face from same clip ---
            other = load_visual_other(data[0], self.visualPath, numFrames,
                                      self.allLines, visualAug=True)
            visualFeatures2.append(other)
            if other.sum() > 0:   # non-zero = real second face found
                two_face += 1
            labels.append(load_label(data, numFrames))

        two_pct = 100.0 * two_face / len(batchList)
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures2)), \
               torch.LongTensor(numpy.array(labels)), \
               two_pct   # logged in ASD_ablation2.py

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.allLines   = open(trialFileName).read().splitlines()
        self.miniBatch  = self.allLines

    def __getitem__(self, index):
        line      = [self.miniBatch[index]]
        numFrames = int(line[0].split('\t')[1])
        audioSet  = generate_audio_set(self.audioPath, line)
        data      = line[0].split('\t')
        audioFeatures  = [load_audio(data, self.audioPath, numFrames,
                                     audioAug=False, audioSet=audioSet)]
        visualFeatures = [load_visual(data, self.visualPath, numFrames, visualAug=False)]
        other = load_visual_other(data[0], self.visualPath, numFrames,
                                  self.allLines, visualAug=False)
        visualFeatures2 = [other]
        labels = [load_label(data, numFrames)]
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures2)), \
               torch.LongTensor(numpy.array(labels)), \
               0.0   # two_pct not tracked during val

    def __len__(self):
        return len(self.miniBatch)
