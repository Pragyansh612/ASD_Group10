import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):
    noiseName = random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
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
    if audioAug:
        augType = random.randint(0, 2)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        elif augType == 2:
            # Environmental noise
            snr_db = random.uniform(5, 20)
            signal_power = numpy.mean(audio.astype(float)**2) + 1e-8
            noise_power = signal_power / (10**(snr_db/10))
            noise = numpy.random.normal(0, numpy.sqrt(noise_power), len(audio))
            audio = (audio.astype(float) + noise).clip(-32768, 32767).astype(numpy.int16)
    audio = python_speech_features.mfcc(audio, 16000, numcep=13,
                winlen=0.025*25/fps, winstep=0.010*25/fps)
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
    if visualAug:
        new = int(H * random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H-new), numpy.random.randint(0, H-new)
        M = cv2.getRotationMatrix2D((H/2, H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
        blur_aug  = random.random() < 0.3
        mask_aug  = random.random() < 0.3
        blur_sigma = random.uniform(0.5, 2.0)
        mask_mean = None
    else:
        augType = 'orig'
        blur_aug = mask_aug = False

    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H, H))
        if augType == 'flip':
            face = cv2.flip(face, 1)
        elif augType == 'crop':
            face = cv2.resize(face[y:y+new, x:x+new], (H, H))
        elif augType == 'rotate':
            face = cv2.warpAffine(face, M, (H, H))
        if blur_aug:
            ksize = int(blur_sigma * 3) * 2 + 1
            face = cv2.GaussianBlur(face, (ksize, ksize), blur_sigma)
        if mask_aug:
            if mask_mean is None:
                mask_mean = int(face.mean())
            face[56:112, :] = mask_mean
        faces.append(face)
    faces = numpy.array(faces)
    return faces

def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

def is_all_negative(data):
    """Check if a track is all non-speaking (hard negative candidate)"""
    labels = data[3].replace('[', '').replace(']', '')
    return all(int(l) == 0 for l in labels.split(','))

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch  = []
        mixLst = open(trialFileName).read().splitlines()

        # Build a lookup: videoClipID -> list of all-negative tracks
        self.hardNegPool = {}
        for line in mixLst:
            data = line.split('\t')
            clipID = data[0][:11]  # video name
            if is_all_negative(data):
                if clipID not in self.hardNegPool:
                    self.hardNegPool[clipID] = []
                self.hardNegPool[clipID].append(line)

        sortedMixLst = sorted(mixLst,
            key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])),
            reverse=True)
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
        audioFeatures, visualFeatures, labels = [], [], []

        # Add hard negatives: for each batch, sample one hard negative per clip
        hardNegLines = []
        for line in batchList:
            data   = line.split('\t')
            clipID = data[0][:11]
            if clipID in self.hardNegPool and len(self.hardNegPool[clipID]) > 0:
                hn = random.choice(self.hardNegPool[clipID])
                if hn not in batchList:  # avoid duplicates
                    hardNegLines.append(hn)

        # Deduplicate hard negatives
        hardNegLines = list(set(hardNegLines))[:max(1, len(batchList)//4)]
        fullBatch = batchList + hardNegLines

        audioSet = generate_audio_set(self.audioPath, fullBatch)

        for line in fullBatch:
            data = line.split('\t')
            af = load_audio(data, self.audioPath, numFrames, audioAug=True, audioSet=audioSet)
            vf = load_visual(data, self.visualPath, numFrames, visualAug=True)
            lb = load_label(data, numFrames)
            maxAudio = int(numFrames * 4)
            if af.shape[0] < maxAudio:
                af = numpy.pad(af, ((0, maxAudio-af.shape[0]), (0,0)), 'wrap')
            af = af[:maxAudio]
            if vf.shape[0] < numFrames:
                vf = numpy.pad(vf, ((0, numFrames-vf.shape[0]), (0,0), (0,0)), 'wrap')
            vf = vf[:numFrames]
            if lb.shape[0] < numFrames:
                lb = numpy.pad(lb, (0, numFrames-lb.shape[0]), 'wrap')
            lb = lb[:numFrames]
            audioFeatures.append(af)
            visualFeatures.append(vf)
            labels.append(lb)


        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch  = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        line = self.miniBatch[index]
        data = line.split('\t')
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, [line])
        audioFeatures.append(load_audio(data, self.audioPath, int(data[1]),
                                        audioAug=False, audioSet=audioSet))
        visualFeatures.append(load_visual(data, self.visualPath, int(data[1]),
                                          visualAug=False))
        labels.append(load_label(data, int(data[1])))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
