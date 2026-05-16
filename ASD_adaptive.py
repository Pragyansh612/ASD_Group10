import torch, numpy, os, glob, pickle, math, cv2, tqdm, argparse
import python_speech_features
from scipy.io import wavfile
from ASD import ASD as ASD_baseline
from ASD_transformer import ASD as ASD_transformer

class AdaptiveASD:
    """
    Face-Count Adaptive Inference (FCAI):
    - 1 face visible  → LR-ASD baseline (94.11% AVA, high accuracy)
    - 2+ faces visible → Transformer (71.25% Columbia, low domain drop)
    """
    def __init__(self, baseline_path, transformer_path):
        print("Loading baseline model...")
        self.model_single = ASD_baseline()
        self.model_single.loadParameters(baseline_path)
        self.model_single.eval()

        print("Loading transformer model...")
        self.model_multi = ASD_transformer()
        self.model_multi.loadParameters(transformer_path)
        self.model_multi.eval()

        print("FCAI ready")

    def score_clip(self, audioFeature, videoFeature, num_faces_in_track):
        """Score a single clip using appropriate model based on face count."""
        durationSet = {1,1,1,2,2,2,3,3,4,5,6}
        length = min((audioFeature.shape[0] - audioFeature.shape[0]%4)/100,
                     videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length*100)),:]
        videoFeature = videoFeature[:int(round(length*25)),:,:]

        model = self.model_multi if num_faces_in_track >= 2 else self.model_single

        allScore = []
        for duration in durationSet:
            batchSize = int(math.ceil(length/duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(
                        audioFeature[i*duration*100:(i+1)*duration*100,:]
                    ).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(
                        videoFeature[i*duration*25:(i+1)*duration*25,:,:]
                    ).unsqueeze(0).cuda()
                    embedA = model.model.forward_audio_frontend(inputA)
                    embedV = model.model.forward_visual_frontend(inputV)
                    out    = model.model.forward_audio_visual_backend(embedA, embedV)
                    score  = model.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        return numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
