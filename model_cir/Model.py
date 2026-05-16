import torch
import torch.nn as nn
from model_cir.Classifier import Fusion, Detector
from model_cir.Encoder import visual_encoder, audio_encoder

class ASD_Model(nn.Module):
    def __init__(self):
        super(ASD_Model, self).__init__()
        self.visualEncoder = visual_encoder()
        self.audioEncoder  = audio_encoder()
        self.fusion   = Fusion(256)
        self.detector = Detector(256)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x  # (B, T, 128)

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x  # (B, T, 128)

    def forward_audio_visual_backend(self, x1, x2):
        x = self.fusion(x1, x2)
        x = self.detector(x)
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_frontend_multi(self, x_target, x_other=None, two_face_count=None, total_count=None):
        """
        Mean-pool target + other face embeddings.
        Falls back to target-only if x_other is None.
        Logs % of samples with 2 faces.
        """
        feat_target = self.forward_visual_frontend(x_target)  # (B, T, 128)

        if x_other is not None:
            feat_other = self.forward_visual_frontend(x_other)  # (B, T, 128)
            fused = 0.5 * feat_target + 0.5 * feat_other       # mean pool
            if two_face_count is not None and total_count is not None:
                two_face_count[0] += feat_target.shape[0]
        else:
            fused = feat_target  # fallback: single face
        
        if total_count is not None:
            total_count[0] += feat_target.shape[0]

        return fused  # (B, T, 128) — same shape, Fusion unchanged

    def forward(self, audioFeature, visualFeature, visualFeature2=None):
        audioEmbed  = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend_multi(visualFeature, visualFeature2)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        outsV  = self.forward_visual_backend(visualEmbed)
        return outsAV, outsV
