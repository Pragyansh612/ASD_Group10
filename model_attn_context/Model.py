import torch
import torch.nn as nn
from model_attn_context.Classifier import Fusion, Detector
from model_attn_context.Encoder import visual_encoder, audio_encoder

class ASD_Model(nn.Module):
    def __init__(self):
        super(ASD_Model, self).__init__()
        self.visualEncoder = visual_encoder()
        self.audioEncoder  = audio_encoder()
        self.fusion   = Fusion(256)
        self.detector = Detector(256)
        # Attention gate: learns how much to weight the other face
        # Input: concat of target + other features -> scalar gate
        self.context_gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

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

    def forward_visual_frontend_multi(self, x_target, x_other=None):
        feat_target = self.forward_visual_frontend(x_target)  # (B, T, 128)
        if x_other is not None:
            feat_other = self.forward_visual_frontend(x_other)  # (B, T, 128)
            # Attention gate: concat target+other -> weight alpha
            combined = torch.cat([feat_target, feat_other], dim=-1)  # (B, T, 256)
            alpha = self.context_gate(combined)  # (B, T, 1)
            # Weighted combination: alpha controls how much other face contributes
            fused = feat_target + alpha * feat_other
        else:
            fused = feat_target
        return fused  # (B, T, 128)

    def forward(self, audioFeature, visualFeature, visualFeature2=None):
        audioEmbed  = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend_multi(visualFeature, visualFeature2)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        outsV  = self.forward_visual_backend(visualEmbed)
        return outsAV, outsV
