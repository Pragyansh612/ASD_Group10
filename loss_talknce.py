import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
    def __init__(self):
        super(lossAV, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels=None, r=1):
        x = x.squeeze(1)
        x = self.FC(x)
        if labels is None:
            predScore = x[:, 1]
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            x1 = x / r
            x1 = F.softmax(x1, dim=-1)[:, 1]
            nloss = self.criterion(x1, labels.float())
            predScore = F.softmax(x, dim=-1)
            predLabel = torch.round(F.softmax(x, dim=-1))[:, 1]
            correctNum = (predLabel == labels).sum().float()
            return nloss, predScore, predLabel, correctNum

class lossV(nn.Module):
    def __init__(self):
        super(lossV, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels, r=1):
        x = x.squeeze(1)
        x = self.FC(x)
        x = x / r
        x = F.softmax(x, dim=-1)
        nloss = self.criterion(x[:, 1], labels.float())
        return nloss

class TalkNCE(nn.Module):
    """
    Supervised contrastive loss for audio-visual correspondence.
    Applied only on active speaking frames (label=1).
    Pulls audio and visual embeddings of same speaking frame together,
    pushes apart speaking vs non-speaking embeddings.
    Reference: TalkNCE (ICASSP 2024)
    """
    def __init__(self, temperature=0.07):
        super(TalkNCE, self).__init__()
        self.temperature = temperature
        self.proj_audio  = nn.Linear(128, 64)
        self.proj_visual = nn.Linear(128, 64)

    def forward(self, audio_embed, visual_embed, labels):
        """
        audio_embed:  (T, 128)
        visual_embed: (T, 128)
        labels:       (T,) binary 0/1
        """
        if labels.sum() < 2:
            return torch.tensor(0.0, device=audio_embed.device)

        # Project to contrastive space
        a = F.normalize(self.proj_audio(audio_embed), dim=-1)   # (T, 64)
        v = F.normalize(self.proj_visual(visual_embed), dim=-1)  # (T, 64)

        # Only use speaking frames as positives
        speaking_mask = labels.bool()
        if speaking_mask.sum() < 2:
            return torch.tensor(0.0, device=audio_embed.device)

        a_spk = a[speaking_mask]  # (N_spk, 64)
        v_spk = v[speaking_mask]  # (N_spk, 64)

        # For each speaking frame: audio_i should be close to visual_i
        # and far from all other visual embeddings
        sim_matrix = torch.matmul(a_spk, v_spk.T) / self.temperature  # (N, N)

        # Diagonal = positive pairs (same frame)
        N = sim_matrix.shape[0]
        labels_nce = torch.arange(N, device=audio_embed.device)
        loss = F.cross_entropy(sim_matrix, labels_nce)
        return loss
