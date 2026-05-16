import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE
from loss import lossAV, lossV
from model_ablation2.Model import ASD_Model

class CaMIB_Module(nn.Module):
    """
    Causal Information Bottleneck for ASD.
    Splits 128-dim embedding into:
    - causal (64-dim): lip-audio synchrony, transfers across domains
    - shortcut (64-dim): inter-face co-occurrence, domain-specific
    Predicts from causal only, isolates shortcut.
    """
    def __init__(self, embed_dim=128, causal_dim=64):
        super(CaMIB_Module, self).__init__()
        self.causal_dim   = causal_dim
        self.shortcut_dim = embed_dim - causal_dim

        # Mask generator: learns to separate causal from shortcut
        self.mask_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()  # soft mask in [0,1]
        )

        # Causal encoder: extract causal features
        self.causal_proj = nn.Linear(embed_dim, causal_dim)

        # Shortcut encoder: extract shortcut features
        self.shortcut_proj = nn.Linear(embed_dim, self.shortcut_dim)

        # Final predictor uses only causal features
        self.predictor = nn.Linear(causal_dim, 128)  # back to 128 for lossAV

    def forward(self, embed):
        """
        embed: (T, 128)
        returns: causal_embed (T, 128), ib_loss scalar
        """
        # Generate soft mask
        mask = self.mask_gen(embed)  # (T, 128)

        # Split into causal and shortcut
        causal_embed   = self.causal_proj(embed * mask)          # (T, 64)
        shortcut_embed = self.shortcut_proj(embed * (1 - mask))  # (T, 64)

        # Information bottleneck: minimize mutual info between shortcut and label
        # Proxy: minimize shortcut variance (compress irrelevant info)
        ib_loss = shortcut_embed.var(dim=0).mean()

        # Causal independence: causal and shortcut should be orthogonal
        if causal_embed.shape[0] > 1:
            # Covariance between causal and shortcut should be zero
            c = causal_embed - causal_embed.mean(0)
            s = shortcut_embed - shortcut_embed.mean(0)
            # Cross-covariance matrix (64x64)
            cross_cov = torch.mm(c.T, s) / (c.shape[0] - 1)
            orthog_loss = cross_cov.pow(2).sum()
        else:
            orthog_loss = torch.tensor(0.0, device=embed.device)

        # Project causal back to 128-dim for lossAV compatibility
        causal_out = self.predictor(causal_embed)  # (T, 128)

        return causal_out, ib_loss + 0.1 * orthog_loss


class ASD(nn.Module):
    def __init__(self, lr=0.001, lrDecay=0.95, lambda_camib=0.1, **kwargs):
        super(ASD, self).__init__()
        self.model        = ASD_Model().cuda()
        self.lossAV       = lossAV().cuda()
        self.lossV        = lossV().cuda()
        self.camib        = CaMIB_Module(embed_dim=128, causal_dim=64).cuda()
        self.lambda_camib = lambda_camib
        self.optim        = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler    = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
              (sum(p.numel() for p in self.model.parameters()) / 1e6))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, lossAV_acc, camib_acc, loss_acc = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        epoch_two_pct = []

        for num, batch in enumerate(loader, start=1):
            audioFeature, visualFeature, visualFeature2, labels, two_pct = batch
            self.zero_grad()

            audio   = audioFeature[0].cuda()
            visual  = visualFeature[0].cuda()
            visual2 = visualFeature2[0].cuda()
            labels_cuda = labels[0].reshape((-1)).cuda()
            epoch_two_pct.append(float(two_pct))

            audioEmbed  = self.model.forward_audio_frontend(audio)
            feat_target = self.model.forward_visual_frontend(visual)
            feat_other  = self.model.forward_visual_frontend(visual2)
            visualEmbed = 0.5 * feat_target + 0.5 * feat_other
            outsAV_raw  = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsV       = self.model.forward_visual_backend(visualEmbed)

            # Apply CaMIB: split into causal and shortcut
            outsAV, camib_loss = self.camib(outsAV_raw)

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels_cuda)
            nlossV = self.lossV.forward(outsV, labels_cuda)
            nloss  = nlossAV + 0.5 * nlossV + self.lambda_camib * camib_loss

            lossAV_acc += nlossAV.detach().cpu().numpy()
            camib_acc  += camib_loss.detach().cpu().numpy()
            loss_acc   += nloss.detach().cpu().numpy()
            top1  += prec
            nloss.backward()
            self.optim.step()
            index += len(labels_cuda)

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100*(num/loader.__len__())) +
                " LossAV: %.5f, CaMIB: %.08f, ACC: %2.2f%% \r" %
                (lossAV_acc/num, camib_acc/num, 100*(top1/index)))
            sys.stderr.flush()

        sys.stdout.write("\n")
        return loss_acc/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for batch in tqdm.tqdm(loader):
            audioFeature = batch[0]
            visualFeature = batch[1]
            labels = batch[2]
            with torch.no_grad():
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outsAV_raw  = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                outsAV, _   = self.camib(outsAV_raw)
                labels_cuda = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels_cuda)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = pandas.Series(['SPEAKING_AUDIBLE' for _ in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s" % (evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model." % origName)
                    continue
            para = loadedState[origName]
            if selfState[name].size() != para.size():
                print("Wrong parameter length: %s" % origName)
                continue
            selfState[name].copy_(para)
