import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE
from loss import lossAV, lossV
from model_cir.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr=0.001, lrDecay=0.95, lambda_cir=0.1, **kwargs):
        super(ASD, self).__init__()
        self.model      = ASD_Model().cuda()
        self.lossAV     = lossAV().cuda()
        self.lossV      = lossV().cuda()
        self.lambda_cir = lambda_cir
        self.optim      = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler  = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
              (sum(p.numel() for p in self.model.parameters()) / 1e6))

    def context_independence_loss(self, outsAV_target, outsAV_other):
        """
        Penalize correlation between predictions for face A and face B.
        outsAV_target, outsAV_other: (T, 128) features before FC
        """
        # Apply FC to get logits then probabilities
        logits_target = self.lossAV.FC(outsAV_target.squeeze(1))  # (T, 2)
        logits_other  = self.lossAV.FC(outsAV_other.squeeze(1))   # (T, 2)
        prob_target = F.softmax(logits_target, dim=-1)[:, 1]  # (T,)
        prob_other  = F.softmax(logits_other,  dim=-1)[:, 1]  # (T,)

        if prob_target.shape[0] < 2:
            return torch.tensor(0.0).cuda()

        # Center
        prob_target = prob_target - prob_target.mean()
        prob_other  = prob_other  - prob_other.mean()

        # Pearson correlation
        num   = (prob_target * prob_other).sum()
        denom = (prob_target.norm() * prob_other.norm()).clamp(min=1e-8)
        corr  = num / denom  # scalar

        # Penalize any strong correlation (positive or negative)
        cir_loss = corr.abs()
        return cir_loss

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, lossV_acc, lossAV_acc, loss_acc, cir_acc = 0, 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        epoch_two_pct = []

        for num, batch in enumerate(loader, start=1):
            audioFeature, visualFeature, visualFeature2, labels, two_pct = batch
            self.zero_grad()
            audio   = audioFeature[0].cuda()
            visual  = visualFeature[0].cuda()
            visual2 = visualFeature2[0].cuda()
            epoch_two_pct.append(float(two_pct))

            audioEmbed  = self.model.forward_audio_frontend(audio)

            # Encode both faces separately
            feat_target = self.model.forward_visual_frontend(visual)
            feat_other  = self.model.forward_visual_frontend(visual2)

            # Mean pool for ASD prediction (same as ablation2)
            visualEmbed = 0.5 * feat_target + 0.5 * feat_other

            outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsV  = self.model.forward_visual_backend(visualEmbed)

            labels_cuda = labels[0].reshape((-1)).cuda()
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels_cuda)
            nlossV = self.lossV.forward(outsV, labels_cuda)

            # CIR loss: get predictions for other face independently
            if float(two_pct) > 0:
                audio_other  = self.model.forward_audio_frontend(audio)
                visualEmbed_other = feat_other
                outsAV_other = self.model.forward_audio_visual_backend(
                    audio_other, visualEmbed_other)
                cir_loss = self.context_independence_loss(outsAV, outsAV_other)
            else:
                cir_loss = torch.tensor(0.0).cuda()

            nloss = nlossAV + 0.5 * nlossV + self.lambda_cir * cir_loss

            lossV_acc  += nlossV.detach().cpu().numpy()
            lossAV_acc += nlossAV.detach().cpu().numpy()
            loss_acc   += nloss.detach().cpu().numpy()
            cir_acc    += cir_loss.detach().cpu().numpy()
            top1  += prec
            nloss.backward()
            self.optim.step()
            index += len(labels_cuda)

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100*(num/loader.__len__())) +
                " LossAV: %.5f, CIR: %.8f, ACC: %2.2f%%, 2-face: %.1f%% \r" %
                (lossAV_acc/num, cir_acc/num, 100*(top1/index),
                 sum(epoch_two_pct)/len(epoch_two_pct)))
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
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels_cuda = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels_cuda)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
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
