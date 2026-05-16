import torch
import torch.nn as nn
import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE
from loss import lossAV, lossV
from model_ablation2.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr=0.001, lrDecay=0.95, **kwargs):
        super(ASD, self).__init__()
        self.model     = ASD_Model().cuda()
        self.lossAV    = lossAV().cuda()
        self.lossV     = lossV().cuda()
        self.optim     = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
              (sum(p.numel() for p in self.model.parameters()) / 1e6))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, lossV_acc, lossAV_acc, loss_acc = 0, 0, 0, 0, 0
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
            # mean-pool: target + other face (zeros if no second face → same as target-only)
            feat_target = self.model.forward_visual_frontend(visual)
            feat_other  = self.model.forward_visual_frontend(visual2)
            visualEmbed = 0.5 * feat_target + 0.5 * feat_other

            outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsV  = self.model.forward_visual_backend(visualEmbed)

            labels_cuda = labels[0].reshape((-1)).cuda()
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels_cuda)
            nlossV = self.lossV.forward(outsV, labels_cuda)
            nloss  = nlossAV + 0.5 * nlossV

            lossV_acc  += nlossV.detach().cpu().numpy()
            lossAV_acc += nlossAV.detach().cpu().numpy()
            loss_acc   += nloss.detach().cpu().numpy()
            top1  += prec
            nloss.backward()
            self.optim.step()
            index += len(labels_cuda)

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100*(num/loader.__len__())) +
                " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%%, 2-face: %.1f%% \r" %
                (lossV_acc/num, lossAV_acc/num, loss_acc/num, 100*(top1/index),
                 sum(epoch_two_pct)/len(epoch_two_pct)))
            sys.stderr.flush()

        mean_two_pct = sum(epoch_two_pct) / len(epoch_two_pct)
        sys.stdout.write("\n")
        sys.stdout.write("  >> Epoch %d: avg 2-face=%.1f%%, fallback=%.1f%%\n" %
                         (epoch, mean_two_pct, 100 - mean_two_pct))
        return loss_acc/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for batch in tqdm.tqdm(loader):
            audioFeature, visualFeature, visualFeature2, labels, _ = batch
            with torch.no_grad():
                audio   = audioFeature[0].cuda()
                visual  = visualFeature[0].cuda()
                visual2 = visualFeature2[0].cuda()

                audioEmbed  = self.model.forward_audio_frontend(audio)
                feat_target = self.model.forward_visual_frontend(visual)
                feat_other  = self.model.forward_visual_frontend(visual2)
                visualEmbed = 0.5 * feat_target + 0.5 * feat_other

                outsAV      = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels_cuda = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels_cuda)
                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)

        evalLines = open(evalOrig).read().splitlines()[1:]
        evalRes   = pandas.read_csv(evalOrig)
        evalRes['score'] = pandas.Series(predScores)
        evalRes['label'] = pandas.Series(['SPEAKING_AUDIBLE' for _ in evalLines])
        evalRes.drop(['label_id'],    axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s" % (evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState   = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model." % origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" %
                      (origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(loadedState[origName])
