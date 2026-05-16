import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE
from loss_talknce import lossAV, lossV, TalkNCE
from model_talknce.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr=0.001, lrDecay=0.95, lambda_nce=0.1, **kwargs):
        super(ASD, self).__init__()
        self.model       = ASD_Model().cuda()
        self.lossAV      = lossAV().cuda()
        self.lossV       = lossV().cuda()
        self.talkNCE     = TalkNCE(temperature=0.07).cuda()
        self.lambda_nce  = lambda_nce
        self.optim       = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler   = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
              (sum(p.numel() for p in self.model.parameters()) / 1e6))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, lossV_acc, lossAV_acc, nce_acc, loss_acc = 0, 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']

        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audio  = audioFeature[0].cuda()
            visual = visualFeature[0].cuda()
            labels_cuda = labels[0].reshape((-1)).cuda()

            audioEmbed  = self.model.forward_audio_frontend(audio)
            visualEmbed = self.model.forward_visual_frontend(visual)
            outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsV  = self.model.forward_visual_backend(visualEmbed)

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels_cuda)
            nlossV  = self.lossV.forward(outsV, labels_cuda)

            # TalkNCE contrastive loss
            # audioEmbed, visualEmbed: (B, T, 128) → reshape to (B*T, 128)
            B = audioEmbed.shape[0]
            a_flat = audioEmbed.reshape(-1, 128)   # (B*T, 128)
            v_flat = visualEmbed.reshape(-1, 128)  # (B*T, 128)
            l_flat = labels_cuda                   # (B*T,)
            nce_loss = self.talkNCE(a_flat, v_flat, l_flat)

            nloss = nlossAV + 0.5 * nlossV + self.lambda_nce * nce_loss

            lossV_acc  += nlossV.detach().cpu().numpy()
            lossAV_acc += nlossAV.detach().cpu().numpy()
            nce_acc    += nce_loss.detach().cpu().numpy()
            loss_acc   += nloss.detach().cpu().numpy()
            top1  += prec
            nloss.backward()
            self.optim.step()
            index += len(labels_cuda)

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100*(num/loader.__len__())) +
                " LossAV: %.5f, NCE: %.5f, ACC: %2.2f%% \r" %
                (lossAV_acc/num, nce_acc/num, 100*(top1/index)))
            sys.stderr.flush()

        sys.stdout.write("\n")
        return loss_acc/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels_cuda = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels_cuda)
                predScore = predScore[:, 1].detach().cpu().numpy()
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
