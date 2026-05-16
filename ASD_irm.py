import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE
from loss import lossAV, lossV
from model_ablation2.Model import ASD_Model  # uses multi-face dataloader

class ASD(nn.Module):
    def __init__(self, lr=0.001, lrDecay=0.95, lambda_irm=1.0, **kwargs):
        super(ASD, self).__init__()
        self.model      = ASD_Model().cuda()
        self.lossAV     = lossAV().cuda()
        self.lossV      = lossV().cuda()
        self.lambda_irm = lambda_irm
        self.optim      = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler  = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
              (sum(p.numel() for p in self.model.parameters()) / 1e6))

    def irm_penalty(self, logits, labels):
        """
        IRM penalty: gradient of loss w.r.t. dummy scalar.
        If large → representation is NOT invariant across environments.
        We minimize this to force invariant representations.
        """
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
        loss  = F.cross_entropy(logits * scale, labels)
        grad  = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, lossAV_acc, lossV_acc, irm_acc, loss_acc = 0, 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        epoch_two_pct = []

        for num, batch in enumerate(loader, start=1):
            audioFeature, visualFeature, visualFeature2, labels, two_pct = batch
            self.zero_grad()

            audio   = audioFeature[0].cuda()
            visual  = visualFeature[0].cuda()
            visual2 = visualFeature2[0].cuda()
            labels_cuda = labels[0].reshape((-1)).cuda()
            two_pct_val = float(two_pct)
            epoch_two_pct.append(two_pct_val)

            audioEmbed  = self.model.forward_audio_frontend(audio)
            feat_target = self.model.forward_visual_frontend(visual)
            feat_other  = self.model.forward_visual_frontend(visual2)
            visualEmbed = 0.5 * feat_target + 0.5 * feat_other
            outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsV  = self.model.forward_visual_backend(visualEmbed)

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels_cuda)
            nlossV = self.lossV.forward(outsV, labels_cuda)

            # IRM: define env1=single-face, env2=multi-face
            # Get logits for IRM penalty
            logits = self.lossAV.FC(outsAV.squeeze(1))  # (T, 2)

            if two_pct_val > 0:
                # env2: multi-face — compute IRM penalty
                irm_pen = self.irm_penalty(logits, labels_cuda)
            else:
                # env1: single-face — compute IRM penalty
                irm_pen = self.irm_penalty(logits, labels_cuda)

            # Anneal IRM: start with small lambda, increase over epochs
            # This follows the IRM paper recommendation
            irm_weight = self.lambda_irm * min(1.0, epoch / 5.0)
            nloss = nlossAV + 0.5 * nlossV + irm_weight * irm_pen

            lossAV_acc += nlossAV.detach().cpu().numpy()
            lossV_acc  += nlossV.detach().cpu().numpy()
            irm_acc    += irm_pen.detach().cpu().numpy()
            loss_acc   += nloss.detach().cpu().numpy()
            top1  += prec
            nloss.backward()
            self.optim.step()
            index += len(labels_cuda)

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100*(num/loader.__len__())) +
                " LossAV: %.5f, IRM: %.08f, ACC: %2.2f%%, 2-face: %.1f%% \r" %
                (lossAV_acc/num, irm_acc/num, 100*(top1/index),
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
