import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE
from loss import lossAV, lossV
from model_ablation2.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr=0.001, lrDecay=0.95, lambda_scer=0.1, **kwargs):
        super(ASD, self).__init__()
        self.model       = ASD_Model().cuda()
        self.lossAV      = lossAV().cuda()
        self.lossV       = lossV().cuda()
        self.lambda_scer = lambda_scer
        self.optim       = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler   = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
              (sum(p.numel() for p in self.model.parameters()) / 1e6))

    def scer_loss(self, embed_single, embed_multi, labels_single, labels_multi):
        if embed_single.shape[0] < 2 or embed_multi.shape[0] < 2:
            return torch.tensor(0.0).cuda()
        spk_s = labels_single == 1
        spk_m = labels_multi  == 1
        if spk_s.sum() < 1 or spk_m.sum() < 1:
            return torch.tensor(0.0).cuda()
        mean_s = embed_single[spk_s].mean(dim=0)
        mean_m = embed_multi[spk_m].mean(dim=0)
        return F.mse_loss(mean_s, mean_m)

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, lossAV_acc, scer_acc = 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']

        for num, batch in enumerate(loader, start=1):
            audioFeature, visualFeature, visualFeature2, labels, two_pct = batch
            self.zero_grad()

            audio   = audioFeature[0].cuda()   # (N, T, 13*4)
            visual  = visualFeature[0].cuda()  # (N, T, H, W)
            visual2 = visualFeature2[0].cuda() # (N, T, H, W) — zero if no second face
            labels_cuda = labels[0].cuda()     # (N, T)

            # --- per-clip single/multi mask ---
            # visual2[i].sum() > 0 means clip i has a real second face
            has_second = (visual2.reshape(visual2.shape[0], -1).abs().sum(dim=1) > 0)  # (N,)

            # --- forward ---
            audioEmbed  = self.model.forward_audio_frontend(audio)
            feat_target = self.model.forward_visual_frontend(visual)
            feat_other  = self.model.forward_visual_frontend(visual2)
            visualEmbed = 0.5 * feat_target + 0.5 * feat_other

            fusedEmbed = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  # (N*T, 128)
            outsV      = self.model.forward_visual_backend(visualEmbed)

            labels_flat = labels_cuda.reshape(-1)
            nlossAV, _, _, prec = self.lossAV.forward(fusedEmbed, labels_flat)
            nlossV = self.lossV.forward(outsV, labels_flat)

            # --- SCER: split fusedEmbed by environment ---
            # fusedEmbed is (N*T, 128), has_second is (N,)
            # expand has_second to (N*T,)
            N = visual.shape[0]
            T = visual.shape[1]
            has_second_flat = has_second.unsqueeze(1).expand(N, T).reshape(-1)  # (N*T,)

            single_mask = ~has_second_flat
            multi_mask  =  has_second_flat

            if single_mask.sum() > 1 and multi_mask.sum() > 1:
                e_s = fusedEmbed[single_mask]
                e_m = fusedEmbed[multi_mask]
                l_s = labels_flat[single_mask]
                l_m = labels_flat[multi_mask]
                scer = self.scer_loss(e_s, e_m, l_s, l_m)
            else:
                scer = torch.tensor(0.0).cuda()

            nloss = nlossAV + 0.5 * nlossV + self.lambda_scer * scer

            lossAV_acc += nlossAV.detach().cpu().item()
            scer_acc   += scer.detach().cpu().item()
            top1  += prec
            nloss.backward()
            self.optim.step()
            index += labels_flat.shape[0]

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100*(num/loader.__len__())) +
                " LossAV: %.5f, SCER: %.08f, ACC: %2.2f%% \r" %
                (lossAV_acc/num, scer_acc/num, 100*(top1/index)))
            sys.stderr.flush()

        sys.stdout.write("\n")
        return lossAV_acc/num, lr

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
