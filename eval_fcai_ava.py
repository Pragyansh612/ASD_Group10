import sys, os, torch, argparse, warnings, glob, pandas, tqdm, subprocess
from subprocess import PIPE
sys.path.insert(0, '/usershome/cs671_user6/asd_project/LR-ASD')
from ASD import ASD as ASD_baseline
from ASD_transformer import ASD as ASD_trans
from dataLoader import val_loader
from utils.tools import init_args
import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataPathAVA', type=str, default='data')
parser.add_argument('--savePath', type=str, default='exps/fcai_ava_eval')
parser.add_argument('--evalDataType', type=str, default='val')
args = parser.parse_args()
args = init_args(args)
os.makedirs(args.modelSavePath, exist_ok=True)

# Load both models
print("Loading models...")
baseline = ASD_baseline()
baseline.loadParameters('exps/exp_baseline/model/model_0022.model')
baseline.eval()

transformer = ASD_trans()
transformer.loadParameters('exps/ablation1_transformer/model/model_0039.model')
transformer.eval()

# Load val data
loader = val_loader(trialFileName=args.evalTrialAVA,
                    audioPath=os.path.join(args.audioPathAVA, args.evalDataType),
                    visualPath=os.path.join(args.visualPathAVA, args.evalDataType))
valLoader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=4)

# Get face counts from val_orig.csv to decide routing
import pandas as pd
gt_df = pd.read_csv('data/csv/val_orig.csv')
frame_face_counts = gt_df.groupby(['video_id','frame_timestamp'])['entity_id'].count()

predScores = []
for audioFeature, visualFeature, labels in tqdm.tqdm(valLoader):
    with torch.no_grad():
        # Get entity_id to look up face count - use batch size as proxy
        n_entities = audioFeature[0].shape[0]  # batch size = num faces in segment
        
        if n_entities >= 2:
            model = transformer
        else:
            model = baseline
        
        audioEmbed  = model.model.forward_audio_frontend(audioFeature[0].cuda())
        visualEmbed = model.model.forward_visual_frontend(visualFeature[0].cuda())
        outsAV = model.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        labels_cuda = labels[0].reshape((-1)).cuda()
        _, predScore, _, _ = model.lossAV.forward(outsAV, labels_cuda)
        predScore = predScore[:,1].detach().cpu().numpy()
        predScores.extend(predScore)

# Save and evaluate
evalLines = open(args.evalOrig).read().splitlines()[1:]
labels_s = pandas.Series(['SPEAKING_AUDIBLE' for _ in evalLines])
scores_s = pandas.Series(predScores)
evalRes = pandas.read_csv(args.evalOrig)
evalRes['score'] = scores_s
evalRes['label'] = labels_s
evalRes.drop(['label_id'], axis=1, inplace=True)
evalRes.drop(['instance_id'], axis=1, inplace=True)
evalRes.to_csv(args.evalCsvSave, index=False)

cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s" % (args.evalOrig, args.evalCsvSave)
mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
print(f"\nFCAI AVA val mAP: {mAP:.2f}%")
