import os
import numpy as np
from PIL import Image
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure

ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2019-04-20 23:11:17'
name = 'davis'
gt_root = '/home/qub/data/saliency/davis/GT'

args = {
    'snapshot': '30000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
results = {}

save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']))
folders = os.listdir(save_path)
folders.sort()
for folder in folders:
    imgs = os.listdir(os.path.join(save_path, folder))
    imgs.sort()

    for img in imgs:
        print(os.path.join(folder, img))
        gt = np.array(Image.open(os.path.join(gt_root, folder, img)).convert('L'))
        pred = np.array(Image.open(os.path.join(save_path, folder, img)).convert('L'))
        precision, recall, mae = cal_precision_recall_mae(pred, gt)

        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)

fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

print ('test results:')
print (results)

# THUR15K + DAVIS snap:10000 {'davis': {'mae': 0.03617724595417807, 'fmeasure': 0.8150494537915058}}
# THUR15K + DAVIS(input no mea & std) snap:30000 {'davis': {'mae': 0.03403602471853535}, 'fmeasure': 0.8208723312824877}
# THUR15K + DAVIS snap:30000 {'davis': {'mae': 0.02795341027164935}, 'fmeasure': 0.846696146351338}
# THUR15K + DAVIS resize:473*473 snap:30000 {'davis': 'mae': 0.02464488739008121, ''fmeasure': 0.8753527027151914}
# THUR15K + DAVIS resize:473*473 model:R1 high and low, snap:30000 {'davis': {'fmeasure': 0.8657611483587979, 'mae': 0.028688147260396805}}
# THUR15K + DAVIS resize:473*473 model: model prior recurrent snap:30000 {'davis': {'mae': 0.02533309706615563, 'fmeasure': 0.8745875295714605}}
# THUR15K + DAVIS resize:473*473 model: model prior recurrent + feature maps plus
# snap:30000 {'davis': {'fmeasure': 0.8751256401745396, 'mae': 0.025352599605078505}}

# VideoSaliency_2019-05-03 00:54:21 is better, using model_prior, R3Net base and add previous frame supervision and recurrent GRU motion extraction
# training details, first, directly train R3Net using DAFB2 and THUR15K, second, finetune the model by add recurrent module and GRU, then finetune twice
# using DAFB2 and THUR15K but dataloader shuffle=false in order to have consecutive frames. The specific super parameter is in VideoSaliency_2019-05-03 00:54:21
# VideoSaliency_2019-05-01 23:29:39 and VideoSaliency_2019-04-20 23:11:17/30000.pth