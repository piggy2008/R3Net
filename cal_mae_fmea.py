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

# MSRA10K + DAVIS snap:10000 {'davis': {'mae': 0.03617724595417807, 'fmeasure': 0.8150494537915058}}
# MSRA10K + DAVIS(input no mea & std) snap:30000 {'davis': {'mae': 0.03403602471853535}, 'fmeasure': 0.8208723312824877}
# MSRA10K + DAVIS snap:30000 {'davis': {'mae': 0.02795341027164935}, 'fmeasure': 0.846696146351338}
# MSRA10K + DAVIS resize:473*473 snap:30000 {'davis': 'mae': 0.02464488739008121, ''fmeasure': 0.8753527027151914}
# MSRA10K + DAVIS finetune + GRU resize:473*473 snap:30000 {'davis': {'mae': 0.026164189245066576, 'fmeasure': 0.867808967452949}}
# MSRA10K + DAVIS finetune + GRU resize:473*473 snap:20000 {'davis': {'fmeasure': 0.8705511507216693, 'mae': 0.025285929420594357}}