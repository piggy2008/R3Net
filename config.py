# coding: utf-8
import os

# szu 169 sever
datasets_root = '/home/ty/data'
# local pc
# datasets_root = '/home/qub/data/saliency'

# For each dataset, I put images and masks together
msra10k_path = os.path.join(datasets_root, 'msra10k')
ecssd_path = os.path.join(datasets_root, 'ecssd')
hkuis_path = os.path.join(datasets_root, 'hkuis')
pascals_path = os.path.join(datasets_root, 'pascals')
dutomron_path = os.path.join(datasets_root, 'dutomron')
sod_path = os.path.join(datasets_root, 'sod')
video_train_path = os.path.join(datasets_root, 'Pre-train')
davis_path = os.path.join(datasets_root, 'davis')