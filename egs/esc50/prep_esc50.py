# -*- coding: utf-8 -*-
# @Time    : 10/19/20 5:15 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_esc50.py

import numpy as np
import json
import os
import zipfile
import wget
import soundfile as sf
import argparse

# label = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc50_label.csv', delimiter=',', dtype='str')
# f = open("/data/sls/scratch/yuangong/aed-pc/src/utilities/esc_class_labels_indices.csv", "w")
# f.write("index,mid,display_name\n")
#
# label_set = []
# idx = 0
# for j in range(0, 5):
#     for i in range(0, 10):
#         cur_label = label[i][j]
#         cur_label = cur_label.split(' ')
#         cur_label = "_".join(cur_label)
#         cur_label = cur_label.lower()
#         label_set.append(cur_label)
#         f.write(str(idx)+',/m/07rwj'+str(idx).zfill(2)+',\"'+cur_label+'\"\n')
#         idx += 1
# f.close()
#

# Function to add Gaussian noise
def add_gaussian_noise(audio, noise_factor=0.1):
    """Adds Gaussian noise to an audio signal."""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

# Argument parser for noise level
parser = argparse.ArgumentParser(description='Prepare ESC-50 dataset with Gaussian noise.')
parser.add_argument('--noise_level', type=float, default=0.0, help='Level of Gaussian noise to add.')
args = parser.parse_args()

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
# Download and extract ESC-50 dataset if not present
if not os.path.exists('./data/ESC-50-master'):
    esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    wget.download(esc50_url, out='./data/')
    with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/')
    os.remove('./data/ESC-50-master.zip')

    # Define paths
    base_dir = './data/ESC-50-master/'
    audio_dir = base_dir + f'audio_16k_noise{args.noise_level:.1f}/'
    
    # Create directories if they don't exist
    os.makedirs(audio_dir, exist_ok=True)
    
    # Resample and add noise
    audio_list = get_immediate_files(base_dir + 'audio')
    for audio in audio_list:
        path = audio_dir + audio
        print('Processing ' + audio)
        os.system(f'sox {base_dir}/audio/{audio} -r 16000 {path}')
        y, sr = sf.read(path)
        y_noisy = add_gaussian_noise(y, noise_factor=args.noise_level)
        sf.write(path, y_noisy, sr)

label_set = np.loadtxt('./data/esc_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

# fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

for fold in [1,2,3,4,5]:
    base_path = "./data/ESC-50-master/audio_16k/"
    meta = np.loadtxt('./data/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
    train_wav_list = []
    eval_wav_list = []
    for i in range(0, len(meta)):
        cur_label = label_map[meta[i][3]]
        cur_path = meta[i][0]
        cur_fold = int(meta[i][1])
        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)

    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open('./data/datafiles/esc_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)

print('Finished ESC-50 Preparation')
