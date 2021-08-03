# This file will go to the root directory and execute commands to convert existing dicom images to png

import os
import subprocess

# directories = os.listdir('/home/piero/morespace/Documents/Phd/kaggle/data/train')
#
# for directory in directories:
#     subdirs = os.listdir(directory)
#     for subdir in subdirs:
#         os.wal

number = 100
count = 0
walker = os.walk('/home/piero/morespace/Documents/Phd/kaggle/data/train')

for (directory, subdirectory, file) in walker:
    if file:
        file_name = file[0].split('.')[0]
        d = subprocess.call(['dcm2pnm', '+on', f'{directory}/{file[0]}', f'/home/piero/morespace/Documents/Pedestrians-Instance-Segmentation/Kaggle/PNGImages/{file_name}.png'])
        print(d)
        count +=1

    if count > number:
        break
a = 1
