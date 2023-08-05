import torch
import pyiqa
import glob
import os
import csv

# print(pyiqa.list_models())

# create metric with default setting
iqa_metric = pyiqa.create_metric('lpips', device=torch.device('cuda'))

niqe_metric = pyiqa.create_metric('niqe', device=torch.device('cuda'))
brisque_metric = pyiqa.create_metric('brisque', device=torch.device('cuda'))

# check if lower better or higher better
# print(niqe_metric.lower_better)
# print(brisque_metric.lower_better)

# example for iqa score inference
# Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
# score_fr = iqa_metric(img_tensor_x, img_tensor_y)
# score_nr = iqa_metric(img_tensor_x)

input_dir = './Datasets/test/TestReal/input/'
result_dir = './results/TestReal/'

input_path = sorted(glob.glob(os.path.join(input_dir, '*.png')))
result_path = sorted(glob.glob(os.path.join(result_dir, '*.png')))
print(f'Length of input is {len(input_path)} and length of result is {len(result_path)}')
# assert len(input_path) == len(result_path), 'The input and result image should have the same number!'

f = open('result.csv', 'w', encoding='utf8', newline='')
csvwrite = csv.writer(f)
csvwrite.writerow(['Image_Index', 'NIQE', 'BRISQUE'])

niqe_total = 0.
brisque_total = 0.
for i in range(len(input_path)):
    score_niqe = niqe_metric(result_path[i])
    score_brisque = brisque_metric(result_path[i])
    print(f'Image {i+1}, NIQE is {score_niqe.item()} amd BRISQUE is {score_brisque.item()}')
    csvwrite.writerow([i, score_niqe.item(), score_brisque.item()])
    niqe_total += score_niqe.item()
    brisque_total += score_brisque.item()

niqe_avg = niqe_total / len(input_path)
brisque_avg = brisque_total / len(input_path)
print(f'Average NIQE is {niqe_avg} and Average BRISQUE is {brisque_avg}')