import os
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from PIL import Image


train_lr_dir = 'div2k/DIV2K_train_LR_bicubic/X2'
train_cr_dir = 'div2k/DIV2K_train_LR_bicubic/fixed_CR_50'
train_hr_dir = 'div2k/DIV2K_train_HR'

def compute_data_dict(filelist, data_dict):
    for file in filelist:
        if 'png' in file:
            img_id = file[:4]
            if int(img_id) % 50 == 0:
                print(img_id)

            curr_dict = {}
            hr_img = Image.open(os.path.join(train_hr_dir, file.replace('x2', '')))
            hr_img_np = np.array(hr_img)

            filepath = os.path.join(train_lr_dir, file)
            lr_img = Image.open(filepath)
            cr_filepath = os.path.join(train_cr_dir, file.replace('png', 'jpeg'))
            cr_img = Image.open(cr_filepath)
            img_size = os.path.getsize(filepath)
            cr_img_size = os.path.getsize(cr_filepath)

            if not os.path.exists(os.path.join(train_cr_dir + '_x2', file.replace('png', 'jpeg'))):
                print("did not find:", os.path.join(train_cr_dir + '_x2', file.replace('png', 'jpeg')))
                continue
            cr_img_np_x2 = np.array(Image.open(os.path.join(train_cr_dir + '_x2', file.replace('png', 'jpeg'))))
            lr_img_np_x2 = np.array(Image.open(os.path.join(train_lr_dir + '_x2', file)))
                        
            lr_ssim = ssim(lr_img_np_x2, hr_img_np, multichannel=True)
            cr_ssim = ssim(cr_img_np_x2, hr_img_np, multichannel=True)

            curr_dict['lr_filename'] = os.path.abspath(filepath)
            curr_dict['lr_ssim'] = lr_ssim
            curr_dict['lr_filesize'] = os.path.getsize(filepath)
            curr_dict['cr_ssim'] = cr_ssim
            curr_dict['cr_filesize'] = os.path.getsize(cr_filepath)

            split = 3
            pr_ssim_list = []
            naive_pr_ssim_list = []
            for pr_id in range(split ** 2):
                pr_img_np_x2 = np.array(Image.open(os.path.join(train_cr_dir + '_x2',
                    file.replace('.png', '_' + str(pr_id) + '.png'))))
                pr_img = Image.open(os.path.join(train_cr_dir,
                    file.replace('.png', '_' + str(pr_id) + '.png')))
                naive_pr_img_np_x2 = np.array(pr_img.resize(size=(pr_img.size[0]*2, pr_img.size[1]*2), resample=Image.BICUBIC))

                pr_ssim = ssim(pr_img_np_x2, hr_img_np, multichannel=True)
                naive_pr_ssim = ssim(naive_pr_img_np_x2, hr_img_np, multichannel=True)
                curr_dict['pr_' + str(pr_id) + '_ssim'] = pr_ssim
                curr_dict['naive_pr_' + str(pr_id) + '_ssim'] = naive_pr_ssim
                curr_dict['pr_' + str(pr_id) + '_filesize'] = os.path.getsize(
                    os.path.join(train_cr_dir,
                        file.replace('.png', '_' + str(pr_id) + '.png'))
                )

                pr_ssim_list.append(pr_ssim)
                naive_pr_ssim_list.append(naive_pr_ssim)

            curr_dict['pr_ssim_list'] = pr_ssim_list
            curr_dict['naive_pr_ssim_list'] = naive_pr_ssim_list
            curr_dict['best_patch'] = np.array(pr_ssim_list).argmax()
            curr_dict['best_patch_ssim'] = np.array(pr_ssim_list).max()
            curr_dict['naive_best_patch'] = np.array(naive_pr_ssim_list).argmax()
            curr_dict['naive_best_patch_ssim'] = pr_ssim_list[np.array(naive_pr_ssim_list).argmax()]

            data_dict[int(img_id)] = curr_dict

filelist = sorted(os.listdir(train_lr_dir))[:301]
num_workers = 16
num_files_per_worker = len(filelist) // num_workers

import multiprocessing
import pickle

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(num_workers):
        sublist = filelist[i * num_files_per_worker:(i + 1) * num_files_per_worker]
        p = multiprocessing.Process(target=compute_data_dict, args=(sublist, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    with open('data_dict_v3.pkl', 'wb') as handle:
        pickle.dump(dict(return_dict), handle, protocol=pickle.HIGHEST_PROTOCOL)