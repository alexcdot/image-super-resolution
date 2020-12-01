import os
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from PIL import Image


train_lr_dir = 'div2k/DIV2K_train_LR_bicubic/X2'
train_cr_dir = 'div2k/DIV2K_train_LR_bicubic/fixed_CR_50'
train_hr_dir = 'div2k/DIV2K_train_HR'

if not os.path.exists(train_cr_dir):
    os.mkdir(train_cr_dir)


def create_images(inputs_list, index):
    for file in inputs_list:
        if 'png' in file:
            img_id = file[:4]
            if int(img_id) % 50 == 0:
                print(img_id)
            filepath = os.path.join(train_lr_dir, file)
            lr_img = Image.open(filepath)
            cr_filepath = filepath.replace('png', 'jpeg')
            lr_img.save(cr_filepath,'JPEG', dpi=[300, 300], quality=50)
            cr_img = Image.open(cr_filepath)

            cr_img_np = np.array(cr_img)
            lr_img_np = np.array(lr_img)

            split = 3
            h, w, _ = lr_img_np.shape

            h_subsize = h // split
            w_subsize = w // split
            for h_block in range(split):
                for w_block in range(split):
                    # patch resolution
                    pr_img_np = cr_img_np.copy()
                    pr_img_np[
                        h_block * h_subsize:(h_block + 1) * h_subsize,
                        w_block * w_subsize:(w_block + 1) * w_subsize
                    ] = lr_img_np[
                        h_block * h_subsize:(h_block + 1) * h_subsize,
                        w_block * w_subsize:(w_block + 1) * w_subsize
                    ]

                    pr_img = Image.fromarray(pr_img_np)
                    pr_id = h_block * split + w_block
                    pr_filepath = os.path.join(train_cr_dir, file.replace('.png', '_' + str(pr_id) + '.png'))
                    pr_img.save(pr_filepath)
    print(index + "is finished")

filelist = sorted(os.listdir(train_lr_dir))
num_workers = 16
num_files_per_worker = len(filelist) // num_workers

import multiprocessing

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    jobs = []
    for i in range(num_workers):
        sublist = filelist[i * num_files_per_worker:(i + 1) * num_files_per_worker]
        p = multiprocessing.Process(target=create_images, args=(sublist, i))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()