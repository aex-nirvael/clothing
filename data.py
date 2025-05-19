'''
copyright Alex Whelan 2025
code for data processing
'''

import glob
import os
import cv2
import numpy as np

from torch.utils import data


class VITONDataset(data.Dataset):
    def __init__(self):
        super(VITONDataset, self).__init__()

        self.reference_path = "C:/Users/Alexf/Documents/Python Scripts/zalando-hd-resized/train/agnostic-v3.2"
        self.clothing_path = "C:/Users/Alexf/Documents/Python Scripts/zalando-hd-resized/train/cloth"
        self.gt_path = "C:/Users/Alexf/Documents/Python Scripts/zalando-hd-resized/train/image"

        self.reference_images = sorted(glob.glob(os.path.join(self.reference_path, "*.jpg")))
        self.clothing_images = sorted(glob.glob(os.path.join(self.clothing_path, "*.jpg")))
        self.gt_images = sorted(glob.glob(os.path.join(self.gt_path, "*.jpg")))

        if len(self.reference_images) != len(self.clothing_images) or len(self.reference_images) != len(self.gt_images):
            print(f"number of images not equal: ref {len(self.reference_images)}, cloth {len(self.clothing_images)}, gt {len(self.gt_images)}")



    def __len__(self):
        return len(self.reference_images)
    
    def load_image(self, image):

        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (192, 256), interpolation = cv2.INTER_LINEAR)

        # normalise to [-1,1]
        image = (image / 127.5) - 1.0

        return image
    
    
    def __getitem__(self, index):

        ref_path = self.reference_images[index]
        cloth_path = self.clothing_images[index]
        gt_path = self.gt_images[index]

        ref_image = self.load_image(ref_path)
        cloth_image = self.load_image(cloth_path)
        gt_image = self.load_image(gt_path)

        # stack or return tuple

        return (ref_image, cloth_image, gt_image)


class VITONDataLoader:
    def __init__(self, dataset, batch_size, train=True):
        super(VITONDataLoader, self).__init__()

        if train:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                num_workers=0, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
        

def save_image(image):
    """
    (B,3,H,W) tensor -> converts each image to np image and renormalise to [0,255]
    """
    image_np = image.permute(1,2,0).cpu().detach().numpy()
    # normalise to 0-255
    image_np = (image_np + 1.0) * 127.5

    return image_np


def save_images(refs, clothings, gts, preds, outdir, step):
    """
    (B,3,H,W) tensor -> saves each image in batch to out dir
    """
    for i, (ref, clothing, gt, pred) in enumerate(zip(refs, clothings, gts, preds)):
        ref_np = save_image(ref)
        clothing_np = save_image(clothing)
        gt_np = save_image(gt)
        pred_np = save_image(pred)

        all_images = np.concatenate((ref_np, clothing_np, gt_np, pred_np), axis=1)
        cv2.imwrite(f"{outdir}/image_step{step}_batch{i}.png", all_images)

