import os
import torch
import numpy as np
import imageio
import cv2


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class cityscapesLoader:

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, img_path, in_size):
        self.img_path = img_path
        self.n_classes = 19
        self.files = recursive_glob(rootdir=self.img_path, suffix=".png")
        self.files.sort()
        self.files_num = len(self.files)
        self.data = []
        self.size = (in_size[1], in_size[0])  # (width, height)
        self.mean = np.array([.485, .456, .406])
        self.std = np.array([.229, .224, .225])

    def load_frames(self):
        for idx in range(self.files_num):
            img_path = self.files[idx].rstrip()
            img_name = img_path.split('/')[-1]
            folder = img_path.split('/')[-2]

            img = imageio.imread(img_path)
            ori_h, ori_w = img.shape[:2]
            target_w, target_h = self.size

            # Compute scale to fit within target size while preserving aspect ratio
            scale = min(target_w / ori_w, target_h / ori_h)
            new_w = int(ori_w * scale)
            new_h = int(ori_h * scale)

            # Resize
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Padding
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top

            padded_img = cv2.copyMakeBorder(
                resized_img, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )

            # Normalize
            padded_img = padded_img / 255.0
            padded_img = (padded_img - self.mean) / self.std
            padded_img = padded_img.transpose(2, 0, 1)
            padded_img = padded_img[np.newaxis, :]
            padded_img = torch.from_numpy(padded_img).float()

            # Store image + metadata
            self.data.append([
                padded_img, img_name, folder,
                (ori_h, ori_w),  # original size
                (pad_top, pad_bottom, pad_left, pad_right),  # padding info
                (new_h, new_w)  # resized (before padding)
            ])

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb
