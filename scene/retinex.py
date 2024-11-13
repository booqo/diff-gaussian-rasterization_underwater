import os

import numpy as np
import cv2
import torch
import rawpy



class retinex:
    def __init__(self):
        self.sigma_list = [15, 80, 250]
        self.low_clip = 0.01
        self.high_clip = 0.99
        self.add = 0.5
        # self.intensity = None
        pass

    def forward(self, img, save_path):

        img = np.uint8(img.numpy()*255) + 1

        intensity = np.sum(img, axis=0) / img.shape[0]

        # multi-scale retinex
        retinex = np.zeros_like(intensity)
        for sigma in self.sigma_list:
            retinex += np.log10(intensity) - np.log10(cv2.GaussianBlur(intensity, (0, 0), sigma))

        retinex = retinex / len(self.sigma_list)


        intensity1 = np.expand_dims(retinex, 0)

        total = intensity1.shape[0] * intensity1.shape[1]
        for i in range(intensity1.shape[2]):
            unique, counts = np.unique(intensity1[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < self.low_clip:
                    low_val = u
                if float(current) / total < self.high_clip:
                    high_val = u
                current += c

            intensity1[:, :, i] = np.maximum(np.minimum(intensity1[:, :, i], high_val), low_val)

        intensity1 = (intensity1 - np.min(intensity1)) / \
                     (np.max(intensity1) - np.min(intensity1)) * \
                     255.0 + 1.0

        img_enhance = np.zeros_like(img)

        for y in range(img_enhance.shape[1]):
            for x in range(img_enhance.shape[2]):
                B = np.max(img[:, y, x])
                A = np.minimum(256.0 / B, intensity1[0, y, x] / intensity[y, x])
                img_enhance[0, y, x] = A * img[0, y, x]
                img_enhance[1, y, x] = A * img[1, y, x]
                img_enhance[2, y, x] = A * img[2, y, x]

        img_enhance = np.uint8(img_enhance - 1.0)

        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)

        cv2.imwrite(save_path, cv2.cvtColor(np.uint8(img_enhance.transpose(1, 2, 0)), cv2.COLOR_RGB2BGR))


        img_enhance = np.float64(img_enhance)
        return torch.tensor(img_enhance/255)

    def backward(self, img):
        pass

