import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer
from PIL import Image
import pdb

class GFPGAN():
    """
    version: GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3
    upscale: The final upsampling scale of the image. Default: 2
    bg_upsampler: background upsampler. Default: realesrgan
    bg_tile: Tile size for background sampler, 0 for no tile during testing. Default: 400
    only_center_face: Only restore the center face
    aligned: Input are aligned faces
    weight: Adjustable weights
    """

    def __init__(self,
                 version='1.3',
                 upscale=2,
                 bg_upsampler='realesrgan',
                 bg_tile=400,
                 only_center_face=None,
                 aligned=None,
                 weight=0.5,
                 ):
        
        self.version = version
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.bg_tile = bg_tile
        self.only_center_face = only_center_face
        self.aligned = aligned
        self.weight = weight

    # ------------------------ set up background upsampler ------------------------

    def process_images(self, img_list):

        if self.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                            'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=self.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        # ------------------------ set up GFPGAN restorer ------------------------
        if self.version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif self.version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif self.version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif self.version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif self.version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Wrong model version {self.version}.')

        # determine model paths
        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        restorer = GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

        # ------------------------ restore ------------------------
        restored_imgs = []
        for input_img in img_list:

            # restore faces and background if necessary
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=self.aligned,
                only_center_face=self.only_center_face,
                paste_back=True,
                weight=self.weight)
            
            # save restored img
            restored_imgs.append(restored_img)
        
        return restored_imgs
        

if __name__ == '__main__':

    gf = GFPGAN()

    #Bring in a random image

    img = Image.open("/Users/bryanchia/Desktop/stanford/classes/cs/cs324/fm-fairness/cheerleader-1a.png")
    
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    imgs = gf.process_images([img])

    img = Image.fromarray(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))

    img.save("/Users/bryanchia/Desktop/stanford/classes/cs/cs324/fm-fairness/cheerleader-1a-fixed.png")