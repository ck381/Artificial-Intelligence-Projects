import math
import argparse
import os
import sys

from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder
import random
from PIL import Image
import numpy as np
import torch
import glob
from Core.Frame_Extractor import *

from RAFT.core.raft import RAFT


#  This class extracts a single static motion field that represents the general motion in the input video
class motion_estimator():


    def __init__(self,images_folder):


        self.DEVICE = 'cuda'
        self.video = r'Demo_motion_video\fire.mp4'
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default=r'RAFT\models\raft-sintel.pth', help="restore checkpoint")
        parser.add_argument('--path', default=images_folder,
                            help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        self.args = parser.parse_args()
        

        #extractor = Frame_Extractor(self.video)
        #frames = extractor.get_frames()


    # Loads in an image and converts to torch tensor on GPU
    def load_image(self,imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.DEVICE)

    # Calculate magnitude of flow vector
    def magnitude(self,x, y):
        m = math.hypot(x, y)
        return m
        
    # Calculate angle of flow vector
    def angle(self,x, y):
        a = math.degrees(math.atan(y / x))
        return a

    # Visualize flo
    def viz(self,img, flo, counter, args):

        img = img[0].permute(1, 2, 0).cpu().numpy()
        flo = flo[0].permute(1, 2, 0).cpu().numpy()

        return flo

    # Extracts the motion from the video and returns a single static flow field
    def demo(self,args):
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(self.DEVICE)
        model.eval()

        with torch.no_grad():
            images = glob.glob(os.path.join(args.path, '*.png')) + \
                     glob.glob(os.path.join(args.path, '*.jpg'))

            images = sorted(images)

            counter = 0

            width = None
            height = None
            flo_arrays = []


            for imfile1, imfile2 in zip(images[:-1], images[1:]):
                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)

                if counter < 1:
                    height = image1.shape[2]
                    width = image1.shape[3]


                counter = counter + 1

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                print(image1.shape, image2.shape)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                flo = self.viz(image1, flow_up, counter, args)
                flo_arrays.append(flo)
                counter = counter + 1

            length = len(flo_arrays)

            stored = np.zeros(shape=(height,width,2))
            print(stored.shape)

            for x in range(0,width):
                for y in range(0,height):
                    a = 0
                    b = 0
                    for z in range(0,length - 1):
                        a = a + flo_arrays[z][y][x][0]
                        b = b + flo_arrays[z][y][x][1]


                    stored[y][x][0] = a
                    stored[y][x][1] = b

            print(stored[200][200])

            # map flow to rgb image
            flo = flow_viz.flow_to_image(stored)
            img_flo = np.concatenate([flo], axis=0)


            # Visualize extracted flow field
            import cv2

            cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
            cv2.waitKey(0)

            return stored





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'models\raft-sintel.pth', help="Optical Flow model file")
    parser.add_argument('--path', default = r'Core\Video_Frames', help="Image frame directory")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    estimator = motion_estimator()

    flo = estimator.demo(args)






