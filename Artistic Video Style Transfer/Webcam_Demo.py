import cv2

from Style_Transfer.models import TransformerNet
from Style_Transfer.utils import *
from Core.Predictor import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from torchvision.utils import save_image
from PIL import Image

# Some basic setup:
# Setup detectron2 logger
import PIL
import detectron2
from PIL.Image import Image
from detectron2.utils.logger import setup_logger
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import *




# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



from PIL import Image

from torchvision import transforms, models


# -------------------------------------------------------------------------------------------------------------------- #

def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

setup_logger()

# -------------------------------------------------------------------------------------------------------------------- #


cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

counter = 0
network = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = style_transform()

# Define model and load model checkpoint
transformer = TransformerNet().to(device)
transformer.load_state_dict(torch.load(r'Style_Transfer\Checkpoints\starry_night_10000.pth'))
transformer.eval()

# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #


predictor = Predictor()
predictor_o = predictor.get_predictor()
DEVICE = 'cuda'


parser = argparse.ArgumentParser()
parser.add_argument('--model', default=r"RAFT\models\raft-sintel.pth")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
ar = parser.parse_args()

model = torch.nn.DataParallel(RAFT(ar))
model.load_state_dict(torch.load(ar.model))
model = model.module
model.to(DEVICE)
model.eval()

while True:

    ret, frame = cap.read()
    dim = (480,360)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite('webcam_frame' + str(counter) + '.jpg', frame)



    if counter % 2 == 0 and counter > 1:
        print(counter)



        # Prepare input
        image_tensor = Variable(transform(Image.open('webcam_frame' + str(counter) + '.jpg'))).to(device)
        image_tensor = image_tensor.unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_image = denormalize(transformer(image_tensor)).cpu()

        # Save image
        save_image(stylized_image, r'Core\Stylised_Frames\stylised.jpg')


        im = cv2.imread(r'Core\Stylised_Frames\stylised.jpg')


        print(frame.shape)
        x = 250
        y = 250


        outputs = predictor.get_outputs(predictor_o,frame, x, y)

        masks = predictor.get_masks(outputs)
        boxes = predictor.get_boxes(outputs)
        classes = predictor.get_classes(outputs)
        print(classes)
        masks = masks.cpu().detach().numpy()

        # FLOW FIELD ESTIMATION

        flo = None

        with torch.no_grad():
            path = cv2.imread('webcam_frame' + str(counter) + '.jpg')
            path2 = cv2.imread('webcam_frame' + str(counter - 1) + '.jpg')

            print(path)
            print(path2)

            im1 = Image.fromarray(np.uint8(path)).convert('RGB')
            im2 = Image.fromarray(np.uint8(path2)).convert('RGB')

            image1 = load_image(im1)
            image2 = load_image(im2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            print(flo.shape)




        img = frame

        for i in range(0, y):
            for j in range(0, x):
                Flag = False
                if flo[i][j][0] > 1:
                    Flag = True
                if Flag:
                    img[i][j] = im[i][j]


        

        cv2.imshow('Input', img)





        c = cv2.waitKey(1)
        if c == 27:
            break

        #
    counter = counter + 1

cap.release()
cv2.destroyAllWindows()