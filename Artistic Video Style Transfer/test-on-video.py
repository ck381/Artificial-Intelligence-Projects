from random import randint
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
import time
import keyboard
import sys
import Style_Transfer.flow_viz
import argparse
from Core.Styliser import *
import sys; sys.path.insert(1, "detectron2_repo\projects\PointRend")
import point_rend
from Core.Predictor import *
from Core.Frame_Extractor import *
from PIL import Image
import glob
from RAFT.core.raft import RAFT
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import *
from Core.flow_viz import *
import time


def get_list_frames(path): # Retrieves all .png files in the given directory and returns them as a list of numpy arrays

    image_list = []
    for filename in glob.glob(path + '\*.png'):
        im = Image.open(filename)
        im = np.array(im)
        image_list.append(im)

    return image_list

# Retrieves image size

def get_image_size(images): # Takes in a list of images and returns the size of the images
    image = images[0]
    x,y = image.shape[1],image.shape[0]
    return x,y

# Convert to torch tensor and move to GPU

def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()

    return img[None].to(DEVICE)


# Performs the correct stylisation techbnique based on the input parameters.
def get_direction(args,flo,img,stylised):

    direction = args.direction.lower()

    if direction == 'up':
        if flo[i][j][1] > 1:
            img[i][j] = stylised[i][j]
    if direction == 'down':
        if flo[i][j][1] < 1:
            img[i][j] = stylised[i][j]
    if direction == 'left':
        if flo[i][j][0] < 1:
            img[i][j] = stylised[i][j]
    if direction == 'right':
        if flo[i][j][0] > 1:
            img[i][j] = stylised[i][j]
    




# -------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = r'RAFT\models\raft-sintel.pth')
    parser.add_argument('--path', default = r'Demo\Demo-frames' ,help="Path containing input video frames")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--direction', default = 'right')
    parser.add_argument('--segmentation', default = 'y')
    ar = parser.parse_args()

    # --------------------------------------------- #

    # FRAME EXTRACTION

    #extractor = Frame_Extractor(args.video_path)
    # total_frames = extractor.get_frames()  # Deconstruct video into individual frames | Returns the total number of frames extracted (Integer)
    images = get_list_frames(ar.path)
    x,y = 1024, 436

    # --------------------------------------------- #

    # RAFT OPTICAL FLOW INITIALISATION

    DEVICE = 'cuda'

 
    model = torch.nn.DataParallel(RAFT(ar))
    model.load_state_dict(torch.load(ar.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    counter = 17
    total_frames = 3
    # --------------------------------------------- #

    # IMAGE DIRECTORY LOOP
    for i in range(1,3):

        original_video_path = r'Demo\Demo-frames'

        if counter > 9:
            frame = original_video_path + r'\frame_00'
        else:
            frame = original_video_path + r'\frame_000'

        frame = frame + str(counter) + '.png'


        pic = cv2.imread(frame)

        frame1 = None

        if counter + 1 > 9:
            frame1 = original_video_path + r'\frame_00'
        else:
            frame1 = original_video_path + r'\frame_000'

        frame1 = frame1 + str(counter + 1) + '.png'

        pic2 = cv2.imread(frame1)



        # --------------------------------------------- #

        # STYLE TRANSFER


        styliser = Styliser(r'Style_Transfer\Checkpoints\starry_night_10000.pth')
        transformer = styliser.create_model()
        im = Image.fromarray(np.uint8(Image.open(frame))).convert('RGB')
        image_tensor = styliser.prepare_input(im)
        styliser.stylise_image(image_tensor, transformer, 1)
        stylised = cv2.imread(r'Core\Stylised_Frames\stylised.jpg')



        # --------------------------------------------- #


        # SEMANTIC SEGMENTATION

        predictor = Predictor()
        predictor_o = predictor.get_predictor()
        print(type(images[i]))
        outputs = predictor.get_outputs(predictor_o, pic, x, y)

        masks = predictor.get_masks(outputs)
        boxes = predictor.get_boxes(outputs)
        classes = predictor.get_classes(outputs)
        masks = masks.cpu().detach().numpy()


        # --------------------------------------------- #

        # FLOW FIELD ESTIMATION

        a = time.time()

        flo = None

        with torch.no_grad():
            im1 = Image.fromarray(np.uint8(pic)).convert('RGB')
            im2 = Image.fromarray(np.uint8(pic2)).convert('RGB')

            image1 = load_image(im1)
            image2 = load_image(im2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            

            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            
           

        b = time.time()
        print(b - a)

        # --------------------------------------------- #

        # IMAGE RENDERING

        img = cv2.imread(frame)

        segmentation = ar.segmentation.lower()
        if segmentation == 'y':
            for z in range(0, len(classes)):
                print(z)
                if classes[z] == 0:
                    b = boxes[z]
                    b = predictor.boxToNumpy(b)
                    startX, startY, endX, endY = predictor.getBoundingBox(b)
                    for i in range(0, y):
                        for j in range(0, x):
                            get_direction(ar, flo, img, stylised)

        if segmentation == 'n':

            for i in range(0, y):
                for j in range(0, x):
                    get_direction(ar, flo, img, stylised)



        # --------------------------------------------- #

        # IMAGE VISUALISATION
        #print(type(img))
        #print(img)
        cv2.imshow('Input', img)
        cv2.waitKey(0)
        name = r'C:\Users\conor\Desktop\DynamicStylisation\Images\image' + str(counter) + '.png'
        counter = counter  + 1
        print(name)
        cv2.imwrite(name,stylised)







        






