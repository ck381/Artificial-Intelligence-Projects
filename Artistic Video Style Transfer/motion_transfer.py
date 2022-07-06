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
from Core.Predictor import *

from Core.flow_viz import *
import argparse



from Motion_Estimator import motion_estimator
from Core.Styliser import *
import sys; sys.path.insert(1, "detectron2_repo\projects\PointRend")
import point_rend






# -------------------------------------------------------------------------------------------------------------------

class NeuralNetwork:

    def __init__(self):
        self.content_image = r'C:\Users\conor\Desktop\NeuralTransfer\content_flo.png'
        self.style_image = r'C:\Users\conor\RAFT\RAFT\motion_image.png'
        self.content_weight = 0
        self.style_weight = 1e20
        self.style_weights = {'conv1_1': 1.55, 'conv2_1': 0.75, 'conv3_1': 0.55, 'conv4_1': 0.35, 'conv5_1': 0.25}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.vgg = None

        self.content_features = None
        self.style_features = None
        self.grams = None
        self.target = None
        self.optimizer = None

        self.final_result = None
        self.incrementer = 1

        self.motion_flo = None
        self.content_motion_flo = None
        torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------------------------------------------------------------------------

    # Loads the pretrained VGG model and returns it
    def getVGG(self):

        vgg = models.vgg19().features
        print(vars(vgg))

        for param in vgg.parameters():
           param.requires_grad_(False) # Makes the weights of the model unchangeable

        self.vgg = vgg


        return vgg # Return the model


    def get_features(self,motion_flo, model):

       layers = {'0': 'conv1_1','5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '23': 'conv4_2',  # content
                      '28': 'conv5_1'}

       features = {}
       x = motion_flo
       print(x.shape)



       for name, layer in model._modules.items():
           x = layer(x)
           if name in layers:
              features[layers[name]] = x

       return features

    def change_pooling(self):
        i = 0
        for layer in self.vgg:
            if isinstance(layer, torch.nn.Conv2d):
                self.vgg[i] = torch.nn.Conv2d(in_channels = 2, out_channels = 2, kernel_size=(2,2))
            i = i + 1
            #

    # Calculates the content loss (root mean squared error)
    def calculate_content_loss(self,target_features,content_features):
        difference = target_features['conv4_2'] - content_features['conv4_2']
        squared_difference = difference ** 2
        loss = torch.mean(squared_difference)
        return loss


    def process_Image(self,motion_flo):

        flo = self.transform(motion_flo).unsqueeze(0)

        return flo

    def generate_gram_matrix(self,features):

        grams = {}
        for layer in features:
            matrix = self.gram_matrix(self.style_features[layer])
            grams[layer] = matrix

        return grams

    def gram_matrix(self, input):
        a, b, c, d = input.size() 

        features = input.view(a * b, c * d) 
        G = torch.mm(features, features.t())  # compute the gram product

      

        # normalize gram matrix by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


    def gen_optimizer(self):

        optimizer = optim.Adam([self.target], lr=0.0016)
        return optimizer




    def setup(self):

        self.motion_flo = self.process_Image(self.motion_flo).to(self.device)
        self.motion_flo = self.motion_flo.float()

        self.content_motion_flo = self.process_Image(self.content_motion_flo).to(self.device)
        self.content_motion_flo = self.content_motion_flo.float()

        self.vgg = self.getVGG()
        self.vgg = self.vgg.to(self.device)

        # ------------------- # Style

        print(self.motion_flo.shape)


        self.content_features = self.get_features(self.content_motion_flo, self.vgg)
        self.style_features = self.get_features(self.motion_flo, self.vgg) # Get style features for the motion
        self.grams = self.generate_gram_matrix(self.style_features) # Get the gram matrices from the style features


        self.target = self.content_motion_flo.clone().requires_grad_(True).to(self.device)

        # ----------------- # Optimiser

        self.optimizer = self.gen_optimizer()


    def get_total_loss(self,s_loss,c_loss):
        a = c_loss * self.content_weight
        b = s_loss * self.style_weight
        loss = a + b
        return loss

    def run(self):

        for i in range(1,3000):

            print(i)

            target_features = self.get_features(self.target, self.vgg)

            content_loss = self.calculate_content_loss(target_features,self.content_features)

            style_loss = 0
            for layer in self.style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = self.grams[layer]
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (d * h * w)

            total_loss = self.get_total_loss(style_loss, content_loss)
            self.optimizer.zero_grad()  # Reset the gradients of the optimizer
            total_loss.backward(retain_graph = True)  # Propagate the total loss backwards
            self.optimizer.step()

            print(type(self.target))

            image = self.target.to("cpu").clone().detach()
            image = image.numpy().squeeze()

            print(total_loss)

            if i % 20 == 0:
                flo = flow_to_image(image)
                img_flo = np.concatenate([flo], axis=0)

                import cv2

                cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
                cv2.waitKey(1)

        return image
            



if __name__ == '__main__':

    
    # ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = r'RAFT\models\raft-sintel.pth')
    parser.add_argument('--path', default = r'Demo\Demo-frames' ,help="Path containing input video frames")
    parser.add_argument('--style_path', default = r'Core\Video_Frames' ,help="Path containing the style input video frames")
    parser.add_argument('--style_model', default = r'Style_Tranfer\Checkpoints' ,help="Style Transfer model path .pth")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    ar = parser.parse_args()



    network = NeuralNetwork()
    vgg = network.getVGG()
    network.change_pooling()

    # - Style motion estimator - #
    estimator = motion_estimator(ar.style_path)

    motion_flo = estimator.demo(estimator.args)
    network.motion_flo = motion_flo
    
    total_frames = 50


    # --------------------------------------------- #

    for i in range(1,total_frames):
    
        print(1)
    
        # Create a new style transfer network

        network = NeuralNetwork()
        vgg = network.getVGG()
        network.change_pooling()
        network.motion_flo = motion_flo # Style video motion representation


        # - Content motion estimator - #

        content_estimator = motion_estimator(ar.path)

        network.content_motion_flo = content_estimator.demo(content_estimator.args) # Content image flow field

        network.setup()
        stylised_motion = network.run() # Render new stylised flow field using content flow and style flow

   

        x = 1024 # Image width
        y = 436 # Image height

        frame = None

        original_video_path = ar.path

        counter = 1

        if counter > 9:
            frame = original_video_path + r'\frame_00'
        else:
            frame = original_video_path + r'\frame_000'

        frame = ar.path  + r'\frame_00'

        styliser = Styliser(ar.style_model)
        transformer = styliser.create_model()
        image_tensor = styliser.prepare_input(frame + str(counter) + '.png')
        styliser.stylise_image(image_tensor, transformer, 1)

        stylised = cv2.imread(r'Core\Stylised_Frames\stylised.jpg')

        predictor = Predictor()
        predictor_o = predictor.get_predictor()

        frame = frame + str(counter) + '.png'
        f = cv2.imread(frame)
        print(f)
        outputs = predictor.get_outputs(predictor_o, frame, x, y)
    
        masks = predictor.get_masks(outputs)
        boxes = predictor.get_boxes(outputs)
        classes = predictor.get_classes(outputs)
        print(classes)
        masks = masks.cpu().detach().numpy()


        img = f
        for z in range(0, len(classes)):
            if classes[z] == 0:
                b = boxes[z]
                b = predictor.boxToNumpy(b)
                startX, startY, endX, endY = predictor.getBoundingBox(b)
                for i in range(0, 436):
                    for j in range(0, 1024):
                        Flag = False
                        if stylised_motion[0][i][j] > 1:
                            Flag = True
                        if masks[z][i][j] == True and Flag:
                            img[i][j] = stylised[i][j]

        img_flo = np.concatenate([img], axis=0)

        cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
        cv2.waitKey(0)

        # --------------------------------------------- #









