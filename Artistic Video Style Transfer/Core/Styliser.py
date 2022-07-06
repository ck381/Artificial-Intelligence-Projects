from PIL import Image
from Style_Transfer.models import TransformerNet
from Style_Transfer.utils import style_transform, denormalize
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import cv2



# This class is a wrapper for the fast style transfer network
class Styliser():

    def __init__(self,checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = style_transform()

    def create_model(self):
        # Define model and load model checkpoint
        transformer = TransformerNet().to(self.device)
        transformer.load_state_dict(torch.load(self.checkpoint))
        transformer.eval()

        return transformer

    def prepare_input(self,image):
        # Prepare input
        image_tensor = Variable(self.transform(image)).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def stylise_image(self,image_tensor,transformer,a):
        # Stylize image
        with torch.no_grad():
            stylized_image = denormalize(transformer(image_tensor)).cpu()
            print(stylized_image.shape)
            print('SEARCH')
            save_image(stylized_image, r'Core\Stylised_Frames\stylised.jpg')



