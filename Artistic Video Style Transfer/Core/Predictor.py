from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import sys; sys.path.insert(1, "detectron2_repo/projects/PointRend")
import point_rend

# This is a wrapper for the semantic segmentation network 
class Predictor:

    def __init__(self):
        self.cfg = None

    # This creates and returns an object that is later used for creating predictions
    def get_predictor(self):
        print('Setting up Detectron2 instance segmentation network...')
        setup_logger()
        cfg = get_cfg()
        self.cfg = cfg
        # Add PointRend module
        point_rend.add_pointrend_config(cfg)
        # Load suitable config file
        cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
        # cfg.merge_from_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"

        predictor = DefaultPredictor(cfg)



        return predictor


    # Retrieves the outputs using the predictor
    def get_outputs(self,predictor,image,x,y):
        # im = cv2.imread(image)  # Load the video frame
        im = image
        im = cv2.resize(im, (x, y), interpolation=cv2.INTER_AREA)
        #cv2.imwrite(image,im) # Write to file
        outputs = predictor(im)  # Perform instance segmentation on the frame

        #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2) # Visualizer object
        #v = v.draw_instance_predictions(outputs["instances"].to("cpu")) # Draw predictions and move to CPU
        #cv2.imwrite("img.jpg",v.get_image()[:, :, ::-1]) # Write predicted image to file

        return outputs

    # Returns masks from the predicted image
    def get_masks(self,outputs):
        masks = outputs["instances"].pred_masks  # Retrieve the frame masks
        return masks

    # Returns bounding boxes
    def get_boxes(self,outputs):
        boxes = outputs["instances"].pred_boxes  # Retrieve the frame prediction boxes
        return boxes

    # Returns a list of detected classes (eg. person)
    def get_classes(self,outputs):
        classes = outputs["instances"].pred_classes  # Retrieve the frame prediction boxes
        return classes

    # Converts bounding boxes tensor to numpy array 
    def boxToNumpy(self,b):
        b = b.get_tensor()
        b = b.cpu()
        b = b.numpy()
        return b

    # Retrieve the bounding box coordinates 
    def getBoundingBox(self,array):
        startX = int(round(array[0][0]))
        startY = int(round(array[0][1]))
        endX = int(round(array[0][2]))
        endY = int(round(array[0][3]))

        return startX, startY, endX, endY