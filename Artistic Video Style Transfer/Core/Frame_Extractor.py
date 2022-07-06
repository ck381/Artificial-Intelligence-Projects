import cv2


# This class breaks down an input video into its individual frames
class Frame_Extractor:
    def __init__(self,path):
        self.path = path
        self.writePath = r'Core\Video_Frames\frame'
        self.extension = '.jpg'

    # Saves each frame as an image and returns the total number of frames extrated
    def get_frames(self):
        frame_value = 0
        print('Extracting frames from the video...')


        cap = cv2.VideoCapture(self.path)
        i = 1

        while (cap.isOpened()):
            print(i)
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(self.writePath + str(i) + self.extension,frame)
            i += 1

        cap.release()
        cv2.destroyAllWindows()
        frame_value = i

        return frame_value