import cv2
from constants import width, height
import numpy as np

class Sample_vid:
    def __init__(self, save_dir:str, sample:bool):

        self.length = 400
        self.dirs=[]
        self.tuples=[]
        self.save_dir = f"{save_dir}/sample_vid.avi"

        if sample == False:
            self.add_tuple = lambda x,y:None
            self.generate = lambda:None


    def add_tuple(self, data_tuple,dir):
        self.tuples.append(data_tuple)
        self.dirs.append(dir)

    def generate(self):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(self.save_dir, fourcc, 32.0, ((width*3), (height*2)), True)
        caps = []

        for dir in self.dirs:
            caps.append(cv2.VideoCapture(dir))


        for frame_n in np.arange(self.length):
            comb = ()
            for i, cap in enumerate(caps):
                _, frame_ = cap.read(cv2.IMREAD_GRAYSCALE)
                for marker in self.tuples[i]:
                    marker = marker[frame_n]
                    #print(marker)
                    frame_ = cv2.circle(frame_, (int(marker[0]), int(marker[1])), 4, (0,255,0), 1)

                comb = (*comb, frame_)
            comb_img = np.vstack((np.hstack(comb[:3]), np.hstack(comb[3:])))
            out.write(comb_img)
        out.release()
        print("video generated, ", self.save_dir)
