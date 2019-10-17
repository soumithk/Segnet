import os
import cv2 as cv
import numpy as np

class Test(object):
    """docstring forData."""

    def __init__(self, path,batchsize):
        # super(Data, self).__init__()
        self.path = path
        self.batchsize = batchsize
        self.cur_index = -batchsize
        self.dataread()

    def dataread(self):
        path = self.path + '../testing/image_2'
        self.test = []

        for filename in os.listdir(path):
          img = cv.imread(os.path.join(path,filename))

          if img is not None:
            img1 = cv.resize(img,(224,224))
            self.test.append(img1)

    def next(self):
        self.cur_index += self.batchsize

        if self.cur_index + self.batchsize >= len(self.images):
            return False, np.array([]) #, np.array([])

        return True,np.array(self.test[self.cur_index : self.cur_index + self.batchsize]) #,np.array(self.new_images[self.cur_index : self.cur_index + self.batchsize])#.reshape((-1, 224, 244, 1))#.reshape((self.batchsize,-1))
