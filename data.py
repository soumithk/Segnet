import os
import cv2 as cv
import numpy as np


class Data(object):
    """docstring forData."""

    def __init__(self, path,batchsize):
        # super(Data, self).__init__()
        self.path = path
        self.batchsize = batchsize
        self.cur_index = -batchsize
        self.run()

    def dataread(self):
        path = self.path + '/semantic_rgb'
        self.images = []
        for filename in os.listdir(path):
          img = cv.imread(os.path.join(path,filename))
          if img is not None:
            img1 = cv.resize(img,(224,224))
            self.images.append(img1)

    def datamodify(self):
        self.new_images = []
        for i in self.images:
          roi_i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
          mask = cv.inRange(roi_i, np.array([128, 63, 127]), np.array([129, 64, 128]))
          self.new_images.append(mask)

    def datawrite(self):
        save_path = self.path + '/semantic_gray/'
        for i in range(len(self.new_images)):
          if(i<10):
            name = save_path+'00000'+str(i)+'_10.png'
          elif (i>=10 and i<100):
            name = save_path+'0000' +str(i)+'_10.png'
          else:
            name =  save_path+'000' + str(i) + '_10.png'
          cv.imwrite(name,self.new_images[i])

    # def resize(self):
    #     self.size_images = []
    #     for i in new_images:
    #
    #         self.size_images.append(img)

    def next(self):
        self.cur_index += self.batchsize
        if self.cur_index + self.batchsize >= len(self.images):
            return False, np.array([]), np.array([])

        return True, np.array(self.images[self.cur_index : self.batchsize]),
                        np.array(self.new_images[self.cur_index : self.cur_index + self.batchsize])


    def run(self):
        self.dataread()
        self.datamodify()
        # self.resize()
        # self.datawrite()
