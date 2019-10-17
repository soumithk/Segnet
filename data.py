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

        self.train = []
        path1 = self.path + '/image_2'
        for filename in os.listdir(path1):
          img = cv.imread(os.path.join(path1,filename))

          if img is not None:
            img1 = cv.resize(img,(224,224))
            self.train.append(img1)

    def datamodify(self):
        self.new_images = []

        for i in self.images:
          roi_i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
          mask = cv.inRange(roi_i, np.array([128, 63, 127]), np.array([129, 64, 128]))
          self.new_images.append(np.expand_dims(mask,axis=2))

    # def datawrite(self):
    #     save_path = self.path + '/semantic_gray/'
    #
    #     for i in range(len(self.new_images)):
    #       if(i<10):
    #         name = save_path+'00000'+str(i)+'_10.png'
    #
    #       elif (i>=10 and i<100):
    #         name = save_path+'0000' +str(i)+'_10.png'
    #
    #       else:
    #         name =  save_path+'000' + str(i) + '_10.png'
    #
    #       cv.imwrite(name,self.new_images[i])

    # def resize(self):
    #     self.size_images = []
    #     for i in new_images:
    #
    #         self.size_images.append(img)


    def flip(self):
        self.flip_images = []
        self.flip_mask = []

        for i in self.train:
            img_hor = cv.flip(i,1)
            self.flip_images.append(img_hor)

        for i in self.new_images:
            mas_hor = cv.flip(i,1)
            self.flip_mask.append(np.expand_dims(mas_hor, axis = 2))


    def next(self):
        self.cur_index += self.batchsize

        if self.cur_index + self.batchsize >= 2*len(self.images):
            return False, np.array([]), np.array([])

        elif self.cur_index  >= len(self.images):
            index = self.cur_index - len(self.images)
            return True,np.array(self.flip_images[index : index + self.batchsize]), np.array(self.flip_mask[index : index + self.batchsize])#.reshape((-1, 224, 244, 1))

        return True,np.array(self.train[self.cur_index : self.cur_index + self.batchsize]),np.array(self.new_images[self.cur_index : self.cur_index + self.batchsize])#.reshape((-1, 224, 244, 1))#.reshape((self.batchsize,-1))


    def run(self):
        self.dataread()
        self.datamodify()
        self.flip()
        # self.resize()
        # self.datawrite()
