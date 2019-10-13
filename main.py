import os
import cv2 as cv
import numpy as np
import data as dp


if __name__ == '__main__':
    path = '../data_semantics/training'
    dat = dp.Data(path,10)
    dat.run()
    img,mask = dat.next()
