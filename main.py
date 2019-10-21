import os
import cv2 as cv
import numpy as np
import data as dp
from keras import losses
from model2 import Segnet
from utils import data_generator, counter

def main(path, input_size, n_classes=2, batch_size=16, epochs_count=30):

    train_gen = data_generator(path, batch_size)
    model = Segnet(input_size, n_classes)
    model.build_model('adam', loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
    # model.compile('sgd', loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
    # model.train_generator(train_gen, steps_per_epoch = 24,
    #         epochs=epochs_count, save_path = "./models")
    # print("Training Done....")
    model.test_generator(train_gen, counter(path, batch_size), "./models/weights.15.hdf5")

if __name__ == '__main__':
    main('./data/testing', input_size=(224,224,3), epochs_count = 30)

