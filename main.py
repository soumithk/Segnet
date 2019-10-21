import os
import cv2 as cv
import numpy as np
import data as dp
from keras import losses
from model import Segnet
from utils import data_generator, counter

def main(path, input_size, n_classes=2, batch_size=16, epochs_count=30):

    train_gen = data_generator(path, batch_size)
    model = Segnet(input_size, n_classes)
    model.build_model('adam', loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
    model.evaluate_generator(train_gen, 11, "./models/modelsweights.10.hdf5")
    # model.compile('sgd', loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
    # model.train_generator(train_gen, steps_per_epoch = 24,
    #         epochs=epochs_count, save_path = "./models")
    # print("Training Done....")
    # model.test_image("./models/modelsweights.30.hdf5", "./data/testing/image_2/000000_10.png", "./1.png")
    # model.test_image("./models/modelsweights.30.hdf5", "./data/testing/image_2/000001_10.png", "./2.png")
    # model.test_image("./models/modelsweights.30.hdf5", "./data/testing/image_2/000002_10.png", "./3.png")

if __name__ == '__main__':
    main('./data/cityscapes_data/', input_size=(224,224,3), epochs_count = 30)
