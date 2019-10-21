from data import Data

#Generate Data in batches for Segnet to train
def data_generator(path, batchsize):

    while True:
        d = Data(path, batchsize)
        ret, x, y = d.next()

        while ret:
            yield x, y
            ret, x, y = d.next()

# Calculates No of Batches for Data
def counter(path, batchsize):

    d = Data(path, batchsize)
    ret, x, y = d.next()
    count  = 0

    while ret:
        count = count+1
        ret, x, y = d.next()

    return count
