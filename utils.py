from data import Data

def data_generator(path, batchsize):

    while True:
        d = Data(path, batchsize)
        ret, x, y = d.next()

        while ret:
            yield x, y
            ret, x, y = d.next()

def counter(path, batchsize):

    d = Data(path, batchsize)
    ret, x, y = d.next()
    count  = 0

    while ret:
        count = count+1
        ret, x, y = d.next()

    return count
