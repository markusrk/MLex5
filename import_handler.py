from matplotlib import pyplot as plt
from pathlib import Path
from copy import deepcopy
import numpy as np

def train_test_split(x,y):
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    sss.get_n_splits(x, y)

    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return x_train,x_test,y_train,y_test
    #return {"x_train":x_train,"x_test":x_test,"y_train":y_train,"y_test":y_test}

def import_chars74k():
    # Import each letter to a separate list
    d = []
    p = Path.cwd()
    q = p / "chars74k-lite"
    print(q)
    for pth in q.iterdir():
        try:
            d.append([])
            for im in pth.iterdir():
                d[-1].append(plt.imread(im))
        except NotADirectoryError:
            del d[-1]
            continue
    x = []
    y = []
    c = [0] * 26
    for i in range(len(d)):
        for im in d[i]:
            x.append(im)
            ci = deepcopy(c)
            ci[i] = 1
            y.append(ci)
    return np.array(x), np.array(y)

def get_train_test_chars74():
    return train_test_split(*import_chars74k())

def get_normalized():
    x_train, x_test, y_train, y_test = get_train_test_chars74()
    return x_train/255, x_test/255, y_train, y_test

def get_pca(dims=20):
    x,y = import_chars74k()
    x = x.reshape((-1, 400))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=dims)
    pca.fit(x)
    x = pca.transform(x)
    return train_test_split(x,y)


def import_for_tf():
    x,y = import_chars74k()
    x = x.reshape((-1,400))
    d = []
    for i in range(len(x)):
        d.append([x[i],y[i]])
    return d

if __name__ == "__main__":
    import_chars74k()

    face = plt.imread('/Users/markus/PycharmProjects/MLex5/chars74k-lite/a/a_0.jpg')

    plt.imshow(face)
    plt.show()