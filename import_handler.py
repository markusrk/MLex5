from matplotlib import pyplot as plt
from pathlib import Path
from copy import deepcopy


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
    return x, y

if __name__ == "__main__":
    import_chars74k()

    face = plt.imread('/Users/markus/PycharmProjects/MLex5/chars74k-lite/a/a_0.jpg')

    plt.imshow(face)
    plt.show()