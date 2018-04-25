import numpy as np
from sklearn.externals import joblib
from PIL import Image, ImageDraw


def detect(path, cutoff=0.75, vstep=5,hstep=5,size=20):
    # Load classifier and image
    clf = joblib.load('classifiers/current.pkl')
    img_gray = Image.open(path)
    ar = np.array(img_gray)/255
    # convert image to color
    img = Image.new('RGB',img_gray.size)
    img.paste(img)
    img_draw = ImageDraw.Draw(img)

    for y in range(0,len(ar)-size,vstep):
        for x in range(0,len(ar[0])-size,hstep):
            if clf.predict_proba(ar[y:y+size,x:x+size].reshape((-1,400))).max() > cutoff:
                draw(img_draw,(x,y),size)

    img.show()


def draw(imgd, point, size=20):
    imgd.rectangle((point[0],point[1],point[0]+size,point[1]+size),outline="red")




if __name__ == "__main__":
    print('started')
    detect('detection_images/detection-1.jpg')