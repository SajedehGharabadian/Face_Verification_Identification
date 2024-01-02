from insightface.app import FaceAnalysis
import cv2
import argparse
from scipy.spatial import distance
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--modelpack',type=str,default='buffalo_s')
parser.add_argument('--image1',type=str)
parser.add_argument('--image2',type=str)

opt = parser.parse_args()

app = FaceAnalysis(name=opt.modelpack,providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print(opt.image1)
path1 = opt.image1
img1 = cv2.imread(path1)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
result1 = app.get(img1)
embedding1 = result1[0]['embedding']

path2 = opt.image2
img2 = cv2.imread(path2)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
result2 = app.get(img2)
embedding2 = result2[0]['embedding']

dst = distance.euclidean(embedding1, embedding2)

fig = plt.figure(figsize=(6, 6))

if dst < 24:
    print('✅Same Person✅')
    fig.add_subplot(1,2, 1)
    plt.imshow(img1)
    fig.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.suptitle('Same Person', size=28)

    plt.show()
    

else:
    print('⛔Different Person⛔')
    fig.add_subplot(1,2, 1)
    plt.imshow(img1)
    fig.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.suptitle('Different Person',size=24)
    plt.show()
    
    
