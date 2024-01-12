import os
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from scipy.spatial import distance
import argparse
import shutil
import glob


class FaceIdentification:
    def __init__(self):
        # self.dataset_path = './input/'
        # self.update_dataset = './update/'
        self.app = FaceAnalysis(name='buffalo_s',providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.face_features = []

    def preprocess(self,dataset_path):
        self.dataset_path = dataset_path
        for person_name in os.listdir(self.dataset_path):
            file_path = os.path.join(self.dataset_path,person_name)
            if os.path.isdir(file_path):
                for image_name in os.listdir(file_path):
                    image_path = os.path.join(file_path,image_name)
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    result = self.app.get(img)
                    if len(result) > 1:
                        print('Warning:more one person in image')
                        continue
                    embedding = result[0]['embedding']
                    my_dict = {'name':person_name,"embedding":embedding}
                    self.face_features.append(my_dict)

        #print(face)
        np.save('Face_bank/face_bank.npy',self.face_features)

    def update(self,update_path):
        self.update_dataset = update_path
        self.preprocess(self.update_dataset)
        i=0
        subdirs = [x[1] for x in os.walk('update')]
        for root, dirs, files in os.walk("./update/"):
            for file in files:
                if file.endswith(".jpg"):
                    shutil.move('update/'+subdirs[0][0],os.path.join('input',subdirs[0][0]))
                i += 1
                if i == 1:
                    break


    def predict(self,imag_path,threshold):
        self.face_bank = np.load('Face_bank/face_bank.npy',allow_pickle=True)
        self.image_path = imag_path
        self.threshold = threshold
        self.img = cv2.imread(self.image_path)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.results = self.app.get(self.img)
        self.result_img = self.img.copy()
        for result in self.results:
            cv2.rectangle(self.result_img,(int(result.bbox[0]),int(result.bbox[1])),(int(result.bbox[2]),int(result.bbox[3]))
                        ,(255,0,0),2)
            for person in self.face_bank:
                person_embedding = person['embedding']
                new_person_embedding = result['embedding']
                dst = distance.euclidean(person_embedding,new_person_embedding)

                if dst < self.threshold:
                    cv2.putText(self.result_img,person['name'],(int(result.bbox[0])-40,int(result.bbox[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_AA)
                    print(person['name'])
                    break
                
            else:
                cv2.putText(self.result_img,'Unknown',(int(result.bbox[0])-40,int(result.bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_AA)
    
        self.result_img = cv2.resize(self.result_img, (650, 400)) 
        cv2.imshow('frame',cv2.cvtColor(self.result_img,cv2.COLOR_BGR2RGB)) 
        cv2.waitKey(0) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset',type=str)
    parser.add_argument('--image',type=str)
    parser.add_argument('--threshold',type=int,default=25)
    parser.add_argument('--update',type=str)
    opt = parser.parse_args()


    face_test = FaceIdentification()
    face_test.preprocess(opt.input_dataset)
    face_test.update(opt.update)
    face_test.predict(opt.image,opt.threshold)
