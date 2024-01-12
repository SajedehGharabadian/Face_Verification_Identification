# Face Verification and Identification 

## Description

◻️ Use [insightface](https://github.com/deepinsight/insightface) libarary for verification face.

## How to install
```
pip insatll -r requirements.txt
```

# Verification

## How to run
```
python face_verification.py --image1 name_img1.jpg --image2 name_img2.jpg
```

# Identification

## Description
◻️ first we create face_bank.npy with images in input directory

◻️ when we want to use --update, we put directory of new person in update directory then run programme and subdirectory of update moves to input 

◻️ with --image, we can take infrence

## How to run
```
python class_face_identification.py --input_dataset ./input/ --update ./update/  --image infrence.jpg --threshold threshold
```

