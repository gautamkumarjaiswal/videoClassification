# Video Classification
Real-time video classification using Deep Learning

Download complete folder and install packages from requirements.txt. There may be some additional packages, if you want to skip those then you have to manually select else you can install all using 
pip install -r requirements.txt

To train model from sketch use 'train.py'
To test pre-trained model use 'predict_video.py', before testing please download trained model and sample test video from google drive link given in the article.

To test trained model using webcam (real-time video) use 'predict_video_realtime.py'.

For more detail kindly visit article published at Medium

https://towardsdatascience.com/average-rolling-based-real-time-calamity-detection-using-deep-learning-ae51a2ffd8d2

Edit: Database collected to train model is now available at https://drive.google.com/file/d/1NvTyhUsrFbL91E10EPm38IjoCg6E2c6q/view?usp=sharing

Database size: 1.77GB
Number of classes: 4
1) Cyclone: 928 images
2) Earthquake: 1350 images
3) Flood: 1073 images
4) Wildfire: 1077 images

Another application of same proejct to detect Robbery, Accident and Fire is at
https://drive.google.com/file/d/11KBgD_W2yOxhJnUMiyBkBzXDPXhVmvCt/view?usp=sharing

Database size: 987MB
Number of classes: 3
1) Robbery: 2073 images
2) Accident: 887 images
3) Fire: 1405 images

1) Image samples are collected from google; therefore, pre-processing may be required.
2) Some image samples may be irrelevant, therefore remove it before training model.
3) As the number of images in each class is different, which may cause a biased result. Kindly balance it or use an appropriate technique such as Stratified k-fold to train the model.
4) Images may be subjected to copyright.
