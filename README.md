# Emotion-Based-Face-Expression-Detection-Through-Facial-Feature-Recognition-OpenCv-and-CNN


# Facial Detection, Recognition and Emotion Detection

<img src="test_images/20180901150822_new.jpg" width="300" height= "0.2" />



## Introduction

Humans have always had the innate ability to recognize and distinguish between faces. The same has been achieved by computers using opencv and deep learning. This blog briefly throws some light on this ability of computers to excel in facial detection, facial recognition and emotion detection by using the results of experimentation and analysis done on these topics. The blog  has been divided into  three parts:

1. ### Facial Detection

2. ### Facial Recognition

3. ### Emotion Detection

We will walk around through these topics one by one briefly.

### Facial Detection

Detecting all the faces from an image. The facial detection is an first and important part in bringing out the results of facial recognition. It can be achieved by using the amazing python library "face_recognition" which performs very well in detecting location of faces from an image. The following image shows there are two faces detected from the given image.




![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

The below snippet shows how to use the face_recognition library for detecting faces.

```
face_locations = face_recognition.face_locations(image)
top, right, bottom, left = face_locations[0]
face_image = image[top:bottom, left:right]
```

The full code can be taken from the github.

### Facial Recognition

Facial Recognition verifies if  two faces are same. The use of facial recognition is huge in security, bio-metrics, entertainment, personal safety, etc. The python library "face_recognition" offers a very good performance in recognizing if two faces match with each other giving the result as True or False. The steps involved in facial recognition are

- Find face in an image
- Analyze facial feature
- Compare against both the faces
- Returns True if matched or else False.


### Conclusion:

​	We can clearly see the wonders of AI in [facial recognition](https://github.com/reddyprasade/Face--Recognition-with-Opencv). The amazing python library of 			  face_recognition, pretrained  deep learning models and open-cv  have already gained so much performance and have made our life easier. There are lots of other materials that are helpful and bring into picture different approaches used in achieving the same goal.

