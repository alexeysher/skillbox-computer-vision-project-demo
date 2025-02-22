#### About app
It\'s an easy-to-use app for creating a trailer purpose for videos with one person.
This app demonstrates abilities of a special computer vision 
[model](https://github.com/alexeysher/skillbox-computer-vision-project).
The model is deployed on [Vertex AI Platform](https://cloud.google.com/vertex-ai) 
which is a part of [Google Cloud Platform](https://cloud.google.com).
Face detecting and video processing are realized utilizing [OpenCV](https://docs.opencv.org/).

#### About model
The model predicts valence (positivity) and arousal (intensity) of human emotion
by his facial expression.
The one is based on [EfficientNetB0](https://arxiv.org/abs/1905.11946) model.
[TenserFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) frameworks 
have been utilized for model building.
[Transfer learning & fine turning](https://keras.io/guides/transfer_learning/) 
approaches were applied during model learning.

#### How it works
In the beginning, the app recognize the human's emotion intensity in each frame of video.

After that app is searching for fragments containing a peak of emotion intensity.
Each such one should meet following criteria:
- emotion intensity values are located between specified bounds 
- frag
- sss

#### How to use
For trailer creation just follow the next steps:
1. Upload video from your file storage.

    Just drag it from your file explorer and drop it on the related control in [Video](http://localhost:8501/#video). 
    Alternatively, you can click the *Browser files* button and select the one in the file explorer.
    When file uploading finish the video will be opened in the player 
    under the file downloading control.

2. Wait util the video will be processed

    Emotion intensity in each point will be recognized.
    When the process ends you will see the intensity chart under the video player.

3. Adjust fragment detection criteria:

    Use the related controls in [Fragments](http://localhost:8501/#fragments) section. 
    Detected fragments are listed in the table under the controls.

4. Chose fragments that should be included in trailer.

    Just select in 

5. Save and download trailer.

    Click `Create trailer...` button in [Trailer](http://localhost:8501/#trailer) section.
    When trailer creation finish click `Download trailer...` button to save the trailer to your storage.

#### Note
For testing app please download a small [collection]() of short videos.
