import os
from collections import namedtuple
from typing import Union

import numpy as np
from PIL import Image, ImageOps

os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tf_keras import models

EMOTIONS = {
    'anger': (-0.41, 0.79),  # anger, rage
    'contempt': (-0.57, 0.66),  # contempt
    'disgust': (-0.67, 0.49),  # disgust
    'fear': (-0.12, 0.78),  # fear
    'happy': (0.9, 0.16),  # happiness
    'neutral': (0.0, 0.0),  # neutral
    'sad': (-0.82, -0.4),  # sadness
    'surprise': (0.37, 0.91),  # surprise
    'uncertain': (-0.5, 0.0),  # uncertain
}

EmotionPrediction = namedtuple('EmotionPrediction', ['emotion', 'probability'])
ValenceArousalPrediction = namedtuple('ValenceArousalPrediction', ['emotion', 'error', 'valence', 'arousal'])


class FaceEmotionRecognitionNet:

    def __init__(self, model_path: str, emotions=None):
        """
        file_path: Path to saved model.
        emotions: Emotion descriptions.
        """
        # Load model
        if emotions is None:
            emotions = EMOTIONS
        self.__model = models.load_model(filepath=model_path, compile=False, safe_mode=False)
        self.__emotions = emotions

    def predict(self, face_image: np.array) -> Union[EmotionPrediction, ValenceArousalPrediction]:
        """Предсказание эмоции человека по изображению его лица.

        Arguments:
        - face_image: изображение лица человека.
        """
        image = Image.fromarray(face_image)
        size = max(image.width, image.height)
        # Padding image to get squared form
        padded_image = ImageOps.pad(image, (size, size))
        # Resizing image to optimal size
        resized_image = padded_image.resize(self.__model.input_shape[1:3])
        # Getting a prediction
        tensor = np.asarray(resized_image)[None, ...]
        predicts = self.__model.predict(tensor, verbose=0)[0]
        if isinstance(self.__emotions, (list, tuple)):
            # Готовим результирующие данные
            probability = predicts.max()
            label = predicts.argmax()
            emotion = self.__emotions[label]
            return EmotionPrediction(emotion, probability)
        else:
            valence, arousal = predicts
            dists = np.apply_along_axis(lambda a: np.linalg.norm(a - np.array(list(self.__emotions.values())), axis=1),
                                        arr=predicts[None, ...], axis=1)
            error = dists.min()
            label = dists.argmin()
            emotion = list(self.__emotions.keys())[label]
            return ValenceArousalPrediction(emotion, error, valence, arousal)
