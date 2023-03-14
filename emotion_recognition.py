from typing import Union

from collections import namedtuple
from PIL import Image, ImageOps
import numpy as np
from keras import models

EMOTIONS = {
    'anger': (-0.41, 0.79),  # гнев, злость
    'contempt': (-0.57, 0.66),  # презрение
    'disgust': (-0.67, 0.49),  # отвращение
    'fear': (-0.12, 0.78),  # страх
    'happy': (0.9, 0.16),  # веселый
    'neutral': (0.0, 0.0),  # нейтральный
    'sad': (-0.82, -0.4),  # грусть
    'surprise': (0.37, 0.91),  # удивление
    'uncertain': (-0.5, 0.0),  # неуверенность
}

EmotionPrediction = namedtuple('EmotionPrediction', ['emotion', 'probability'])
ValenceArousalPrediction = namedtuple('ValenceArousalPrediction', ['emotion', 'error', 'valence', 'arousal'])


class FaceEmotionRecognitionNet:

    def __init__(self, model_path: str, emotions=None):
        """
        file_path: путь к файлу сохраненной модели.
        emotions: предсказываемые эмоции.
        """
        # Загружаем модель
        if emotions is None:
            emotions = EMOTIONS
        self.__model = models.load_model(filepath=model_path, compile=False)
        self.__emotions = emotions

    def predict(self, face_image: np.array) -> Union[EmotionPrediction, ValenceArousalPrediction]:
        """Предсказание эмоции человека по изображению его лица.

        Аргументы:
        - face_image: изображение лица человека.
        """
        image = Image.fromarray(face_image)
        size = max(image.width, image.height)
        # Делаем изображение квадратным
        padded_image = ImageOps.pad(image, (size, size))
        # Подгоняем размер изображения
        resized_image = padded_image.resize(self.__model.input_shape[1:3])
        # Получаем предсказание
        tensor = np.asarray(resized_image)[None, ...]
        predicts = self.__model.predict(tensor, verbose=0)[0]
        # Готовим результирующие данные
        if isinstance(self.__emotions, (list, tuple)):
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
