import hashlib
import pickle
import math
from datetime import timedelta, datetime
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from google.cloud import storage
from google.cloud import aiplatform
from google.oauth2 import service_account
from scipy.signal import find_peaks

import subprocess


st.set_page_config(page_title='Creating trailers', page_icon=':film_frames:',
                   layout="wide", initial_sidebar_state="expanded",
                   menu_items={
                       "About": "# This is a header. This is an *extremely* cool app!"
                   }
)

BACKGROUND_COLOR = 'white'
COLOR = 'black'

# Settings
MIN_STEP = 0.1
MAX_POINTS_NUMBER = 500

# Face detector settings
HAAR_FILE = 'haarcascade_frontalface_default.xml'
SCALE_FACTOR = 1.1  # Scale factor
MIN_NEIGHBORS = 3  # Min. neighbors number
MIN_FACE_SIZE = 64  # Min. face size

# Google Cloud
GC_CREDENTIAL_INFO = st.secrets['gc-service-account'] # Credential info
GC_BUCKET_ID = st.secrets['gc-storage']['bucket_id'] # Bucket id
GC_LOCATION_ID = st.secrets['gc-aiplatform']['location_id'] # Location id
GC_ENDPOINT_ID = st.secrets['gc-aiplatform']['endpoint_id'] # Endpoint id
GC_INTENSITIES_PATH = 'intensities'  # Processed video intensity of intensities folder path
GC_HYPERPARAMS_PATH = 'hyperparams'  # Processed video hyperparams folder path
GC_FRAGMENTS_PATH = 'fragments'  # Processed video fragments folder path
GC_TRAILER_PATH = 'trailer'  # Processed video fragments folder path


folder_path = Path('./static')
if not folder_path.exists():
    folder_path.mkdir()


class GoogleCloud:

    def __init__(self,
                 credential_info=GC_CREDENTIAL_INFO,
                 bucket_id: str = GC_BUCKET_ID,
                 endpoint_id: str = GC_ENDPOINT_ID):
        credentials = service_account.Credentials.from_service_account_info(credential_info)
        storage_client = storage.Client(credential_info['project_id'], credentials=credentials)
        self.bucket = storage_client.bucket(bucket_id)
        self.endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id, project=credential_info['project_id'],
                                            credentials=credentials)

    def _create_blob(self, file_path: str | Path):
        """
        Creates blob for file operations with file on GC.
        """
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()
        blob = self.bucket.blob(file_path)
        return blob

    def download_file(self, file_path: str | Path, downloaded_file_path: str | Path) -> bool:
        """
        Downloads file from GC.
        """
        if isinstance(downloaded_file_path, Path):
            downloaded_file_path = downloaded_file_path.as_posix()
        blob = self._create_blob(file_path)
        try:
            blob.download_to_filename(downloaded_file_path)
        except:
            return False
        return True

    def upload_file(self, file_path: str | Path, uploading_file_path: str | Path) -> bool:
        """
        Uploads file to GC.
        """
        if isinstance(uploading_file_path, Path):
            uploading_file_path = uploading_file_path.as_posix()
        blob = self._create_blob(file_path)
        try:
            blob.upload_from_filename(uploading_file_path)
        except:
            return False
        return True


class VideoData(bytes):
    """
    Video static.
    """


class VideoInfo:
    """
    Video info:
    frame_width [int] -
    frame_height [int] -
    frame_rate [float] -
    frames_number [int] -
    fourcc [float] -
    duration [timedelta] -
    duration_str [str] -
    """

    def __init__(self, file_name: str):
        capture = cv2.VideoCapture(file_name)
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
        self.frames_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        capture.release()
        self.duration = timedelta(seconds=self.frames_number / self.frame_rate)
        self.duration_str = f'{self.duration.seconds // 60:02d}:{self.duration.seconds % 60:02d}'

    def __str__(self):
        return (f'{self.frame_width=}, {self.frame_height=}, {self.frame_rate=}, {self.frames_number=}, '
                f'{self.fourcc=}, {self.duration=}, {self.duration_str=}')

    def __repr__(self):
        return self.__str__()


class Video:

    _temp_file_path = 'temp.mp4'

    def _compute_md5(self, algorithm='md5') -> str:
        """Computes the hash of video static using the specified algorithm."""
        hash_func = hashlib.new(algorithm)
        hash_func.update(self._data)
        return hash_func.hexdigest()

    def _save(self):
        with open(self.file_name, mode='wb') as video:
            video.write(self._data)

    def __init__(self, file_name: str | Path, data: VideoData):
        if isinstance(file_name, str):
            file_name = Path(file_name)
        self.file_name = f'video{file_name.suffix}'
        self._data = data
        self._save()
        self.info = VideoInfo(self.file_name)
        name = file_name.stem
        md5 = self._compute_md5()
        self.id = f'{name}_{md5}'
        self._video_capture = cv2.VideoCapture(self.file_name)

    def get_frame(self, frame: int):
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, data = self._video_capture.read()
        if not ret:
            return
        return data

    def save_screenshot(self, frame: int, file_path: str | Path):
        data = self.get_frame(frame)
        image = Image.fromarray(data)
        st.write(f'{file_path=}')
        image.save(file_path)

    def save_fragment(self, start_frame: int, frames_number: int, file_path: str | Path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self._temp_file_path, fourcc, self.info.frame_rate,
            (self.info.frame_width, self.info.frame_height))
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(frames_number):
            _, frame = self._video_capture.read()
            video_writer.write(frame)
        video_writer.release()
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()
        args = f"ffmpeg -y -i ./{self._temp_file_path} -c:v libx264 ./{file_path}".split(" ")
        subprocess.call(args=args)
        Path(self._temp_file_path).unlink()


class Data:
    _index_name: str
    _columns: list[str]
    data: pd.DataFrame
    hash: int

    def __init__(self):
        self._create()

    def _create(self):
        self.data = pd.DataFrame(columns=self._columns)
        self.data.index.name = self._index_name
        self._update_hash()

    def _update_hash(self):
        self.hash = pd.util.hash_pandas_object(self.data)


class Storable(Data):

    _file_path: str

    def __init__(self, gc: GoogleCloud, gcs_folder_path: str, gcs_file_name: str):
        super().__init__()
        self._gc = gc
        if isinstance(gcs_folder_path, str):
            gcs_folder_path = Path(gcs_folder_path)
        self._gcs_file_path = gcs_folder_path / gcs_file_name

    def download_data_from_gcs(self, length: int | None = None) -> bool:
        if not self._gc.download_file(self._gcs_file_path, self._file_path):
            return False
        with open(self._file_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            return False
        if list(data.keys()) != self._columns:
            return False
        if length is not None:
            if len(data[list(data.keys())[0]]) != length:
                return False
        self.data = pd.DataFrame.from_dict(data)
        self.data.index.name = self._index_name
        self._update_hash()
        return True

    def upload_data_to_gcs(self) -> bool:
        data = self.data.to_dict()
        with open(self._file_path, 'wb') as f:
            pickle.dump(data, f)
        return self._gc.upload_file(self._gcs_file_path, self._file_path)


class Intensities(Storable):
    """
    Emotion intensities in processed video
    """

    _file_path = 'static/intensities.dat'
    _index_name = 'step'
    _columns = ['intensity', 'time']

    def __init__(self,
                 video: Video,
                 gc: GoogleCloud, gcs_folder_path: str | Path = GC_INTENSITIES_PATH,
                 min_step: float = MIN_STEP, max_points_number: int = MAX_POINTS_NUMBER,
                 haar_file_name: str = HAAR_FILE):
        super().__init__(gc, gcs_folder_path, video.id + '.dat')
        self._video = video
        self._min_step_time = min_step
        self._max_points_number = max_points_number
        self._haar_file_name = haar_file_name
        self._faces_number = 0

        # Calculating maximal number of points (steps)
        min_step_frames = math.ceil(self._min_step_time * self._video.info.frame_rate)
        self.points_number = math.ceil(self._video.info.frames_number / min_step_frames)
        if self.points_number > self._max_points_number:
            self.points_number = self._max_points_number

        # Calculating number of frames between neighbour points (per step)
        self.step_frames = math.ceil(self._video.info.frames_number / self.points_number)

        # Calculating time between neighbour points (per step)
        self.step_time = self.step_frames / self._video.info.frame_rate

        # Calculating number of points
        self.points_number = math.ceil(self._video.info.frames_number / self.step_frames)

        # Downloading static from GCS
        if super().download_data_from_gcs(length=self.points_number):
            return

        # Recognizing emotion intensities
        self._recognize_intensities()
        self._update_hash()

        # Uploading static to GCS
        super().upload_data_to_gcs()

    def _recognize_intensities(self):
        """Recognize intensity (arousal) of the intensities in each video frame."""

        def _extract_face(frame, detector):
            """Extracts a face from the frame."""
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = detector.detectMultiScale(gray_frame, SCALE_FACTOR, MIN_NEIGHBORS, 0,
                                                   (MIN_FACE_SIZE, MIN_FACE_SIZE))
            if len(face_rects) == 0:
                return
            x = y = w = h = 0
            for face_rect in face_rects:
                if face_rect[2] > w and face_rect[3] > h:
                    x, y, w, h = face_rect
            face = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            return np.asarray(face).tolist()

        # Creating face detector
        haar_file = (Path(cv2.__file__).parent / 'static' / self._haar_file_name).as_posix()
        face_detector = cv2.CascadeClassifier(haar_file)

        # Creating video capture
        video_capture = cv2.VideoCapture(self._video.file_name)

        # Showing progress bar
        empty = st.empty()
        progress_bar = empty.progress(0.0)
        start_time = datetime.now()

        intensity = 0.
        values = []
        fails = []
        for step_index in range(self.points_number):
            # Retrieving video frame at point
            frame_index = step_index * self.step_frames
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, image = video_capture.read()
            if ret:
                # Extracting face at frame
                face_image = _extract_face(image, face_detector)
                # Recognizing emotion intensity
                if face_image is not None:
                    self._faces_number += 1
                    intensity = self._gc.endpoint.predict(
                        instances=[face_image], use_dedicated_endpoint=True).predictions[0][-1]
            else:
                fails.append(step_index)

            # Adding to result list
            values.append(intensity)

            # Updating progress bar
            current_time = datetime.now()
            elapsed_time = current_time - start_time
            elapsed_time_str = str(elapsed_time).split('.')[0]
            mean_speed = elapsed_time / (step_index + 1)
            left_time = mean_speed * (self.points_number - step_index - 1)
            left_time_str = str(left_time).split('.')[0]
            percent = (step_index + 1) / self.points_number
            progress_bar.progress(
               percent,
                f'{percent:.0%} [Elapsed time: {elapsed_time_str}, Left time: {left_time_str}]'
            )

        # Releasing video capture
        video_capture.release()

        # Hiding progress bar
        empty.empty()

        # Creating static
        self.data['intensity'] = values
        today = datetime.today()
        start_date = datetime(today.year, today.month, today.day)
        end_date = start_date + timedelta(seconds=self.step_time * (self.points_number - 1))
        self.data['time'] = pd.date_range(start=start_date, end=end_date, periods=self.points_number)


class HyperParams(Storable):
    """
    Hyperparameters for searching fragments to trailer creation
    """

    _file_path = 'static/hyperparams.dat'
    _columns = ['low_limit', 'high_limit',
                'init_low_bound', 'init_high_bound',
                'low_bound', 'high_bound', 'step']
    _index_name = 'name'
    _index = pd.Index(
        [
            'emotion_intensity',
            'fragment_duration',
            'min_time_between_fragments'
         ]
    )

    def __init__(self, video: Video, intensities: Intensities,
                 gc: GoogleCloud, gcs_folder_path: str | Path = GC_HYPERPARAMS_PATH):
        super().__init__(gc, gcs_folder_path, video.id + '.dat')

        # Downloading static from GCS
        if super().download_data_from_gcs(length=len(self._index)):
            return

        # Failed to download static from GCS

        # Creating default static
        intensity_limits = (intensities.data['intensity'].min(), intensities.data['intensity'].max())
        intensity_bounds = (max(intensity_limits[0], min(intensity_limits[1], 0.)), intensity_limits[1])
        intensity_step = min(intensity_limits[1] - intensity_limits[0], 0.01)
        duration_limits = (min(video.info.duration.total_seconds(), 1.), min(video.info.duration.total_seconds(), 15.))
        duration_bounds = (duration_limits[0], duration_limits[1])
        duration_step = min(duration_limits[1] - duration_limits[0], 0.1)
        min_time_limits = (min(video.info.duration.total_seconds(), 1.), min(video.info.duration.total_seconds(), 15.))
        min_time_bounds = (sum(min_time_limits)/2, None)
        min_time_step = min(min_time_limits[1] - min_time_limits[0], 0.1)
        data = [
            [
                *intensity_limits,
                *intensity_bounds,
                *intensity_bounds,
                intensity_step
            ],
            [
                *duration_limits,
                *duration_bounds,
                *duration_bounds,
                duration_step
            ],
            [
                *min_time_limits,
                *min_time_bounds,
                *min_time_bounds,
                min_time_step
            ],
        ]
        self.data[self._columns] = data
        self.data.index = self._index
        self._update_hash()

        # Uploading static to GCS
        super().upload_data_to_gcs()

    def get_limits(self, name: str) -> (float, float):
        low, high = self.data.loc[name, ['low_limit', 'high_limit']]
        return low, high

    def get_init_bounds_and_step(self, name: str) -> (float, float):
        low, high, step = self.data.loc[name, ['init_low_bound', 'init_high_bound', 'step']]
        return low, high, step

    def get_bounds(self, name: str) -> (float, float | None):
        low, high = self.data.loc[name, ['low_bound', 'high_bound']]
        return low, high

    def set_bounds(self, name: str, low: float, high: float | None):
        self.data.loc[name, ['low_bound', 'high_bound']] = low, high
        self._update_hash()
        super().upload_data_to_gcs()


class Fragments(Storable):

    _file_path = 'static/fragments.dat'
    _columns = [
        'start_step', 'peak_step', 'end_step', 'steps',
        'start', 'peak', 'end', 'time',
        'start_intensity', 'peak_intensity', 'end_intensity',
    ]
    _index_name = 'fragment'

    def __init__(self, video: Video, intensities: Intensities, hyperparams: HyperParams,
                 gc: GoogleCloud, gcs_folder_path: str | Path = GC_FRAGMENTS_PATH):
        super().__init__(gc, gcs_folder_path, video.id + '.dat')
        self._video = video
        self._intensities = intensities
        self._hyperparams = hyperparams

        # Downloading static from GCS
        if super().download_data_from_gcs():
            return

        # Failed to download static from GCS

        # Searching fragments
        # self.find_fragments()

        # Saving static to GCS
        # super().upload_data_to_gcs()

    def find_fragments(self):
        # Searching intensity peaks
        peaks, _ = find_peaks(
            self._intensities.data['intensity']
        )
        if len(peaks) == 0:
            return
        super()._create()
        self.data['peak_step'] = peaks
        self.data['peak_intensity'] = self._intensities.data.loc[self.data['peak_step'], 'intensity'].reset_index(drop=True)

        intensity_low_bound, intensity_high_bound = self._hyperparams.get_bounds('emotion_intensity')

        # Excluding fragments which peak intensity is out of bounds
        self.data = self.data.loc[
            self.data['peak_intensity'].between(intensity_low_bound, intensity_high_bound, inclusive='both')
        ]
        self.data.reset_index(drop=True, inplace=True)
        self.data.index.name = self._index_name
        if self.data.shape[0] == 0:
            return

        # Looking for fragment bounds
        self.data['prev_peak_step'] = -1, *self.data['peak_step'].iloc[:-1]
        self.data['next_peak_step'] = *self.data['peak_step'].iloc[1:], *[self._intensities.data.index[-1] + 1]
        valid_intensities = self._intensities.data['intensity'].between(
            intensity_low_bound, intensity_high_bound, inclusive='both')
        for fragment, (prev_step, step, next_step) in self.data[
            ['prev_peak_step', 'peak_step', 'next_peak_step']].iterrows():
            left_part = valid_intensities.loc[prev_step + 1: step - 1].sort_index(ascending=False)
            if sum(left_part) == 0:
                start = step
            elif sum(left_part) == len(left_part):
                start = left_part.index[0]
            else:
                start = left_part.idxmin() + 1
            right_part = valid_intensities.loc[step + 1: next_step - 1]
            if sum(right_part) == 0:
                end = step
            elif sum(right_part) == len(right_part):
                end = right_part.index[-1]
            else:
                end = right_part.idxmin() - 1
            self.data.loc[fragment, ['start_step', 'end_step']] = start, end

        # Union overlapped fragments
        self.data['next_start_step'] = *self.data['start_step'].iloc[1:], self._intensities.points_number + 1
        while True:
            if self.data.shape[0] <= 1:
                break
            overlapped_fragment_indices = self.data.loc[self.data['end_step'] >= self.data['next_start_step']].index
            if overlapped_fragment_indices.empty:
                break
            first_index = overlapped_fragment_indices[0]
            second_index = first_index + 1
            self.data.loc[first_index, ['end_step', 'next_peak_step', 'next_start_step']] = (
                self.data.loc[second_index, ['end_step', 'next_peak_step', 'next_start_step']]
            )
            if self.data.at[first_index, 'peak_intensity'] < self.data.at[second_index, 'peak_intensity']:
                self.data.loc[first_index, ['peak_step', 'peak_intensity']] = (
                    self.data.loc[second_index, ['peak_step', 'peak_intensity']]
                )
            self.data.drop(index=second_index, inplace=True)
            self.data.reset_index(drop=True, inplace=True, names=self._index_name)
            self.data.index.name = self._index_name

        # Calculate fragment steps
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1

        duration_low_bound, duration_high_bound = self._hyperparams.get_bounds('fragment_duration')

        # Excluding too short fragments
        min_steps = math.ceil(duration_low_bound / self._intensities.step_time)
        short_fragment_indices = self.data.loc[self.data['steps'] < min_steps].index
        if not short_fragment_indices.empty:
            self.data.drop(short_fragment_indices, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            self.data.index.name = self._index_name

        # Cutting too long fragments
        max_steps = math.floor(duration_high_bound / self._intensities.step_time)
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1
        long_fragments = self.data.loc[self.data['steps'] > max_steps, ['start_step', 'peak_step', 'end_step', 'steps']]
        if not long_fragments.empty:
            for index, (start_step, peak_step, end_step, steps) in long_fragments.iterrows():
                left_steps = peak_step - start_step + 1
                right_steps = steps - left_steps
                extra_steps = steps - max_steps
                left_extra_steps = math.ceil(extra_steps * left_steps / steps)
                right_extra_steps = extra_steps - left_extra_steps
                start_step += left_extra_steps
                end_step -= right_extra_steps
                steps -= extra_steps
                self.data.loc[index, ['start_step', 'end_step', 'steps']] = start_step, end_step, steps

        # Thinning fragments
        min_time, _ = self._hyperparams.get_bounds('min_time_between_fragments')
        min_steps = math.ceil(min_time / self._intensities.step_time)
        while True:
            self.data['next_start_step'] = (
                *self.data['start_step'].iloc[1:], self._intensities.points_number + min_steps)
            self.data['steps_to_next'] = self.data['next_start_step'] - self.data['end_step']
            if self.data.shape[0] <= 1:
                break
            closed_fragments = self.data.loc[self.data['steps_to_next'] < min_steps].index
            if closed_fragments.empty:
                break
            first_index = closed_fragments[0]
            second_index = first_index + 1
            if self.data.at[first_index, 'peak_intensity'] < self.data.at[second_index, 'peak_intensity']:
                dropping_index = first_index
            else:
                dropping_index = second_index
            self.data.drop(index=dropping_index, inplace=True)
            self.data.reset_index(drop=True, inplace=True, names=self._index_name)
            self.data.index.name = self._index_name

        # Retrieving intensities at bounds
        self.data['start_intensity'] = self._intensities.data.loc[
            self.data['start_step'], 'intensity'].reset_index(drop=True)
        self.data['end_intensity'] = self._intensities.data.loc[
            self.data['end_step'], 'intensity'].reset_index(drop=True)

        # Retrieving time at peaks and bounds
        self.data['start'] = self._intensities.data.loc[
            self.data['start_step'], 'time'].reset_index(drop=True)
        self.data['peak'] = self._intensities.data.loc[
            self.data['peak_step'], 'time'].reset_index(drop=True)
        self.data['end'] = self._intensities.data.loc[
            self.data['end_step'], 'time'].reset_index(drop=True)
        self.data['end'] += pd.to_timedelta(self._intensities.step_time, unit='s')
        self.data['time'] = self.data['end'] - self.data['start']

        # Remove helper fields
        self.data = self.data[self._columns]
        self.data = self.data.convert_dtypes()

        # Updating hash
        self._update_hash()

        # Saving static to GCS
        super().upload_data_to_gcs()


class Trailer(Storable):

    _file_path = 'static/fragments.dat'
    _columns = [
        'screenshot_frame', 'screenshot_file_path', 'screenshot_url',
        'fragment_start_frame', 'fragment_frames_number',
        # 'fragment_file_path', 'fragment_url',
        'selected'
    ]
    _index_name = 'fragment'
    trailer_file_name = 'trailer.mp4'

    def __init__(self,
                 video: Video, intensities: Intensities, fragments: Fragments,
                 gc: GoogleCloud, gcs_folder_path: str | Path = GC_TRAILER_PATH):
        super().__init__(gc, gcs_folder_path, video.id + '.dat')
        self._video = video
        self._intensities = intensities
        self._fragments = fragments

        # Downloading static from GCS
        if super().download_data_from_gcs(length=self._fragments.data.shape[0]):
            # Saving screenshots and fragments
            self._save_screenshots()
            # self._save_fragments()
            return

        # Creating static
        self.create_data()

    def _save_screenshots(self):
        for _, (frame, file_path) in self.data[
            ['screenshot_frame', 'screenshot_file_path']].iterrows():
            self._video.save_screenshot(frame, file_path)

    # def _save_fragments(self):
    #     for _, (start_frame, frames_number, file_path) in self.static[
    #         ['fragment_start_frame', 'fragment_frames_number', 'fragment_file_path']].iterrows():
    #         self._video.save_fragment(start_frame, frames_number, file_path)

    def create_data(self):
        # Creating static
        super()._create()
        self.data['screenshot_frame'] = self._fragments.data['peak_step'] * self._intensities.step_frames
        self.data['screenshot_file_path'] = self._fragments.data.index.map(
            lambda fragment: f'static/fragment_{fragment}.jpg'
        )
        self.data['screenshot_url'] = self.data['screenshot_file_path'].map(
            lambda file_path: f'{st.secrets["server"]["url"]}/app/{file_path}')
        self.data['fragment_start_frame'] = self._fragments.data['start_step'] * self._intensities.step_frames
        self.data['fragment_frames_number'] = self._fragments.data['steps'] * self._intensities.step_frames
        # self.static['fragment_file_path'] = self.static.index.map(
        #     lambda fragment: f'static/fragment_{fragment}.mp4'
        # )
        # self.static['fragment_url'] = self.static['fragment_file_path'].map(
        #     lambda file_path: f'http://localhost:{HTTP_SERVER_PORT}/{file_path}')
        self.data['selected'] = False
        st.write(self.data)
        self._update_hash()

        # Saving static to GCS
        super().upload_data_to_gcs()

        # Saving screenshots and fragments
        self._save_screenshots()
        # self._save_fragments()

    def save(self):
        """Saves the trailer to a file."""
        temp_name = 'temp.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        trailer_writer = cv2.VideoWriter(
            temp_name, fourcc, self._video.info.frame_rate,
            (self._video.info.frame_width, self._video.info.frame_height))
        data = self.data.loc[self.data['selected'], ['fragment_start_frame', 'fragment_frames_number']]
        trailer_frames_number = data['fragment_frames_number'].sum()
        frames_number = 0
        empty = st.empty()
        progress_bar = empty.progress(0.0)
        start_time = datetime.now()
        for fragment, (fragment_start_frame, fragment_frames_number) in data.iterrows():
            for frame in range(fragment_start_frame, fragment_start_frame + fragment_frames_number):
                trailer_writer.write(self._video.get_frame(frame))
                frames_number += 1

                # Updating progress bar
                current_time = datetime.now()
                elapsed_time = current_time - start_time
                elapsed_time_str = str(elapsed_time).split('.')[0]
                mean_speed = elapsed_time / frames_number
                left_time = mean_speed * (trailer_frames_number - frames_number)
                left_time_str = str(left_time).split('.')[0]
                percent = frames_number / trailer_frames_number
                progress_bar.progress(
                    percent,
                    f'{percent:.0%} [Elapsed time: {elapsed_time_str}, Left time: {left_time_str}]'
                )
        empty.empty()
        trailer_writer.release()
        args = f"ffmpeg -y -i ./{temp_name} -c:v libx264 ./{self.trailer_file_name}".split(" ")
        subprocess.call(args=args)
        Path(temp_name).unlink()


def get_help():
    with open('help.md', 'rt') as f:
        help = f.read()
    return help


def upload_video() -> [str, VideoData]:
    """Uploads video file."""
    uploaded_file = st.file_uploader(
        # 'Video file uploading',
        '',
        ['mp4', 'avi', 'mov'],
        # help='Drag and Click the **Browser files** button and select a video file from your storage',
        label_visibility='visible',
        on_change=on_video_upload
    )
    return uploaded_file


def set_hyperparams(hyperparams: HyperParams):
    """Sets up hyperparameters for fragments searching."""

    hyperparam_names = [
        'emotion_intensity',
        'fragment_duration',
        'min_time_between_fragments'
    ]

    hyperparam_titles = [
        'Emotion intensity',
        'Fragment duration [s]',
        'Min. time between fragments [s]'
    ]

    with st.container(border=True):
        for name, title in zip(hyperparam_names, hyperparam_titles):
            low_limit, high_limit = hyperparams.get_limits(name)
            init_low_bound, init_high_bound, step = hyperparams.get_init_bounds_and_step(name)
            low_bound, high_bound = hyperparams.get_bounds(name)
            if np.isnan(high_bound) or high_bound is None:
                low_bound = st.slider(
                    title, min_value=low_limit, max_value=high_limit,
                    value=init_low_bound, step=step,
                    on_change=on_hyperparam_changed)
            else:
                low_bound, high_bound = st.slider(
                    title, min_value=low_limit, max_value=high_limit,
                    value=(init_low_bound, init_high_bound), step=step,
                    on_change=on_hyperparam_changed)
            hyperparams.set_bounds(name, low_bound, high_bound)


def data_hash_func(data: Data) -> int:
    return data.hash


@st.cache_resource(show_spinner=False, hash_funcs={Intensities: data_hash_func, HyperParams: data_hash_func})
def find_fragments(intensities: Intensities, hyperparams: HyperParams, _fragments: Fragments):
    with st.spinner('Searching fragments...'):
        _fragments.find_fragments()


@st.cache_data(show_spinner=False,
               hash_funcs={Intensities: data_hash_func, Fragments: data_hash_func, HyperParams: data_hash_func})
def plot_chart(intensities: Intensities, fragments: Fragments, hyperparams: HyperParams):
    with st.spinner('Plotting chart...'):
        line_color = 'red'
        area_color = '#FFEDED'

        chart = alt.Chart(intensities.data).encode(
            x=alt.X(title='Time', field='time', type='temporal', axis=alt.Axis(format='%M:%S')),
            y=alt.Y(title='Intensity', field='intensity', type='quantitative')
        ).mark_line(
            color=line_color,
            strokeWidth=1,
            interpolate='monotone'
        )
        x_min, x_max = intensities.data['time'].iloc[[0, -1]]
        y_min, y_max = hyperparams.get_bounds('emotion_intensity')
        df = pd.DataFrame.from_dict(
            {
                'time': [x_min, x_max],
                'intensity': [y_min, y_min],
            }
        )
        intensity_low_bound_chart = alt.Chart(df).encode(
                x=alt.X(title='Time', field='time', type='temporal', axis=alt.Axis(format='%M:%S')),
                y=alt.Y(title='Intensity', field='intensity', type='quantitative')
            ).mark_line(color=line_color, strokeWidth=1, strokeDash=[8, 8])
        chart += intensity_low_bound_chart
        df = pd.DataFrame.from_dict(
            {
                'time': [x_min, x_max],
                'intensity': [y_max, y_max],
            }
        )
        intensity_high_bound_chart = alt.Chart(df).encode(
                x=alt.X(title='Time', field='time', type='temporal', axis=alt.Axis(format='%M:%S')),
                y=alt.Y(title='Intensity', field='intensity', type='quantitative')
            ).mark_line(color=line_color, strokeWidth=1, strokeDash=[8, 8])
        chart += intensity_high_bound_chart
        for fragment, (start_step, end_step, start, peak, end,
                       start_intensity, peak_intensity, end_intensity
                       ) in fragments.data[
            ['start_step', 'end_step', 'start', 'peak', 'end',
             'start_intensity', 'peak_intensity', 'end_intensity']
        ].iterrows():
            df = intensities.data.loc[start_step: end_step]
            fragment_chart = alt.Chart(df).encode(
                x=alt.X(title='Time', field='time', type='temporal', axis=alt.Axis(format='%M:%S')),
                y=alt.Y(title='Intensity', field='intensity', type='quantitative')
            ).mark_area(
                line={'color': line_color, 'strokeWidth': 0},
                color=area_color,
            )
            chart += fragment_chart
            df = pd.DataFrame.from_dict(
                {'time': [peak], 'intensity': [peak_intensity]}
            )
            peak_chart = alt.Chart(df).encode(
                x=alt.X(title='Time', field='time', type='temporal', axis=alt.Axis(format='%M:%S')),
                y=alt.Y(title='Intensity', field='intensity', type='quantitative')
            ).mark_circle(color=line_color)
            chart += peak_chart

        with st.expander('Intensity graph'):
            st.altair_chart(chart, use_container_width=True, key='intensity_chart')


def display_fragments_table(intensities: Intensities, fragments: Fragments, trailer: Trailer) -> [int]:
    df = fragments.data[['start', 'peak', 'end']].copy()
    df['time'] = fragments.data['time'].dt.total_seconds()
    df['peak_intensity'] = fragments.data['peak_intensity']
    df['screenshot_url'] = trailer.data['screenshot_url']
    df['intensity_chart'] = None
    for fragment, (start_step, end_step) in fragments.data[['start_step', 'end_step']].iterrows():
        df.at[fragment, 'intensity_chart'] = intensities.data.loc[start_step: end_step, 'intensity'].to_list()

    column_config = {
        '_index': st.column_config.NumberColumn('#'),
        'screenshot_url': st.column_config.ImageColumn('Screenshot'),
        'start': st.column_config.TimeColumn('Start', format='m:ss.SSS'),
        'peak': st.column_config.TimeColumn('Peak', format='m:ss.SSS'),
        'end': st.column_config.TimeColumn('End', format='m:ss.SSS'),
        'time': st.column_config.NumberColumn('Time [s]', format='%.2f'),
        'peak_intensity': st.column_config.NumberColumn('Peak intensity', format='%.3f'),
        'intensity_chart': st.column_config.AreaChartColumn('Intensity', y_min=0., y_max=1.)
    }
    event = st.dataframe(df, use_container_width=True, column_config=column_config, key='fragment_dataframe',
                         on_select='rerun', selection_mode='multi-row', hide_index=True)
    selected_rows = event.selection.rows
    total_time = df.loc[selected_rows, 'time'].sum()

    st.markdown(f'Total time: {total_time:.2f}s')

    trailer.data['selected'] = False
    if len(selected_rows) > 0:
        trailer.data.loc[selected_rows, 'selected'] = True

    return selected_rows


def play_fragment(fragment: int, fragments: Fragments, intensities: Intensities, video: Video):
    start_step, end_step = fragments.data.loc[fragment, ['start_step', 'end_step']]
    start_time = start_step * intensities.step_time
    end_time = (end_step + 1) * intensities.step_time
    st.video(video.file_name, start_time=start_time, end_time=end_time)


def download_trailer(trailer_name: str):
    """The trailer file downloading."""
    with open(trailer_name, mode='rb') as trailer_file:
        st.download_button('Download the trailer...', trailer_file, trailer_name,
                           'video' + Path(trailer_name).suffix[1:], icon=':material/download:')


def show_title_and_help():
    st.markdown('<h2 style="text-align: center;">Trailer creator</h2>', unsafe_allow_html=True)
    with st.expander('See brief help...', icon=':material/help:'):
        st.markdown(get_help())


def on_video_upload():
    st.session_state['video_uploader_changed'] = True


def on_hyperparam_changed():
    st.session_state['hyperparams_changed'] = True


def main():
    show_title_and_help()
    st.markdown('')
    # if 'http_server_is_running' not in st.session_state:
    #     with st.spinner('Running HTTP-server...'):
    #         subprocess.Popen(f'python -m http.server {HTTP_SERVER_PORT}')
    #     st.session_state['http_server_is_running'] = True
    if 'google_cloud' in st.session_state:
        google_cloud = st.session_state['google_cloud']
    else:
        with st.spinner('Connecting to Google Cloud...'):
            google_cloud = GoogleCloud()
    video_col_1, fragment_col_1, trailer_col_1 = st.columns([30, 40, 30], gap='large')
    # video_col_2, fragment_col_2, trailer_col_2 = st.columns([30, 40, 30], gap='large')
    with video_col_1:
        st.markdown('<h3 style="text-align: center;">Video</h3>', unsafe_allow_html=True)
    with fragment_col_1:
        st.markdown('<h3 style="text-align: center;">Fragments</h3>', unsafe_allow_html=True)
        # st.markdown('<p style="font-size:14px"><br></p>', unsafe_allow_html=True)
    with trailer_col_1:
        st.markdown('<h3 style="text-align: center;">Trailer</h3>', unsafe_allow_html=True)
        # st.markdown('<p style="font-size:14px"><br></p>', unsafe_allow_html=True)
    with video_col_1:
        with st.container(border=True):
            uploaded_video = upload_video()

    if uploaded_video is None:
        with fragment_col_1:
            st.info('Please upload video', icon=":material/info:")
        with trailer_col_1:
            st.info('Please upload video', icon=":material/info:")
        return

    # Creating components
    if st.session_state['video_uploader_changed']:
        st.session_state['video_uploader_changed'] = False
        st.session_state['hyperparams_changed'] = True
        video = Video(uploaded_video.name, uploaded_video.read())
        st.session_state['video'] = video
        with video_col_1:
            with st.spinner('Recognizing emotions intensity...'):
                intensities = Intensities(video, google_cloud)
            st.session_state['intensities'] = intensities
        hyperparams = HyperParams(video, intensities, google_cloud)
        st.session_state['hyperparams'] = hyperparams
        fragments = Fragments(video, intensities, hyperparams, google_cloud)
        st.session_state['fragments'] = fragments
        trailer = Trailer(video, intensities, fragments, google_cloud)
        st.session_state['trailer'] = trailer
    else:
        video = st.session_state['video']
        intensities = st.session_state['intensities']
        hyperparams = st.session_state['hyperparams']
        fragments = st.session_state['fragments']
        trailer = st.session_state['trailer']

    with video_col_1:
        st.video(video.file_name)

    with fragment_col_1:
        set_hyperparams(hyperparams)
        find_fragments(intensities, hyperparams, fragments)
        plot_chart(intensities, fragments, hyperparams)
        trailer.create_data()
        selected_fragments = display_fragments_table(intensities, fragments, trailer)

    with trailer_col_1:
        save_trailer = st.button('Create trailer...', icon=':material/movie:',
                                 disabled=len(selected_fragments) == 0)
        if save_trailer:
            with st.spinner('Creating trailer...'):
                trailer.save()
            st.video(trailer.trailer_file_name)
            download_trailer(trailer.trailer_file_name)


if __name__ == '__main__':
    main()
