import hashlib
import pickle
import subprocess
from datetime import timedelta, datetime, date
from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.cloud import storage
from google.cloud import aiplatform
from google.oauth2 import service_account
from scipy.signal import find_peaks

#
# from emotion_recognition import FaceEmotionRecognitionNet, EMOTIONS

st.set_page_config(page_title='Creating trailers', page_icon=':film_frames:',
                   layout="wide", initial_sidebar_state="expanded")

BACKGROUND_COLOR = 'white'
COLOR = 'black'

# Face detector settings
HAAR_FILE = 'haarcascade_frontalface_default.xml'
SCALE_FACTOR = 1.1  # Scale factor
MIN_NEIGHBORS = 3  # Min. neighbors number
MIN_FACE_SIZE = 64  # Min. face size

# Google Cloud Storage
GC_PROJECT_ID = st.secrets['gc-service-account']['project_id'] # Project id
GC_BUCKET_ID = st.secrets['gc-storage']['bucket_id'] # Bucket id
GC_LOCATION_ID = st.secrets['gc-aiplatform']['location_id'] # Location id
GC_ENDPOINT_ID = st.secrets['gc-aiplatform']['endpoint_id'] # Endpoint id
GC_AROUSALS_PATH = 'arousals'  # Processed video arousals folder path
GC_HYPERPARAMS_PATH = 'hyperparams'  # Processed video hyperparams folder path

# Local
AROUSALS_PATH = 'arousals.dat'  # Arousals file path
HYPERPARAMS_PATH = 'hyperparams.dat'  # Processed video hyperparams folder path


class VideoId:
    file_name: str = ''
    md5: str = ''
    id: str = ''


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
    frame_width: int = 0
    frame_height: int = 0
    frame_rate: float = 0.
    frames_number: int = 0
    fourcc: int = 0
    duration: timedelta = timedelta(0, 0, 0, 0, 0, 0, 0)
    duration_str: str = ''


class VideoData(bytes):
    """
    Video data.
    """


class HyperParams:
    """
    Hyperparameters for searching fragments to trailer creation:
    tma_window [float] - time window duration for averaging emotion intensity (arousal) [sec]
    min_arousal [float] - minimum value of the peak of emotions of a fragment
    min_prominence [float] - minimal increase in the intensity (arousal) of emotions of a fragment
    min_duration [float] - minimum fragment duration [sec]
    rel_height [float] - relative height of rise 0...1
    min_distance [float] - minimal time between fragments [sec].
    """
    tma_window: float = 1.0
    min_arousal: float = 0.5
    min_prominence: float = 0.25
    min_duration: float = 1.0
    rel_height: float = 0.5
    min_distance: float = 5.0

    def __str__(self):
        return (f'{self.tma_window=}, {self.min_arousal=}, {self.min_prominence=}, '
                f'{self.min_duration=}, {self.rel_height=}, {self.min_distance=}')

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(
            (self.tma_window, self.min_arousal, self.min_prominence,
             self.min_duration, self.rel_height, self.min_distance)
        )


def create_gc_credentials():
    connection_info = st.secrets["gc-service-account"]
    credentials = service_account.Credentials.from_service_account_info(connection_info)
    return credentials


def connect_to_gc_storage_bucket(
        project_id: str = GC_PROJECT_ID,
        bucket_id: str = GC_BUCKET_ID,
        credentials: service_account.Credentials = None
) -> storage.Bucket:
    storage_client = storage.Client(project_id, credentials=credentials)
    bucket = storage_client.bucket(bucket_id)
    return bucket


def init_aiplatform(project_id: str = GC_PROJECT_ID, location_id: str = GC_LOCATION_ID, credentials=None):
    aiplatform.init(project=project_id, location=location_id, credentials=credentials)


def create_gc_blob(bucket: storage.Bucket, file_path: str | PathLike):
    """
    Creates blob for file operations with file on GC.
    """
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    blob = bucket.blob(file_path)
    return blob


def download_file_from_gc(
        bucket: storage.Bucket, file_path: str | PathLike,
        downloaded_file_path: str | PathLike) -> bool:
    """
    Downloads file from GC.
    """
    if isinstance(downloaded_file_path, Path):
        downloaded_file_path = downloaded_file_path.as_posix()
    blob = create_gc_blob(bucket, file_path)
    try:
        blob.download_to_filename(downloaded_file_path)
    except Exception as e:
        print(e)
        return False
    return True


def upload_file_to_gc(bucket: storage.Bucket, file_path: str | PathLike, uploading_file_path: str | PathLike) -> bool:
    """
    Uploads file to GC.
    """
    if isinstance(uploading_file_path, Path):
        uploading_file_path = uploading_file_path.as_posix()
    blob = create_gc_blob(bucket, file_path)
    try:
        blob.upload_from_filename(uploading_file_path)
    except Exception as e:
        print(e)
        return False
    return True


# def create_emotion_recognizer(
#         gc_file_path: str | PathLike = GC_MODEL_PATH, file_path: str | PathLike = MODEL_PATH,
#         emotions=EMOTIONS) -> FaceEmotionRecognitionNet | None:
#     """Creates emotion recognition model."""
#     print(f'create_emotion_recognizer: {gc_file_path=}')
#     key = 'emotion_recognizer'
#     if key in st.session_state:
#         return st.session_state[key]
#     with st.spinner('Creating the emotion recognition model...'):
#         st.session_state[key] = None
#         try:
#             assert download_file_from_gc(gc_file_path, 'temp.zip')
#             with ZipFile('temp.zip') as zipfile:
#                 zipfile.extractall(file_path)
#             Path('temp.zip').unlink()
#             st.session_state[key] = FaceEmotionRecognitionNet(file_path, emotions)
#         except Exception as e:
#             st.write(e)
#             st.session_state[key] = None
#             st.error('Failed to create emotion recognition model', icon=':material/error:')
#     return st.session_state[key]


def create_face_detector(_haar_file_name: str = HAAR_FILE) -> cv2.CascadeClassifier:
    """Creates face detector."""
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + _haar_file_name)
    return face_detector


def create_emotion_recognizer_endpoint(
        gc_project_id: str = GC_PROJECT_ID, gc_endpoint_id: str = GC_ENDPOINT_ID) -> aiplatform.Endpoint | None:
    """Creates emotion recognition model."""
    key = 'emotion_recognizer_endpoint'
    if key in st.session_state:
        return st.session_state[key]
    with st.spinner('Creating the emotion recognition model endpoint...'):
        st.session_state[key] = None
        try:
            st.write('inited')
            endpoint = aiplatform.Endpoint(endpoint_name=gc_endpoint_id)
            st.write('endpoint')
            st.session_state[key] = endpoint
            st.write('session_state')
        except Exception as e:
            st.write(e)
            st.session_state[key] = None
            st.error('Failed to create emotion recognition model endpoint', icon=':material/error:')
    return st.session_state[key]


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


def compute_video_hash(data: VideoData, algorithm='md5') -> str:
    """Computes the hash of video data using the specified algorithm."""
    hash_func = hashlib.new(algorithm)
    hash_func.update(data)
    return hash_func.hexdigest()


def save_video(video_id: VideoId, _data: VideoData):
    """Saves video data to file"""
    for file in Path('.').iterdir():
        if file.suffix in ('.mp4', '.avi', '.mov'):
            file.unlink()
    with open(video_id.file_name, mode='wb') as video:
        video.write(_data)


def retrieve_video_info(video_id: VideoId) -> VideoInfo:
    """Extracts info about video data from the file."""
    video_info = VideoInfo()
    capture = cv2.VideoCapture(video_id.file_name)
    video_info.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_info.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info.frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
    video_info.frames_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_info.fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    video_info.duration = timedelta(seconds=(video_info.frames_number - 1) / video_info.frame_rate)
    video_info.duration_str = f'{video_info.duration.seconds // 60:02d}:{video_info.duration.seconds % 60:02d}'
    return video_info


# def recognize_video_emotions(video_id: VideoId, video_info: VideoInfo, face_detector: cv2.CascadeClassifier,
#                              emotion_recognizer: FaceEmotionRecognitionNet) -> [float]:
#     """Recognize intensity (arousal) of the emotions in each video frame."""
#
#     def _extract_face(frame, detector):
#         """Extracts a face from the frame."""
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_rects = detector.detectMultiScale(gray_frame, SCALE_FACTOR, MIN_NEIGHBORS, 0,
#                                                (MIN_FACE_SIZE, MIN_FACE_SIZE))
#         if len(face_rects) == 0:
#             return
#         x = y = w = h = 0
#         for face_rect in face_rects:
#             if face_rect[2] > w and face_rect[3] > h:
#                 x, y, w, h = face_rect
#         face = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
#         return face
#
#     video_capture = cv2.VideoCapture(video_id.file_name)
#     frame_index = 0
#     faces_number = 0
#     arousal = 0.0
#     arousals = []
#     empty = st.empty()
#     progress_bar = empty.progress(0.0)
#     while True:
#         ret, image = video_capture.read()
#         if not ret:
#             break
#         face_image = _extract_face(image, face_detector)
#         if face_image is not None:
#             faces_number += 1
#             arousal = emotion_recognizer.predict(face_image)[-1]
#         arousals.append(arousal)
#         frame_index += 1
#         progress_bar.progress(frame_index / video_info.frames_number)
#     video_capture.release()
#     empty.empty()
#     return arousals


def recognize_video_emotions(video_id: VideoId, video_info: VideoInfo, face_detector: cv2.CascadeClassifier,
                             emotion_recognizer_endpoint: aiplatform.Endpoint) -> [float]:
    """Recognize intensity (arousal) of the emotions in each video frame."""

    # def _extract_face(frame, detector):
    #     """Extracts a face from the frame."""
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     face_rects = detector.detectMultiScale(gray_frame, SCALE_FACTOR, MIN_NEIGHBORS, 0,
    #                                            (MIN_FACE_SIZE, MIN_FACE_SIZE))
    #     if len(face_rects) == 0:
    #         return
    #     x = y = w = h = 0
    #     for face_rect in face_rects:
    #         if face_rect[2] > w and face_rect[3] > h:
    #             x, y, w, h = face_rect
    #     face = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
    #     return np.asarray(face).tolist()

    # st.write(f'recognize_video_emotions: {video_id.id=}, {video_info.frames_number=}, {face_detector=},'
    #          f'{emotion_recognizer_endpoint=}')
    # video_capture = cv2.VideoCapture(video_id.file_name)
    frame_index = 0
    faces_number = 0
    arousal = 0.0
    arousals = []
    # empty = st.empty()
    iter_size = 10
    iter_number = video_info.frames_number / iter_size
    iter_index = 0
    # progress_bar = empty.progress(0.0)
    start_time = datetime.now()
    st.write(f'starting loop...')
    # while True:
    #     # ret, image = video_capture.read()
    #     st.write(f'frame read')
    #     if frame_index % 10 > 0:
    #         frame_index += 1
    #         continue
    #     if not ret:
    #         break
    #     st.write(f'extracting face...')
    #     # face_image = _extract_face(image, face_detector)
    #     st.write(f'face extracted')
    #     # if face_image is not None:
    #     #     faces_number += 1
    #         # st.write('predicting...')
    #         # arousal = emotion_recognizer_endpoint.predict(
    #         #     instances=[face_image], use_dedicated_endpoint=True, timeout=5).predictions[0][-1]
    #     #     arousal = 0.
    #     # arousals.append(arousal)
    #     frame_index += 1
    #     iter_index += 1
    #     current_time = datetime.now()
    #     elapsed_time = current_time - start_time
    #     elapsed_time_str = str(elapsed_time).split('.')[0]
    #     left_time = (elapsed_time / iter_index) * (iter_number - iter_index)
    #     left_time_str = str(left_time).split('.')[0]
    #     percent = iter_index / iter_number
    #     progress_bar.progress(
    #        percent,
    #         f'{percent:.0%} [Elapsed time: {elapsed_time_str}, Left time: {left_time_str}]'
    #     )
    # # video_capture.release()
    # empty.empty()
    return arousals


# def get_arousals(
#         video_id: VideoId, video_info: VideoInfo,
#         face_detector: cv2.CascadeClassifier, emotion_recognizer: FaceEmotionRecognitionNet,
#         gc_folder_path: str | PathLike = GC_AROUSALS_PATH, file_name: str | PathLike = AROUSALS_PATH) -> list[float]:
#     with st.spinner('Emotion recognition...'):
#         if isinstance(gc_folder_path, str):
#             gc_folder_path = Path(gc_folder_path)
#         gc_file_path = gc_folder_path / f'{video_id.id}.dat'
#         if download_file_from_gc(gc_file_path, file_name):
#             with open(file_name, 'rb') as f:
#                 arousals = pickle.load(f)
#             return arousals
#         arousals = recognize_video_emotions(video_id, video_info, face_detector, emotion_recognizer)
#         with open(file_name, 'wb') as f:
#             pickle.dump(arousals, f)
#         upload_file_to_gc(gc_file_path, file_name)
#     return arousals
#
#


def get_arousals(
        video_id: VideoId, video_info: VideoInfo,
        face_detector: cv2.CascadeClassifier, emotion_recognizer_endpoint: aiplatform.Endpoint,
        gc_folder_path: str | PathLike = GC_AROUSALS_PATH, file_name: str | PathLike = AROUSALS_PATH) -> list[float]:
    with st.spinner('Recognizing emotions intensity...'):
        st.write(f'{gc_folder_path=}')
        if isinstance(gc_folder_path, str):
            gc_folder_path = Path(gc_folder_path)
        gc_file_path = gc_folder_path / f'{video_id.id}.dat'
        st.write(f'{gc_file_path=}')
        # if download_file_from_gc(gc_file_path, file_name):
        #     with open(file_name, 'rb') as f:
        #         arousals = pickle.load(f)
        #     return arousals
        # arousals = recognize_video_emotions(video_id, video_info, face_detector, emotion_recognizer_endpoint)
        # with open(file_name, 'wb') as f:
        #     pickle.dump(arousals, f)
        # upload_file_to_gc(gc_file_path, file_name)
        arousals = [0, 0, 0]
    return arousals


def init_hyperparams(
        video_id: VideoId, gc_folder_path: str | PathLike = GC_HYPERPARAMS_PATH,
        file_name: str | PathLike = HYPERPARAMS_PATH) -> HyperParams:
    if isinstance(gc_folder_path, str):
        gc_folder_path = Path(gc_folder_path)
    gc_file_path = gc_folder_path / f'{video_id.id}.dat'
    if download_file_from_gc(gc_file_path, file_name):
        with open(file_name, 'rb') as f:
            hyperparams = pickle.load(f)
            # print(f'downloaded: {hyperparams}')
        return hyperparams
    return HyperParams()


def set_hyperparams(
        video_id: VideoId, initial_hyperparams: HyperParams,
        gc_folder_path: str | PathLike = GC_HYPERPARAMS_PATH,
        file_name: str | PathLike = HYPERPARAMS_PATH) -> HyperParams:
    """Sets up hyperparameters for fragments searching."""
    hyperparams = HyperParams()
    with st.sidebar:
        st.markdown('Intensity averaging')
        with st.container(border=True):
            hyperparams.tma_window = st.slider(
                'Time window size [sec]', 0.05, 2.0, initial_hyperparams.tma_window, 0.05,
                on_change=on_tma_window_changed)
        st.markdown('Fragment searching')
        with st.container(border=True):
            hyperparams.min_arousal = st.slider(
                'Min. peak', 0.0, 1.0, initial_hyperparams.min_arousal, 0.05)
            hyperparams.min_prominence = st.slider(
                'Min. prominence', 0.0, 0.5, initial_hyperparams.min_prominence, 0.05)
            hyperparams.rel_height = st.slider(
                'Min. relative height', 0.0, 1.0, initial_hyperparams.rel_height, 0.05)
        st.markdown('Fragment selecting')
        with st.container(border=True):
            hyperparams.min_duration = st.slider(
                'Min. duration [sec]', 0.0, 2.0, initial_hyperparams.min_duration, 0.05)
            hyperparams.min_distance = st.slider(
                'Min. distance [sec]', 0.5, 10.0, initial_hyperparams.min_distance, 0.5)
    if isinstance(gc_folder_path, str):
        gc_folder_path = Path(gc_folder_path)
    gc_file_path = gc_folder_path / f'{video_id.id}.dat'
    with open(file_name, 'wb') as f:
        pickle.dump(hyperparams, f)
    upload_file_to_gc(gc_file_path, file_name)
    return hyperparams


def fit_intensity_curve(
        arousals: [float], twa_window: float, video_info: VideoInfo) -> pd.DataFrame:
    # print(f'fit_intensity_curve: {twa_window=}')
    today = date.today()
    start_date = datetime(today.year, today.month, today.day)
    end_date = start_date + video_info.duration
    trend = pd.DataFrame()
    trend['time'] = pd.date_range(start_date, end_date, periods=video_info.frames_number)
    trend['arousal'] = arousals
    tma_window = int(twa_window * video_info.frame_rate)
    trend['arousal_sma'] = trend['arousal'] \
        .rolling(window=tma_window, min_periods=1, center=True).mean()
    trend['arousal_tma'] = trend['arousal_sma'] \
        .rolling(window=tma_window, min_periods=1, center=True).mean()
    return trend


def find_fragments(
        trend: pd.DataFrame, hyperparams: HyperParams, video_info: VideoInfo) -> pd.DataFrame:
    """Searching video fragments for the trailer."""
    # print(f'find_fragments: {hyperparams=}')
    with st.spinner('Searching fragments...'):
        peaks, properties = find_peaks(
            trend['arousal_tma'],
            height=hyperparams.min_arousal,
            distance=hyperparams.min_distance * video_info.frame_rate,
            prominence=hyperparams.min_prominence,
            width=hyperparams.min_duration * video_info.frame_rate,
            rel_height=hyperparams.rel_height,
        )
        fragments = pd.DataFrame()
        fragments['peak_frame'] = peaks
        fragments['peak_time'] = trend.loc[fragments['peak_frame'].to_list(), 'time'].to_list()
        fragments['peak_arousal'] = properties['peak_heights']
        fragments['start_frame'] = [int(left_ips) for left_ips in properties['left_ips']]
        fragments['start_time'] = trend.loc[fragments['start_frame'].to_list(), 'time'].to_list()
        fragments['start_arousal'] = trend.loc[fragments['start_frame'].to_list(), 'arousal_tma'].to_list()
        fragments['end_frame'] = [int(left_ips) for left_ips in properties['right_ips']]
        fragments['end_time'] = trend.loc[fragments['end_frame'].to_list(), 'time'].to_list()
        fragments['end_arousal'] = trend.loc[fragments['end_frame'].to_list(), 'arousal_tma'].to_list()

    return fragments


def plot_chart(trend: pd.DataFrame, fragments: pd.DataFrame):
    """Plots chart."""
    with st.spinner('Plotting chart...'):
        st.markdown('<h4 style="text-align: center;">Intensity dynamic</h4>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend['time'], y=trend['arousal'],
                mode='lines', name='Intensity',
                line={'width': 1, 'shape': 'spline', 'color': px.colors.qualitative.Prism[7]},
                opacity=0.5
            )
        )
        for fragment, (start_frame, end_frame, peak_time, start_time, end_time) in fragments[
            ['start_frame', 'end_frame', 'peak_time', 'start_time', 'end_time']
        ].iterrows():
            fig.add_trace(
                go.Scatter(
                    x=trend.loc[start_frame: end_frame, 'time'],
                    y=trend.loc[start_frame: end_frame, 'arousal_tma'],
                    mode='lines', name=None,
                    line={'width': 0, 'shape': 'spline', 'color': px.colors.qualitative.Prism[4]},
                    fill='toself',
                    opacity=0.5,
                    showlegend=False,
                )
            )
            # fig.add_vline(
            #     peak_time,
            #     line={'width': 1, 'dash': 'dash', 'color': px.colors.qualitative.Prism[4]}
            # )
            # fig.add_vline(
            #     start_time,
            #     line={'width': 1, 'dash': 'dash', 'color': px.colors.qualitative.Prism[4]}
            # )
            # fig.add_vline(
            #     end_time,
            #     line={'width': 1, 'dash': 'dash', 'color': px.colors.qualitative.Prism[4]}
            # )
        fig.add_trace(
            go.Scatter(
                x=trend['time'], y=trend['arousal_tma'],
                mode='lines', name='Average intensity',
                line={'width': 2, 'shape': 'spline', 'color': px.colors.qualitative.Prism[4]},
                opacity=0.9
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fragments['start_time'], y=fragments['start_arousal'],
                mode='markers', name='Fragment start',
                marker={
                    'symbol': 'line-nw-open',
                    'size': 6, 'color': px.colors.qualitative.Prism[4],
                    'line': {'width': 2, 'color': px.colors.qualitative.Prism[4]}
                },
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fragments['end_time'], y=fragments['end_arousal'],
                mode='markers', name='Fragment end',
                marker={
                    'symbol': 'line-ne-open',
                    'size': 6, 'color': px.colors.qualitative.Prism[4],
                    'line': {'width': 2, 'color': px.colors.qualitative.Prism[4]}
                },
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fragments['peak_time'], y=fragments['peak_arousal'],
                mode='markers', name='Intensity peak',
                marker={
                    'symbol': 'line-ns-open',
                    'size': 6, 'color': px.colors.qualitative.Prism[4],
                    'line': {'width': 2, 'color': px.colors.qualitative.Prism[4]}
                },
            )
        )
        fig.update_xaxes(tickformat="%M:%S")
        fig.update_layout(legend={'orientation': 'h', 'yanchor': 'top', 'y': 1.2, 'xanchor': 'right', 'x': 1.0})
        st.plotly_chart(fig, use_container_width=True)


def display_fragments_table(fragments: pd.DataFrame):
    """Displays the fragments list."""
    st.markdown('<h4 style="text-align: center;">Fragments</h4>', unsafe_allow_html=True)
    for _ in range(3):
        st.markdown('')
    df = fragments[['start_time', 'end_time']]
    df['duration'] = df['end_time'] - df['start_time']
    df['duration'] = df['duration'].dt.total_seconds().round(3).apply(str)
    df['start_time'] = df['start_time'].dt.strftime('%M:%S') + '.' + \
                       (df['start_time'].dt.microsecond // 1000).apply(str)
    df['end_time'] = df['end_time'].dt.strftime('%M:%S') + '.' + \
                     (df['end_time'].dt.microsecond // 1000).apply(str)
    df.rename(columns={'start_time': 'Start', 'end_time': 'End', 'duration': 'Duration [sec]', }, inplace=True)
    df.index = range(1, df.shape[0] + 1)
    df.index.name = 'Number'
    st.table(df)


def save_trailer(video_id: VideoId, video_info: VideoInfo, fragments: pd.DataFrame):
    """Saves the trailer to a file."""
    with st.spinner('Trailer saving...'):
        video_capture = cv2.VideoCapture(video_id.file_name)

        temp_name = 'temp.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        trailer_name = Path(video_id.file_name).stem + '_trailer.mp4'
        trailer_writer = cv2.VideoWriter(
            temp_name, fourcc, video_info.frame_rate,
            (video_info.frame_width, video_info.frame_height))
        trailer_frames_number = (fragments['end_frame'] - fragments['start_frame'] + 1).sum()
        frames_number = 0
        empty = st.empty()
        progress_bar = empty.progress(0.0)
        for fragment, (start_frame, end_frame) in fragments[['start_frame', 'end_frame']].iterrows():
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame + 1):
                _, frame = video_capture.read()
                trailer_writer.write(frame)
                frames_number += 1
                progress_bar.progress(frames_number / trailer_frames_number)
        empty.empty()
        video_capture.release()
        trailer_writer.release()
        args = f"ffmpeg -y -i ./{temp_name} -c:v libx264 ./{trailer_name}".split(" ")
        subprocess.call(args=args)
        Path(temp_name).unlink()
    return trailer_name


def download_trailer(trailer_name: str):
    """The trailer file downloading."""
    with open(trailer_name, mode='rb') as trailer_file:
        st.download_button('Download the trailer...', trailer_file, trailer_name,
                           'video' + Path(trailer_name).suffix[1:])


def show_title_and_help():
    st.markdown('<h2 style="text-align: center;">Trailer creator</h2>', unsafe_allow_html=True)
    with st.expander('See brief help...', icon=':material/help:'):
        st.markdown('''
        #### About app
        It\'s a very simple easy-to-use app for trailer creating for videos with one person.
        This app demonstrates abilities of a special computer vision 
        [model](https://github.com/alexeysher/skillbox-computer-vision-project).
        Face detecting realized with [OpenCV](https://docs.opencv.org/).

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

        Then the app fitting smoothed curve of emotion intensity utilizing rolling averaging.
        For adjusting this you can change time window size.

        After that app is searching for fragments containing a peak of emotion intensity.
        Each such one should meet following criteria:
        - greater then min
        - ...

        #### How to use
        For trailer creation just follow the next steps:
        1. Upload video from your file storage.

            Just drag it from your file explorer and drop it on the related control, 
            or click the *Browser files* button and select the one in the file explorer.
            When file uploading finish the video will be opened in the player 
            under the file downloading control.

        2. Wait util the video will be processed

            Emotion intensity in each frame will be recognized.
            When the process ends you will see the intensity chart under the video player.

        3. Adjust intensity curve fitting.

            Using *Time window size* control on the *left-side panel*  

        4. Adjust fragment detection

            To do that use the following controls on the *left-side panel*:
                - dd
                - ddd
                - dddd

        5. Specify 

        6. Playback the created trailer 
        containing selected fragments 

        7. Download the trailer. 

            In case if the result is not satisfied back to step 3. 

        #### Notes
        For testing app please download a small [collection]() of short videos.
        ''')


def on_video_upload():
    st.session_state['video_uploader_changed'] = True
    st.session_state['tma_window_changed'] = True


def on_tma_window_changed():
    st.session_state['tma_window_changed'] = True


if __name__ == '__main__':
    show_title_and_help()
    st.markdown('')
    with st.spinner('Connecting to Google Cloud...'):
        credentials = create_gc_credentials()
        connect_to_gc_storage_bucket(credentials=credentials)
        init_aiplatform(credentials=credentials)
    if ['face_detector'] in st.session_state:
        face_detector = st.session_state['face_detector']
    else:
        with st.spinner('Creating face detector...'):
            face_detector = create_face_detector()
            st.session_state['face_detector'] = face_detector
    if 'emotion_recognizer_endpoint' in st.session_state:
        emotion_recognizer_endpoint = st.session_state['emotion_recognizer_endpoint']
    else:
        with st.spinner('Connection to emotion recognizer...'):
            emotion_recognizer_endpoint = create_emotion_recognizer_endpoint()
            st.session_state['emotion_recognizer_endpoint'] = emotion_recognizer_endpoint
    video_col_1, trailer_col_1 = st.columns(2, gap='large')
    video_col_2, trailer_col_2 = st.columns(2, gap='large')
    video_col_3, trailer_col_3 = st.columns(2, gap='large')
    with video_col_1:
        st.markdown('<h3 style="text-align: center;">Video</h3>', unsafe_allow_html=True)
    with trailer_col_1:
        st.markdown('<h3 style="text-align: center;">Trailer</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:10px"><br></p>', unsafe_allow_html=True)
    with video_col_1:
        uploaded_video = upload_video()
    if uploaded_video is None:
        with trailer_col_1:
            st.info('Video have to be uploaded', icon=":material/info:")
            st.session_state['initial_hyperparams'] = None
    # else:
    #     if st.session_state['video_uploader_changed']:
    #         st.session_state['video_uploader_changed'] = False
    #         video_data = uploaded_video.read()
    #         video_id = VideoId()
    #         video_id.file_name = uploaded_video.name
    #         video_id.md5 = compute_video_hash(video_data)
    #         video_id.id = f'{Path(video_id.file_name).stem}_{video_id.md5}'
    #         st.session_state['video_id'] = video_id
    #         save_video(video_id, video_data)
    #         video_info = retrieve_video_info(video_id)
    #         st.session_state['video_info'] = video_info
    #         with video_col_2:
    #             arousals = get_arousals(video_id, video_info, face_detector, emotion_recognizer_endpoint)
    #         st.session_state['arousals'] = arousals
    #         initial_hyperparams = init_hyperparams(video_id)
    #         st.session_state['initial_hyperparams'] = initial_hyperparams
    #     else:
    #         video_id = st.session_state['video_id']
    #         video_info = st.session_state['video_info']
    #         arousals = st.session_state['arousals']
    #         initial_hyperparams = st.session_state['initial_hyperparams']
    #     with video_col_2:
    #         st.video(video_id.file_name)
    #     hyperparams = set_hyperparams(video_id, initial_hyperparams)
    #     if st.session_state['tma_window_changed']:
    #         st.session_state['tma_window_changed'] = False
    #         trend = fit_intensity_curve(arousals, hyperparams.tma_window, video_info)
    #         st.session_state['trend'] = trend
    #     else:
    #         trend = st.session_state['trend']
    #     fragments = find_fragments(trend, hyperparams, video_info)
    #     with video_col_3:
    #         plot_chart(trend, fragments)
    #     if fragments.empty:
    #         with trailer_col_1:
    #             st.warning('No fragment is found', icon=":material/warning:")
    #     else:
    #         trailer_name = save_trailer(video_id, video_info, fragments)
    #         with trailer_col_1:
    #             download_trailer(trailer_name)
    #         with trailer_col_2:
    #             st.video(trailer_name)
    #         with trailer_col_3:
    #             display_fragments_table(fragments)
