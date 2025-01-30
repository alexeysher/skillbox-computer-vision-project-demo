import subprocess

import gdown
from shutil import rmtree

from pathlib import Path
from datetime import timedelta, datetime, date

import pandas as pd
import streamlit as st
import cv2
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go

from emotion_recognition import FaceEmotionRecognitionNet, EMOTIONS

st.set_page_config(page_title='Creating trailers', page_icon=':film_frames:',
                   layout="wide", initial_sidebar_state="expanded")

BACKGROUND_COLOR = 'white'
COLOR = 'black'


class VideoInfo:
    """
    Video info.
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
    tma_window: float = 0.
    min_arousal: float = 0.
    min_prominence: float = 0.
    min_duration: float = 0.
    rel_height: float = 0.
    min_distance: float = 0.


MODEL_PATH = 'model'  # Path to saved model
HAAR_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Face detector settings
SCALE_FACTOR = 1.1  # Scale factor
MIN_NEIGHBORS = 3  # Min. neighbors number
MIN_FACE_SIZE = 64  # Min. face size


@st.cache_resource(show_spinner="Creating face detector...")
def create_face_detector():
    """Creates face detector."""
    detector = cv2.CascadeClassifier(HAAR_FILE)
    return detector


@st.cache_resource(show_spinner='Downloading the emotion recognition model...')
def create_emotion_recognizer() -> FaceEmotionRecognitionNet:
    """Creates emotion recognition model."""
    print('!'*100)
    print(st.secrets['model']['url'])
    print('!'*100)
    model_zip_path = Path('tmp.zip')
    model_path = Path(MODEL_PATH)
    gdown.cached_download(st.secrets['model']['url'], path=model_zip_path.as_posix(),
                          postprocess=gdown.extractall, fuzzy=True)
    model = FaceEmotionRecognitionNet(MODEL_PATH, EMOTIONS)
    model_zip_path.unlink()
    rmtree(model_path)
    return model


def upload_video() -> [str, VideoData]:
    """Uploads video file."""
    uploaded_file = st.file_uploader('Video file uploading', ['mp4', 'avi', 'mov'], label_visibility='hidden')
    return uploaded_file


@st.cache_resource
def save_video(file_name: str, _video_data: VideoData):
    """Saves video data to file"""
    for file in Path('.').iterdir():
        if file.suffix in ('.mp4', '.avi', '.mov'):
            file.unlink()
    with open(file_name, mode='wb') as video:
        video.write(_video_data)


@st.cache_resource
def retrieve_video_info(file_name: str) -> VideoInfo:
    """Extracts info about video data from the file."""
    video_info = VideoInfo()
    capture = cv2.VideoCapture(file_name)
    video_info.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_info.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info.frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
    video_info.frames_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_info.fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    video_info.duration = timedelta(seconds=(video_info.frames_number - 1) / video_info.frame_rate)
    video_info.duration_str = f'{video_info.duration.seconds // 60:02d}:{video_info.duration.seconds % 60:02d}'
    return video_info


@st.cache_data(show_spinner='Emotion recognition...')
def recognize_video_emotions(file_name: str, _video_info: VideoInfo, _face_detector,
                             _emotion_recognizer: FaceEmotionRecognitionNet) -> [float]:
    """Recognize intensity (arousal) of the emotions in each video frame."""

    def extract_face(frame, detector):
        """Extracts a face from the frame."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = detector.detectMultiScale(gray_frame, SCALE_FACTOR, MIN_NEIGHBORS, 0,
                                               (MIN_FACE_SIZE, MIN_FACE_SIZE))
        if len(faces_rect) == 0:
            return
        x, y, w, h = faces_rect[0]
        face = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
        return face

    video_capture = cv2.VideoCapture(file_name)
    frame_index = 0
    faces_number = 0
    arousal = 0.0
    arousals = []
    empty = st.empty()
    progress_bar = empty.progress(0.0)
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        face_image = extract_face(image, _face_detector)
        if face_image is not None:
            faces_number += 1
            arousal = _emotion_recognizer.predict(face_image)[-1]
        arousals.append(arousal)
        frame_index += 1
        progress_bar.progress(frame_index / _video_info.frames_number)
    video_capture.release()
    empty.empty()
    return arousals


def set_hyperparams() -> HyperParams:
    """Sets up hyperparameters for fragments searching."""
    hyperparams = HyperParams()
    hyperparams.tma_window = st.slider('Intensity averaging time window [sec]', 0.05, 2.0, 1.0, 0.05)
    hyperparams.min_arousal = st.slider('Intensity peak min. value', 0.0, 1.0, 0.5, 0.05)
    hyperparams.min_prominence = st.slider('Intensity prominence min. value', 0.0, 0.5, 0.25, 0.05)
    hyperparams.min_duration = st.slider('Min. fragment duration [sec]', 0.0, 2.0, 1.0, 0.05)
    hyperparams.rel_height = st.slider('Intensity min. height value', 0.0, 1.0, 0.5, 0.05)
    hyperparams.min_distance = st.slider('Min. distance between fragments [sec]', 0.5, 10.0, 5.0, 0.5)
    return hyperparams


def find_fragments(file_name: str, hyperparams: HyperParams, video_info: VideoInfo, arousals: []) -> \
        (pd.DataFrame, pd.DataFrame):
    """Searching video fragments for the trailer."""

    today = date.today()
    start_date = datetime(today.year, today.month, today.day)
    end_date = start_date + video_info.duration
    trend = pd.DataFrame()
    trend['time'] = pd.date_range(start_date, end_date, periods=video_info.frames_number)
    trend['arousal'] = arousals
    tma_window = int(hyperparams.tma_window * video_info.frame_rate)
    trend['arousal_sma'] = trend['arousal'] \
        .rolling(window=tma_window, min_periods=1, center=True).mean()
    trend['arousal_tma'] = trend['arousal_sma'] \
        .rolling(window=tma_window, min_periods=1, center=True).mean()

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

    return trend, fragments


def plot_chart(trend: pd.DataFrame, fragments: pd.DataFrame):
    """Diagrams output."""
    st.markdown('##### Intensity dynamic')
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
    """Outputs the fragments list."""
    st.markdown('##### Fragments')
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


def save_trailer(file_name: str, video_info: VideoInfo, fragments: pd.DataFrame):
    """Saves the trailer to a file."""

    with st.spinner('Trailer saving...'):
        video_capture = cv2.VideoCapture(file_name)

        temp_name = 'temp.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        trailer_name = Path(file_name).stem + '_trailer.mp4'
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
        st.download_button('Download the trailer...', trailer_file, trailer_name, 'video' + Path(trailer_name).suffix[1:])


def main():
    face_detector = create_face_detector()
    emotion_recognizer = create_emotion_recognizer()
    # st.markdown('<h2 style="text-align: center;">Trailer creation</h2>', unsafe_allow_html=True)
    video_col_1, trailer_col_1 = st.columns(2, gap='large')
    with video_col_1:
        st.markdown('<h3 style="text-align: center;">Video</h3>', unsafe_allow_html=True)
        uploaded_file = upload_video()
        st.markdown(f"[Video files...]({st.secrets['video']['url']})")
    with trailer_col_1:
        st.markdown('<h3 style="text-align: center;">Trailer</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:10px"><br></p>', unsafe_allow_html=True)
        if uploaded_file is None:
            st.info('Video needs to be uploaded')
            return
    file_name = uploaded_file.name
    video_data = uploaded_file.read()
    save_video(file_name, video_data)
    video_col_2, trailer_col_2 = st.columns(2, gap='large')
    with video_col_2:
        st.video(video_data)
    video_info = retrieve_video_info(file_name)
    video_col_3, trailer_col_3 = st.columns(2, gap='large')
    with video_col_3:
        arousals = recognize_video_emotions(file_name, video_info, face_detector, emotion_recognizer)
    with st.sidebar:
        hyperparams = set_hyperparams()
    trend, fragments = find_fragments(file_name, hyperparams, video_info, arousals)
    with video_col_3:
        plot_chart(trend, fragments)
    with trailer_col_1:
        if fragments.empty:
            st.warning('No fragment is found')
            return
    with trailer_col_2:
        trailer_name = save_trailer(file_name, video_info, fragments)
        st.video(trailer_name)
    with trailer_col_3:
        display_fragments_table(fragments)
    with trailer_col_1:
        download_trailer(trailer_name)


main()
