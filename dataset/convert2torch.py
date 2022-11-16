"""
Converts Tensorflow records into gzip file format.

"""


import os
import sys
import collections
import torch
import gzip
import pathlib
from functools import partial
import multiprocessing as mp
from threading import Lock
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['name', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        name='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        name='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        name='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        name='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        name='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        name='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        name='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)


def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def preprocess_frames(dataset_info, example, jpeg='False'):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    if not jpeg:
        frames = tf.map_fn(_convert_frame_data, tf.reshape(frames, [-1]),dtype=tf.float32, back_prop=False)
        dataset_image_dimensions = tuple([dataset_info.frame_size] * 2 + [3])
        frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + dataset_image_dimensions)
        if (64 and 64 != dataset_info.frame_size):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (64,) * 2 + (3,)
            frames = tf.image.resize_bilinear(frames, new_frame_dimensions[:2], align_corners=True)
            frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + new_frame_dimensions)
    return frames


def preprocess_cameras(dataset_info, example, raw):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, dataset_info.sequence_size, 5])
    if not raw:
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
        return cameras
    else:
        return raw_pose_params


def collect_files(path, ext=None, key=None):
    if key is None:
        files = sorted(os.listdir(path))
    else:
        files = sorted(os.listdir(path), key=key)

    if ext is not None:
        files = [f for f in files if os.path.splitext(f)[-1] == ext]

    return [os.path.join(path, fname) for fname in files]

def encapsulate(frames, cameras):
    return Scene(cameras=cameras, frames=frames)


def convert_raw_to_numpy(dataset_info, raw_data, path, jpeg=False):
    feature_map = {
        'frames': tf.compat.v1.FixedLenFeature(
            shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.compat.v1.FixedLenFeature(
            shape=[dataset_info.sequence_size * 5],
            dtype=tf.float32)
    }
    example = tf.compat.v1.parse_single_example(raw_data, feature_map)
    frames = preprocess_frames(dataset_info, example, jpeg)
    cameras = preprocess_cameras(dataset_info, example, jpeg)
    # tf.compat.v1.disable_eager_execution()
    # with tf.compat.v1.train.SingularMonitoredSession() as sess:
    # with tf.compat.v1.train.MonitoredSession() as sess:
        # frames = sess.run(frames)
        # cameras = sess.run(cameras)
    scene = encapsulate(frames, cameras)
    with gzip.open(path, 'wb') as f:
        torch.save(scene, f)


def show_frame(frames, scene, views):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.imshow(frames[scene,views])
    plt.show()

class Counter():
    """
        This is a static thread-safe singleton counter
    """
    _counter = 0
    _lock = Lock()
 
    @staticmethod
    def reset():
        with Counter._lock:
            Counter._counter = 0

    @staticmethod
    def get():
        with Counter._lock:
            val = Counter._counter
            Counter._counter += 1
            return val



def convert_record(filepath, dataset_info, dst_folder):
        engine = tf.compat.v1.python_io.tf_record_iterator(filepath)
        for i, raw_data in enumerate(engine):
            dstpath = os.path.join(dst_folder, f'{Counter.get()}.pt.gz')
            print(f' [-] converting scene {filepath}-{i} into {dstpath}')
            convert_raw_to_numpy(dataset_info, raw_data, dstpath, True)

if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()

    # handle arguments
    parser = ArgumentParser(description='Generative Query Network Implementation')
    parser.add_argument('dataset',
                        type=str,
                        default='shepard_metzler_5_parts',
                        choices=list(_DATASETS.keys()))
    parser.add_argument('-l', '--location',
                        type=str,
                        default='shepard_metzler_5_parts',
                        help='path to dataset')
    args = parser.parse_args()

    # get argument values
    DATASET = args.dataset
    dataset_info = _DATASETS[DATASET]

    # source and destination folders
    src_dataset_path = pathlib.Path(args.location)
    dst_root_path = pathlib.Path(__file__).parent # the destination folder is this file's folder

    # if the source path is relative, then it is relative to this file
    if not src_dataset_path.is_absolute():
        src_dataset_path = dst_root_path / src_dataset_path

    torch_dataset_path = dst_root_path / pathlib.Path(f'{src_dataset_path.parts[-1]}-torch') # postfix the source folder name
    torch_dataset_path_train = torch_dataset_path / 'train'
    torch_dataset_path_test = torch_dataset_path / 'test'

    # create destination folders, these should not exist yet, if they do the program 
    # will exit to prevent overwriting existing folders
    # TODO: remove exists_ok=True
    torch_dataset_path.mkdir(exist_ok=True)
    torch_dataset_path_train.mkdir(exist_ok=True)
    torch_dataset_path_test.mkdir(exist_ok=True)


    cores = mp.cpu_count()

    ## train
    file_names = collect_files(src_dataset_path / 'train')
    with mp.Pool(processes=cores) as pool:
        f = partial(convert_record, dataset_info=dataset_info, dst_folder=torch_dataset_path_train)
        pool.map(f, file_names)
    print(f' [-] {Counter.get()} samples in the train dataset')
    
    ## test
    Counter.reset()
    file_names = collect_files(src_dataset_path / 'test')
    with mp.Pool(processes=cores) as pool:
        f = partial(convert_record, dataset_info=dataset_info, dst_folder=torch_dataset_path_test)
        pool.map(f, file_names)
    print(f' [-] {Counter.get()} samples in the test dataset')
