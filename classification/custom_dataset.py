import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from mmaction.datasets.pipelines import SampleFrames
from typing import List, Union, Tuple, Any


class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 
             2) The second element is the inclusive ending frame id of the video
             3) The third element is the label index.
             4) any following elements are labels in the case of multi-label classification
    """

    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return 1

    @property
    def end_frame(self) -> int:
        return int(self._data[1])

    @property
    def label(self) -> Union[int, List[int]]:
        # Just one label_id
        if len(self._data) == 3:
            return int(self._data[2])
        # Sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[2:]]


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    This class relies on receiving video data in a structure where
    inside a ``ROOT_DATA`` folder, each video lies in its own folder,
    where each video folder contains the frames of the video as
    individual files with a naming convention such as
    img_001.jpg ... img_059.jpg.
    For enumeration and annotations, this class expects to receive
    the path to a .txt file where each video sample has a row with four
    (or more in the case of multi-label, see README on Github)
    space separated values:
    ``VIDEO_FOLDER_PATH      NUMBER_OF_FRAMES      LABEL_INDEX``.
    ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
    excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
    be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
    might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        root_path (str): The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path (str): The .txt annotation file containing
                             one row per video sample as described above.
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        start_index (None): This argument is deprecated.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 keep_tail_frames: bool = False,
                 imagefile_template: str = 'img_{:05d}.jpg',
                 transform=None,
                 test_mode: bool = False
                 ):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.keep_tail_frames = keep_tail_frames
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self._parse_annotationfile()
        self._sanity_check_samples()

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')

    def _parse_annotationfile(self):
        self.video_list = [VideoRecord(
            x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0:
                print(
                    f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n")

    def _get_sample_frames(self, record: VideoRecord) -> 'np.ndarray[int]':
        """
        Use MMAction's SampleFrames to select frames for our RawFrameDataset.'

        Args:
            record (VideoRecord): VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of the
            record are to be loaded from.
        """
        # Set up config for SampleFrames
        config = dict(clip_len=self.clip_len,
                      frame_interval=self.frame_interval,
                      num_clips=self.num_clips,
                      temporal_jitter=self.temporal_jitter,
                      twice_sample=self.twice_sample,
                      out_of_bound_opt=self.out_of_bound_opt,
                      test_mode=self.test_mode,
                      keep_tail_frames=self.keep_tail_frames
                      )

        # Create SampleFrames
        sample_frames = SampleFrames(**config)

        # Set up input for sample_frames
        frame_result = dict(frame_inds=range(1, record.end_frame+1),
                            total_frames=record.num_frames,
                            start_index=1
                            )

        return sample_frames(frame_result)['frame_inds']

    def __getitem__(self, idx: int) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]',
              Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
    ]:
        """
        For video with id idx, loads the frames from SampleFrames.

        Args:
            idx (int): Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record: VideoRecord = self.video_list[idx]

        frame_start_indices: 'np.ndarray[int]' = self._get_sample_frames(
            record)

        return self._get(record, frame_start_indices)

    def _get(self, record: VideoRecord, frame_start_indices: 'np.ndarray[int]') -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]',
              Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
    ]:
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        images = list()

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for frame_index in frame_start_indices:
            image = self._load_image(record.path, frame_index)
            images.append(image)

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    @staticmethod
    def forward(img_list: List[Image.Image]) -> 'torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]':
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
