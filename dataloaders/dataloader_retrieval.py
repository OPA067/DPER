from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from os.path import exists

import random
import numpy as np
from torch.utils.data import Dataset

import torch
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
import dataloaders.video_transforms as video_transforms


class RetrievalDataset(Dataset):

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            max_frames=12,
            video_framerate=1,
            image_resolution=224,
            mode='all',
            config=None
    ):
        self.subset = subset
        self.anno_path = anno_path
        self.video_path = video_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.video_framerate = video_framerate
        self.image_resolution = image_resolution
        self.mode = mode
        self.config = config

        self.video_dict, self.sentences_dict = self._get_anns(self.subset)

        self.video_list = list(self.video_dict.keys())
        self.sample_len = 0

        print("Video Numbers: {}".format(len(self.video_dict)))
        print("Total Pairs: {}".format(len(self.sentences_dict)))

        from .rawvideo_util import RawVideoExtractor
        self.rawVideoExtractor = RawVideoExtractor(framerate=video_framerate, size=image_resolution)
        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.tsfm_dict = {
            'clip_test': Compose([
                Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_resolution),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]),
            'clip_train': Compose([
                RandomResizedCrop(image_resolution, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.image_resolution = image_resolution
        if self.mode in ['all', 'text']:
            self.sample_len = len(self.sentences_dict)
        else:
            self.sample_len = len(self.video_list)
        self.aug_transform = video_transforms.create_random_augment(
            input_size=(self.image_resolution, self.image_resolution),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

    def __len__(self):
        return self.sample_len

    def _get_anns(self, subset='train'):
        raise NotImplementedError

    def _get_text(self, caption):
        if len(caption) == 3:
            _caption_text, s, e = caption
        else:
            raise NotImplementedError

        if isinstance(_caption_text, list):
            caption_text = random.choice(_caption_text)
        else:
            caption_text = _caption_text
        
        words = self.tokenizer.tokenize(caption_text)
        
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words

        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)

        return input_ids, input_mask, s, e

    def _get_rawvideo(self, video_id, s=None, e=None):
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        video = np.zeros((self.max_frames, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = end_time + 1
        video_path = self.video_dict[video_id]

        raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
        raw_video_data = raw_video_data['video']

        if len(raw_video_data.shape) > 3:

            if self.max_frames < raw_video_data.shape[0]:
                sample_indx = np.linspace(0, raw_video_data.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_data[sample_indx, ...]
            else:
                video_slice = raw_video_data

            video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=0)

            slice_len = video_slice.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = video_slice
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def _get_rawvideo_dec(self, video_id, s=None, e=None):
        # speed up video decode via decord.
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        video_path = self.video_dict[video_id]

        if exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(self.video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            if self.subset == "train":
                patch_images = self.aug_transform(patch_images)

            patch_images = torch.stack([self.transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def get_text_len(self):
        return len(self.sentences_dict)

    def get_video_len(self):
        return len(self.video_list)

    def get_text_content(self, ind):
        return self.sentences_dict[ind][1]

    def get_data_name(self):
        return self.__class__.__name__ + "_" + self.subset

    def get_vis_info(self, idx):
        video_id, caption = self.sentences_dict[idx]
        video_path = self.video_dict[video_id]
        return caption, video_path

    def __getitem__(self, idx):

        video_id, caption = self.sentences_dict[idx]
        text_ids, text_mask, s, e = self._get_text(caption)
        video, video_mask = self._get_rawvideo_dec(video_id, s, e)
        # video, video_mask = self._get_rawvideo(video_id, s, e)
        return text_ids, text_mask, video, video_mask, idx, hash(video_id.replace("video", ""))
