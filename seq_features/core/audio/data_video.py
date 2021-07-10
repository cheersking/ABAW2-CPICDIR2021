# get audio
import torchaudio
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import math
import subprocess
from utils import *
from clip_transforms import *
from video import Video
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

# audio_file = '/data/tian/Data/Aff-Wild2-audio/batch1_audio/1-30-1280x720.wav'
# audio, sample_rate = torchaudio.load(audio_file)
# print(audio.shape, sample_rate)
 # audio params
# self.window_size = 20e-3
# self.window_stride = 10e-3
# self.sample_rate = 44100
# num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
# window_fn = torch.hann_window

# self.sample_len_secs = 10
# self.sample_len_frames = self.sample_len_secs * self.sample_rate
# self.audio_shift_sec = 5
# self.audio_shift_samples = self.audio_shift_sec * self.sample_rate
# # transforms
# audio, sample_rate = torchaudio.load(audio_file,
#                                     num_frames=10*44100,
#                                     )
# print(audio.shape, sample_rate)
                                    #  num_frames=min(10*44100,
                                    #                 max(int((1/1000) * 44100),
                                    #                     int(20e-3 * 44100))),
                                    #  offset=max(int((1/1000) * 44100
                                    #                 - sample_len_frames + audio_shift_samples), 0))

class Aff2CompDataset(Dataset):
    def __init__(self, root_dir=''):
        super(Aff2CompDataset, self).__init__()
        self.video_dir = root_dir
        self.extracted_dir = os.path.join(self.video_dir, 'extracted')

        self.clip_len = 8
        self.input_size = (112, 112)
        self.dilation = 6
        self.label_frame = self.clip_len * self.dilation

        # audio params
        # self.window_size = 20e-3 #20ms
        # self.window_stride = 10e-3 #10ms
        self.window_size = 10e-3 #20ms
        self.window_stride = 5e-3 #10ms
        self.sample_rate = 44100
        num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
        window_fn = torch.hann_window

        self.sample_len_secs = 10
        self.sample_len_frames = self.sample_len_secs * self.sample_rate
        self.audio_shift_sec = 5
        self.audio_shift_samples = self.audio_shift_sec * self.sample_rate
        # transforms

        # self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=64,
        #                                                             n_fft=num_fft,
        #                                                             win_length=int(self.window_size * self.sample_rate),
        #                                                             hop_length=int(self.window_stride
        #                                                                            * self.sample_rate),
        #                                                             window_fn=window_fn)
        self.audio_transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate,log_mels=True)

        self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])
        # self.clip_transform = ComposeWithInvert([NumpyToTensor(), Normalize(mean=[0.43216, 0.394666, 0.37645, 0.5],
        #                                                                     std=[0.22803, 0.22145, 0.216989, 0.225])])


        # all_videos = find_all_video_files(self.video_dir)
        # self.cached_metadata_path = os.path.join(self.video_dir, 'dataset.pkl')


    def set_clip_len(self, clip_len):
        assert(np.mod(clip_len, 2) == 0)  # clip length should be even at this point
        self.clip_len = clip_len

    def set_modes(self, modes):
        self.modes = modes

    def __getitem__(self, index):

        # get labels, convert it to one-hot encoding etc.
        # ...
        # compute pseudo-labels for ex and va using the distribution of ex- and va-labels
        # ...
        # original code
        # data = {'AU': None,
        #         'EX': None,
        #         'VA': None,
        #         'Index': index}
        # why is this not working?
        data = {'Index': index}

        # get audio
        # audio_file = os.path.join(self.video_dir, self.video_id[index] + '.wav')
        # audio_file = '/data/tian/Data/Aff-Wild2-audio/batch1_audio/1-30-1280x720.wav'
        audio_file = '/data/tian/Data/Aff-Wild2-audio/audio/2-30-640x360.wav'

        # audio, sample_rate = torchaudio.load(audio_file,
        #                                      num_frames=min(self.sample_len_frames,
        #                                                     max(int((self.time_stamps[index]/1000) * self.sample_rate),
        #                                                         int(self.window_size * self.sample_rate))),
        #                                      offset=max(int((self.time_stamps[index]/1000) * self.sample_rate
        #      
        #                                                - self.sample_len_frames + self.audio_shift_samples), 0))
        # audio, sample_rate = torchaudio.load(audio_file)
        audio, sample_rate = torchaudio.load(audio_file,
                                    num_frames=44100,
                                    )
        #audio[2,441000]
        audio_features = self.audio_transform(audio).detach()#[2,64,1001]
        # print(audio_features)
        # plt.figure()
        # plt.imshow(audio_features.log2()[0,:,:].numpy())
        # plt.savefig('./test0.png')
        # if audio.shape[1] < self.sample_len_frames:
        #     _audio_features = torch.zeros((audio_features.shape[0], audio_features.shape[1],
        #                                    int((self.sample_len_secs / self.window_stride) + 1)))
        #     _audio_features[:, :, -audio_features.shape[2]:] = audio_features
        #     audio_features = _audio_features
        #10s对应1001

        # if self.audio_spec_transform is not None:
        #     audio_features = self.audio_spec_transform(audio_features)
        print(audio_features.shape)
        plt.figure()
        plt.imshow(audio_features[0,:,:].numpy(),cmap='plasma')
        plt.colorbar()
        plt.savefig('./test2-0-1-1.png')

        # data['audio_features'] = audio_features

        # if audio.shape[1] < self.sample_len_frames:
        #     _audio = torch.zeros((1, self.sample_len_frames))
        #     _audio[:, -audio.shape[1]:] = audio
        #     audio = _audio
        data['audio'] = audio

        return data

    def __len__(self):
        return len(self.video_id)

    def __add__(self, other):
        raise NotImplementedError


dataset = Aff2CompDataset()
dataset.__getitem__(0)