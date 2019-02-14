import scipy.io.wavfile as wav
import scipy.signal as sig

import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

import torch
from torch.utils.data import Dataset

from tqdm import tqdm_notebook as tqdm

from . import audio
from . import complx
from . import utils

from itertools import accumulate

from .normalization import denorm_polar


class MultiSet(Dataset):
    def __init__(self, sets):
        lengths = list(map(len, sets))
        
        self.acc_len = list(accumulate(lengths))        
        self.sets = sets
        self.length = sum(lengths)
        self.idx_to_set = {i:utils.acc_to_idx(self.acc_len, i) for i in range(self.length)}

    def __getitem__(self, index):
        set_idx = self.idx_to_set[index]
        return self.sets[set_idx][index - self.acc_len[set_idx]]
        
    def __len__(self):
        return self.length


class PatchedStrokeDS(Dataset):
    def __init__(self, stroke, patch_width, mk_source, idx_mapping=None, transform=lambda x:x, random_patches=True):
        """
        Simple Dataset which takes patches (full height) from a continuous stroke of data.
        
        @param stroke: the data; dimension: [strokes, stroke_height, stroke_width]
        @param patch_width: the width of the patches to return
        @param transformation: transformation applied to the patch (returned together with the original)
        @param idx_mapping which locations of the stroke (* patch_width) are valid
        """
    
        self.stroke = stroke 
        self.patch_width = patch_width
        self.mk_source = mk_source
        self.transform = transform
        self.random_patches = random_patches
        
        if idx_mapping is None:
            idx_mapping = range(data.shape[1] // patch_width - 1)
            
        self.idx_mapping = idx_mapping
    
    def __getitem__(self, index):
        index = self.idx_mapping[index]
        
        if self.random_patches:
            rnd = np.random.randint(0, self.patch_width)
        else:
            rnd = 0
        
        target = self.stroke[:,:,rnd + self.patch_width * index:rnd + self.patch_width * (index + 1)]
        target = self.transform(target)
            
        return self.mk_source(target), target
        
    def __len__(self):
        return len(self.idx_mapping)
        
    def save_data(self, path):
        pk.dump((self.stroke, self.patch_width, self.idx_mapping), open(path, "wb")) 
        
    @staticmethod
    def from_file(self, path, transformation):
        return FreqDS(*pk.load(open(path, "rb")), transformation)
    
    
class WAVAudioDS(PatchedStrokeDS):
    def __init__(self, files, mk_source, preprocess, patch_width, proc_pool, transform = lambda x:x, nperseg = 256):
        """
        Reads all audio files, applies a sftf and combines the result into one continous stroke of which patches are returned with width patch_width
        
        @param files: the audio files to load
        @param transformation: a transformation to apply to the patches before they are returned (the original is returned as well)
        @param preprocess: preprocessing step applied to the frequency data of the audio, this should cast the data to Tensors at the very least
        @param patch_width: the width of the patches that are returned
        @param nperseg: param used for the stft
        """
        
        freq_data = proc_pool.map(Pipeline(preprocess, nperseg), tqdm(files))
        freq_data = list(filter(lambda x: x is not None, freq_data))
        idx_mapping = []
        
        for i, audio_freqs in enumerate(freq_data): 
            idx_mapping.extend(range(i + len(idx_mapping), i + len(idx_mapping) + audio_freqs.shape[-1] // patch_width - 1))
                    
        super(WAVAudioDS, self).__init__(torch.cat(freq_data, dim=-1), patch_width, mk_source, idx_mapping, transform)
        
    @staticmethod
    def freqs_to_torch(freqs, max_freqs):
        return torch.from_numpy(freqs.view(np.float32).reshape(129, 2, -1)[:max_freqs].transpose(1, 0, 2))
        
    @staticmethod
    def torch_to_freqs(audio_freqs, denorm=lambda x:x):
        freqs = denorm(audio_freqs).data.cpu().numpy()
        freqs = freqs[0] + freqs[1] * 1j
        zeros = np.zeros((129, audio_freqs.shape[2]), dtype=np.complex64)
        zeros[:64] = freqs

        return zeros
        
        
class PolarPreprocessing:
    def __init__(self, norm_mag, norm_phase, patch_width, include_phase = True, max_freqs=64):
        self.norm_mag = norm_mag
        self.norm_phase = norm_phase
        self.patch_width = patch_width
        self.max_freqs = max_freqs
        self.include_phase = include_phase
    
    def __call__(self, freqs):
        freqs = audio.cutout_slient(freqs, min_width=self.patch_width)
           
        if freqs is None:
            return None
            
        mod = freqs.shape[1] % self.patch_width
            
        if mod is not 0: 
            freqs = freqs[:,:-mod]
            
        freqs = WAVAudioDS.freqs_to_torch(freqs, self.max_freqs)

        freqs[0], freqs[1] = complx.to_polar(freqs)
        freqs[0], freqs[1] = self.norm_mag(freqs[0]), self.norm_phase(freqs[1])

        if self.include_phase:
            return freqs
        else:
            return freqs[0]
        
        
class TimePostprocessing:
    def __init__(self, fs):
        self.fs = fs
        
    def __call__(self, polar):
        freqs = WAVAudioDS.torch_to_freqs(polar, denorm_polar)
        return audio.istft(freqs, self.fs)[1]
        

class Pipeline:
    def __init__(self, preprocess, nperseg=256):
        self.nperseg = 256
        self.preprocess = preprocess
        
    def __call__(self, file):
        fs, audio_time = audio.read_monaural_wav(file)
        _, _, freqs = audio.stft(audio_time, fs, self.nperseg) 
        return self.preprocess(freqs)