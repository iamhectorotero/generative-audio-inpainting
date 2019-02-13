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


class PatchedStrokeDS(Dataset):
    def __init__(self, stroke, patch_width, transformation, idx_mapping=None):
        """
        Simple Dataset which takes patches (full height) from a continuous stroke of data.
        
        @param stroke: the data; dimension: [strokes, stroke_height, stroke_width]
        @param patch_width: the width of the patches to return
        @param transformation: transformation applied to the patch (returned together with the original)
        @param idx_mapping which locations of the stroke (* patch_width) are valid
        """
    
        self.stroke = stroke 
        self.patch_width = patch_width
        self.transformation = transformation
        
        if idx_mapping is None:
            idx_mapping = range(data.shape[1] // patch_width - 1)
            
        self.idx_mapping = idx_mapping
    
    def __getitem__(self, index):
        index = self.idx_mapping[index]
        rnd = np.random.randint(0, self.patch_width)
        
        target = self.stroke[:,:,rnd + self.patch_width * index:rnd + self.patch_width * (index + 1)]
                
        return self.transformation(target), target
        
    def __len__(self):
        return len(self.idx_mapping)
        
    def save_data(self, path):
        pk.dump((self.stroke, self.patch_width, self.idx_mapping), open(path, "wb")) 
        
    @staticmethod
    def from_file(self, path, transformation):
        return FreqDS(*pk.load(open(path, "rb")), transformation)
    
    
class WAVAudioDS(PatchedStrokeDS):
    def __init__(self, files, transformation, preprocess, patch_width, nperseg = 256):
        """
        Reads all audio files, applies a sftf and combines the result into one continous stroke of which patches are returned with width patch_width
        
        @param files: the audio files to load
        @param transformation: a transformation to apply to the patches before they are returned (the original is returned as well)
        @param preprocess: preprocessing step applied to the frequency data of the audio, this should cast the data to Tensors at the very least
        @param patch_width: the width of the patches that are returned
        @param nperseg: param used for the stft
        """
        file_data = []
        idx_mapping = []
        i = 0
        
        for file in tqdm(files):
            self.fs, audio_time = audio.read_monaural_wav(file)
            audio_freqs = preprocess(audio.stft(audio_time, self.fs, nperseg=nperseg)[2])
                
            if audio_freqs is not None:
                file_data.append(audio_freqs)
                idx_mapping.extend(range(i + len(idx_mapping), i + len(idx_mapping) + audio_freqs.shape[2] // patch_width - 1))
                i += 1
                    
        super(WAVAudioDS, self).__init__(torch.cat(file_data, dim=2), patch_width, transformation, idx_mapping)
        
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
        
    @staticmethod
    def polar_preprocessing(norm_mag, norm_phase, patch_width, max_freqs=64):
        def preprocess(freqs):
            freqs = audio.cutout_slient(freqs, min_width=patch_width)
            
            if freqs is None:
                return None
            
            mod = freqs.shape[1] % patch_width
            
            if mod is not 0: 
                freqs = freqs[:,:-mod]
            
            freqs = WAVAudioDS.freqs_to_torch(freqs, max_freqs)

            freqs[0], freqs[1] = complx.to_polar(freqs)
            freqs[0], freqs[1] = norm_mag(freqs[0]), norm_phase(freqs[1])

            return freqs
        
        return preprocess