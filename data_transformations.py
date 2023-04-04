import torchaudio
import numpy as np
import torch
import torch.nn.functional
from numpy.random import randint

import torchaudio_augmentations as ADA

SAMPLE_RATE = 32000

class DataAugmentation(object):
    def __init__(self, args):
        
        # for corruption
        self.drop_perc = args.drop_perc
        self.drop_type = args.drop_type
        self.drop_align = args.drop_align
        
        self.data_mean = args.data_mean
        self.data_std = args.data_std

        self.num_crops = args.num_crops
        
        self.num_frames = args.num_frames
        self.num_mel_bins = args.num_mel_bins
        
        # number of seconds to be used
        self.num_samples = SAMPLE_RATE * args.secs_per_crop
        
        self.transform = ADA.ComposeMany([ADA.RandomResizedCrop(n_samples=self.num_samples)], num_augmented_samples=self.num_crops)
        
        

    def GMML_drop_rand_patches(self, X, max_block_sz=0.3):
        #######################
        # max_replace: percentage of image to be replaced
        # align: align corruption with the patch sizes
        # max_block_sz: percentage of the maximum block to be dropped
        #######################
       
        np.random.seed()    
        C, H, W = X.size()
        n_drop_pix = np.random.uniform(min(0.5, self.drop_perc), self.drop_perc)*H*W
        mx_blk_height = int(H*max_block_sz)
        mx_blk_width = int(W*max_block_sz)
        
        align = max(1, self.drop_align)
        
        mask = torch.zeros_like(X)
        
        while mask.sum() < n_drop_pix:
            
            ####### get a random block to replace 
            rnd_r = ( randint(0, H-align) // align ) * align
            rnd_c = ( randint(0, W-align) // align ) * align

            rnd_h = min(randint(align, mx_blk_height), H-rnd_r)
            rnd_h = round(rnd_h / align) * align
            rnd_w = min(randint(align, mx_blk_width), W-rnd_c)
            rnd_w = round(rnd_w / align) * align

            if self.drop_type == 'noise':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.empty((C, rnd_h, rnd_w), 
                                                                    dtype=X.dtype, device=X.device).normal_(mean=self.data_mean, std=self.data_std)
                mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1 
            elif self.drop_type == 'zeros':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.zeros((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device)
                mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1 
            elif self.drop_type == 'time':
                X[:, rnd_r:rnd_r+rnd_h, :] = torch.zeros((C, rnd_h, W), dtype=X.dtype, device=X.device)
                mask[:, rnd_r:rnd_r+rnd_h, :] = 1 
            elif self.drop_type == 'freq':
                X[:, :, rnd_c:rnd_c+rnd_w] = torch.zeros((C, H, rnd_w), dtype=X.dtype, device=X.device)
                mask[:, :, rnd_c:rnd_c+rnd_w] = 1 
            else:
                print('Not Implemented!!')

        return X, mask
    
    def _wav2fbank(self, waveform):
        

        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=SAMPLE_RATE, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.num_mel_bins, dither=0.0, frame_shift=10)

        n_frames = fbank.shape[0]
        p = self.num_frames - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.num_frames, :]

        return fbank
        
    def __call__(self, waveform):
        
        #pad waveform if less than num_samples
        if waveform.size()[1] < self.num_samples:
            waveform = torch.cat( (waveform, torch.zeros(1, self.num_samples-waveform.size()[1]) ), dim=-1)
            
        waveform = self.transform(waveform)
        
        # loop over crops
        fbanks, corr, masks = [], [], []
        for wvfrm in waveform:
            fbank = self._wav2fbank(wvfrm)
    
            fbank = (fbank - self.data_mean) / (self.data_std * 2)
            fbank = fbank.unsqueeze(0)
            
            # clean crop 
            fbanks.append(fbank)
            
            # corrupted and masked
            audio_corr = fbank.detach().clone()
            audio_mask = torch.zeros_like(audio_corr)
            if self.drop_perc > 0:
                audio_corr, audio_mask = self.GMML_drop_rand_patches(audio_corr)
                
            corr.append(audio_corr)
            masks.append(audio_mask)
            
            
        return fbanks, corr, masks
