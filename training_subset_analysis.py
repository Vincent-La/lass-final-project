import os
import sys
from typing import Dict, List

import csv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import librosa
import lightning.pytorch as pl
from models.clap_encoder import CLAP_Encoder
import scipy.io.wavfile as wf
import random
import torchaudio
from data.waveform_mixers import dynamic_loudnorm


# sys.path.append('../dcase2024_task9_baseline/')
sys.path.append('../lass-final-project')

from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)


class TrainingSubsetAnalysis:
    def __init__(
        self,
        sampling_rate=16000,
        eval_indexes='lass_synthetic_validation.csv',
        audio_dir='lass_validation',
        output_dir = 'lass_validation_output',
        encoder_type = None,
        config_yaml = None
    ) -> None:
        r"""DCASE T9 LASS evaluator.

        Returns:
            None
        """

        assert encoder_type != None, 'need to initialize encoder_type'

        self.sampling_rate = sampling_rate

        with open(eval_indexes) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        
        self.eval_indexes = eval_indexes
        self.eval_list = eval_list
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.encoder_type = encoder_type

        configs = parse_yaml(config_yaml)
        
        lower_db = configs['data']['loudness_norm']['lower_db']
        higher_db = configs['data']['loudness_norm']['higher_db']
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

        max_clip_len = configs['data']['segment_seconds']
        self.max_length =  max_clip_len * sampling_rate


    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform
    
    def _get_audio_tensor(self, audio_path):
        audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)
        
         # drop short utterance
        if audio_data.size(1) < self.sampling_rate * 0.5:
            raise Exception(f'{audio_path} is too short, drop it ...') 
        
        # convert stero to single channel
        if audio_data.shape[0] > 1:
            # audio_data: [samples]
            audio_data = (audio_data[0] + audio_data[1]) / 2
        else:
            audio_data = audio_data.squeeze(0)

        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
        
        audio_data = audio_data.unsqueeze(0)

        # NOTE: this either randomly crops to make tensor of shape (1, self.max_length)
        # or pads with zeros
        audio_data = self._cut_or_randomcrop(audio_data)         

        return audio_data
    


    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        # print(f'Evaluation on DCASE T9 synthetic validation set.')
        print(f'Evaluating on {self.eval_indexes}')

       
        pl_model.eval()
        device = pl_model.device
        random.seed(1234)

        sisdrs_list = []
        sdris_list = []
        sdrs_list = []

        gather = []
        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):
                
                result = {}
                source, noise, caption = eval_data
                result['caption'] = caption
            
                source_path = os.path.join(self.audio_dir, f'{source}.wav')
                noise_path = os.path.join(self.audio_dir, f'{noise}.wav')
                result['source_path'] = source_path
                result['noise_path'] = noise_path
                
                source = self._get_audio_tensor(source_path)
                # print(source.shape)

                noise_segment = self._get_audio_tensor(noise_path)
                # print(noise_segment.shape)

                # NOTE: this happens essentially twice in training b/c the code is written to 
                # accomodate for >2 audios used in a mixture
                noise = torch.zeros_like(source)
                noise += dynamic_loudnorm(audio=noise_segment, reference=source, **self.loudness_param)
                noise = dynamic_loudnorm(audio=noise, reference=source, **self.loudness_param)

                mixture = source + noise

                # declipping if need be
                max_value = torch.max(torch.abs(mixture))
                if max_value > 1:
                    source *= 0.9 / max_value
                    mixture *= 0.9 / max_value

                mixture = mixture.unsqueeze(0).to(device)
                # print(f'mixture shape {mixture.shape}')


                source_ndarray = source.cpu().numpy()
                sdr_no_sep = calculate_sdr(ref=source_ndarray, est=mixture.cpu().numpy())

                conditions = pl_model.query_encoder.get_query_embed(
                    modality='text',
                    text=[caption],
                    device=device 
                )
                    
                input_dict = {
                    "mixture": mixture,
                    "condition": conditions,
                }
                
                
                sep_segment = pl_model.ss_model(input_dict)["waveform"]
                # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                # sep_segment: (segment_samples,)

                # NOTE: havent tested this block 
                if self.output_dir is not None:
                    # write out input mixture
                    input_path = os.path.join(self.output_dir, f'{os.path.basename(source_path), os.path.basename(noise_path)}_input.wav')
                    wf.write(input_path, self.sampling_rate, mixture)
                    result['input_path'] = input_path

                    # write out separated segment
                    output_path = os.path.join(self.output_dir, f'{os.path.basename(source_path), os.path.basename(noise_path)}_output.wav')
                    wf.write(output_path, self.sampling_rate, sep_segment)
                    result['output_path'] = output_path

                    # COMPUTE SIMILARITIES
                    if self.encoder_type == 'ONE-PEACE':
                        src_audios, audio_padding_masks = pl_model.query_encoder.model.process_audio([input_path])
                        audio_features = pl_model.query_encoder.model.extract_audio_features(src_audios, audio_padding_masks)
                        input_similarity = conditions @ audio_features.T
                        # similarities['input_similarity'] = input_similarity.squeeze(0).cpu().numpy()[0]
                        # print(f'Text Prompt - Mixed Audio Input Similarity: {input_similarity}')  
                        result['input_similarity'] = input_similarity.squeeze(0).cpu().numpy()[0]

                        src_audios, audio_padding_masks = pl_model.query_encoder.model.process_audio([output_path])
                        audio_features = pl_model.query_encoder.model.extract_audio_features(src_audios, audio_padding_masks)
                        output_similarity = conditions @ audio_features.T
                        result['output_similarity'] = output_similarity.squeeze(0).cpu().numpy()[0]
                        # similarities['output_similarity'] = output_similarity.squeeze(0).cpu().numpy()[0]

                        src_audios, audio_padding_masks = pl_model.query_encoder.model.process_audio([source_path])
                        audio_features = pl_model.query_encoder.model.extract_audio_features(src_audios, audio_padding_masks)
                        target_similarity = conditions @ audio_features.T
                        result['target_similarity'] = target_similarity.squeeze(0).cpu().numpy()[0]
                    
                    elif self.encoder_type == 'CLAP':
                        
                        audio_features = pl_model.query_encoder._get_audio_embed(input_dict['mixture'][0])
                        # print('mixture shape: input_dict shape {}'.format(input_dict['mixture'][0].shape))
                        input_similarity = conditions @ audio_features.T
                        result['input_similarity'] = input_similarity.squeeze(0).cpu().numpy()[0]

                        sep_segment_tensor = torch.tensor(sep_segment).unsqueeze(0).to(device)
                        # print(f'sep_segment: {sep_segment_tensor.shape}')
                        audio_features = pl_model.query_encoder._get_audio_embed(sep_segment_tensor)
                        output_similarity = conditions @ audio_features.T
                        result['output_similarity'] = output_similarity.squeeze(0).cpu().numpy()[0]

                        source_tensor = torch.tensor(source).unsqueeze(0).to(device)
                        # print(f'source shape:{source_tensor.shape}')
                        audio_features = pl_model.query_encoder._get_audio_embed(source_tensor)
                        target_similarity = conditions @ audio_features.T
                        result['target_similarity'] = target_similarity.squeeze(0).cpu().numpy()[0]
                
                
                sdr = calculate_sdr(ref=source_ndarray, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source_ndarray, est=sep_segment)

                sisdrs_list.append(sisdr)
                result['sisdr'] = float(sisdr)
                sdris_list.append(sdri)
                result['sdri'] = float(sdri)
                sdrs_list.append(sdr)
                result['sdr'] = float(sdr)

                gather.append(result)

        # mean_sdri = np.mean(sdris_list)
        # mean_sisdr = np.mean(sisdrs_list)
        # mean_sdr = np.mean(sdrs_list)
        
        df_results = pd.DataFrame(gather)
        

        return df_results
    
