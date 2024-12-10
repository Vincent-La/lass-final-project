import os
import sys
import re
import json
from typing import Dict, List

import csv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import librosa
import lightning.pytorch as pl
from models.clap_encoder import CLAP_Encoder
import scipy.io.wavfile as wf


# sys.path.append('../dcase2024_task9_baseline/')
sys.path.append('../lass-final-project')

from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)


class DCASEEvaluatorAnalysis:
    def __init__(
        self,
        sampling_rate=16000,
        eval_indexes='lass_synthetic_validation.csv',
        audio_dir='lass_validation',
        output_dir = 'lass_validation_output',
        encoder_type = None
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
        
        self.eval_list = eval_list
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.encoder_type = encoder_type

    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        print(f'Evaluation on DCASE T9 synthetic validation set.')

       
        pl_model.eval()
        device = pl_model.device

        sisdrs_list = []
        sdris_list = []
        sdrs_list = []

        gather = []
        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):
                
                result = {}
                source, noise, snr, caption = eval_data
                result['caption'] = caption

                # TODO: source-noise-ratio ?
                snr = int(snr)

                source_path = os.path.join(self.audio_dir, f'{source}.wav')
                noise_path = os.path.join(self.audio_dir, f'{noise}.wav')
                result['source_path'] = source_path
                result['noise_path'] = noise_path
                
                source, fs = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                noise, fs = librosa.load(noise_path, sr=self.sampling_rate, mono=True)

                # create audio mixture with a specific SNR level
                source_power = np.mean(source ** 2)
                noise_power = np.mean(noise ** 2)
                desired_noise_power = source_power / (10 ** (snr / 10))
                scaling_factor = np.sqrt(desired_noise_power / noise_power)
                noise = noise * scaling_factor

                mixture = source + noise

                input_path = os.path.join(self.output_dir, f'{os.path.basename(source_path), os.path.basename(noise_path)}_input.wav')
                wf.write(input_path, self.sampling_rate, mixture)
                result['input_path'] = input_path

                # declipping if need be
                max_value = np.max(np.abs(mixture))
                if max_value > 1:
                    source *= 0.9 / max_value
                    mixture *= 0.9 / max_value

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)

                # TODO: modfiy ONE-PEACE to use a similar function here
                conditions = pl_model.query_encoder.get_query_embed(
                    modality='text',
                    text=[caption],
                    device=device 
                )
                    
                input_dict = {
                    "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                    "condition": conditions,
                }
                
                sep_segment = pl_model.ss_model(input_dict)["waveform"]
                # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                # sep_segment: (segment_samples,)


                # write out .wav file
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
                
                
                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source, est=sep_segment)

                # print(type(sdr))
                # print(type(sdri))
                # print(type(sisdr))

                sisdrs_list.append(sisdr)
                result['sisdr'] = float(sisdr)
                sdris_list.append(sdri)
                result['sdri'] = float(sdri)
                sdrs_list.append(sdr)
                result['sdr'] = float(sdr)

                gather.append(result)
                
        
        mean_sdri = np.mean(sdris_list)
        mean_sisdr = np.mean(sisdrs_list)
        mean_sdr = np.mean(sdrs_list)
        
        df_results = pd.DataFrame(gather)
        

        return df_results
    


def eval(evaluator, checkpoint_path, config_yaml='config/audiosep_base.yaml', device = "cuda"):
    configs = parse_yaml(config_yaml)

    # Load model
    query_encoder = CLAP_Encoder().eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    print(f'-------  Start Evaluation  -------')

    # evaluation 
    SISDR, SDRi, SDR = evaluator(pl_model)
    msg_clotho = "SDR: {:.3f}, SDRi: {:.3f}, SISDR: {:.3f}".format(SDR, SDRi, SISDR)
    print(msg_clotho)

    print('-------------------------  Done  ---------------------------')


if __name__ == '__main__':
    dcase_evaluator = DCASEEvaluator(
        sampling_rate=16000,
        eval_indexes='lass_synthetic_validation.csv',
        audio_dir='lass_validation',
    )

    # checkpoint_path='audiosep_16k,baseline,step=200000.ckpt'
    checkpoint_path='checkpoint/audiosep_baseline.ckpt'
    eval(dcase_evaluator, checkpoint_path, device = "cuda")
