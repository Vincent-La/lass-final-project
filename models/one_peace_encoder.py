import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import sys
sys.path.append('ONE-PEACE')
from one_peace.models import from_pretrained


class ONE_PEACE_Encoder(nn.Module):

    def __init__(
        self,
        pretrained_path='ONE-PEACE', 
        model_type = 'one_peace_retrieval'
    ):
        
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = from_pretrained(pretrained_path, 
                                     model_type=model_type,
                                     device=self.device, 
                                     dtype="float32")
        
        self.encoder_type = 'ONE-PEACE'


    # Adapted from ONE-PEACE hub interface because it only handles paths to audio files,
    # not tensor batches see: ONE-PEACE/one_peace/models/one_peace/hub_interface.py
    # NOTE: both ONE-PEACE and default AudioSep setup use a sampling rate of 16000

    # TODO:
    def _process_audio(self, batch):
        
        feats = batch

        # actual ONE-PEACE pytorch model
        model = self.model.model

        # with torch.no_grad():
        #     feats = F.layer_norm(feats, feats.shape)
        # if feats.size(-1) > curr_sample_rate * 15:
        #     start_idx = 0
        #     end_idx = start_idx + curr_sample_rate * 15
        #     feats = feats[start_idx:end_idx]
        # if feats.size(-1) < curr_sample_rate * 1:
        #     feats = feats.repeat(math.ceil(curr_sample_rate * 1 / feats.size(-1)))
        #     feats = feats[:curr_sample_rate * 1]

        # T = self._get_mask_indices_dims(feats.size(-1), self.feature_encoder_spec)
        # audio_padding_mask = torch.zeros(T + 1).bool()
        # feats_list.append(feats)
        # audio_padding_mask_list.append(audio_padding_mask)
        # src_audios = collate_tokens(feats_list, pad_idx=0).to(self.device)
        # src_audios = self.cast_data_dtype(src_audios)
        # audio_padding_masks = collate_tokens(audio_padding_mask_list, pad_idx=True).to(self.device)

        # return src_audios, audio_padding_masks


    # TODO:
    def _get_audio_embed(self, batch):
        
        with torch.no_grad():
            pass



    '''
        batch: List[str]
    '''
    def _get_text_embed(self, batch):

        # TODO: look into this double_batch thing 
        # double_batch = False
        # if len(batch) == 1:
        #     batch = batch * 2
        #     double_batch = True
        with torch.no_grad():
            src_tokens = self.model.process_text(batch)
            embed = self.model.extract_text_features(src_tokens)
        # if double_batch:
        #     embed = embed[0].unsqueeze(0)
        
        return embed.detach()

    
    # NOTE: text queries through encoder seem to be with no_grad() 
    # so ONE-PEACE model shouldnt have a gradient?
    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=0.5, device=None):
        if modality == 'audio':
            embed = self._get_audio_embed(audio)
        elif modality == 'text':
            embed = self._get_text_embed(text)

        # NOTE: not really sure the motivation for this, prob need to reread the AudioSep paper
        elif modality == 'hybrid':
            if random.random() > use_text_ratio:
                embed = self._get_audio_embed(audio)
            else:
                embed = self._get_text_embed(text)
        else:
            raise NotImplementedError("Please check flag 'training_modality'.")

        return embed.float()
