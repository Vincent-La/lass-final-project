import torch
import torch.nn as nn
import random

import sys
sys.path.append('ONE-PEACE')
from one_peace.models import from_pretrained


class ONE_PEACE_Encoder(nn.Module):

    def __init__(
        self,
        pretrained_path='ONE-PEACE',  
    ):
        
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = from_pretrained(pretrained_path, device=self.device, dtype="float32")
        self.encoder_type = 'ONE-PEACE'

    def _get_audio_embed(self, batch):
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

    
    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=0.5, device=None):
        if modality == 'audio':
            embed = self._get_audio_embed(audio)
        elif modality == 'text':
            embed = self._get_text_embed(text)

        # NOTE: not really sure the motivation for this, prob need to reread the AudioSep paper
        elif modality == 'hybird':
            if random.random() > use_text_ratio:
                embed = self._get_audio_embed(audio)
            else:
                embed = self._get_text_embed(text)
        else:
            raise NotImplementedError("Please check flag 'training_modality'.")

        return embed.float()
