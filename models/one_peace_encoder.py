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



    def _get_audio_embed(self, batch):
        pass

    def _get_text_embed(self, batch):

        # TODO: look into this double_batch thing 
        # double_batch = False
        # if len(batch) == 1:
        #     batch = batch * 2
        #     double_batch = True
        with torch.no_grad():
            # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
            text_data = self.tokenizer(batch)
            embed = self.model.get_text_embedding(text_data)
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
