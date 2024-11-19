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


    
    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=0.5, device=None):
        if modality == 'audio':
            embed = self._get_audio_embed(audio)
        elif modality == 'text':
            embed = self._get_text_embed(text)
        elif modality == 'hybird':
            if random.random() > use_text_ratio:
                embed = self._get_audio_embed(audio)
            else:
                embed = self._get_text_embed(text)
        else:
            raise NotImplementedError("Please check flag 'training_modality'.")

        return embed.float()
