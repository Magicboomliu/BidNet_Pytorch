from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import copy
import sys
sys.path.append("../..")

from models.BidNet.attention import MultiheadAttentionRelative

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
layer_idx = 0




class STTM_Complex(nn.Module):
    def __init__(self,hidden_dim=128,num_head=4,num_attn_layers=4):
        super().__init__()
        
        self_attn_later = TransformerSelfAttnLayer(hidden_dim,nhead=num_head)
        self.self_attn_layers = get_clones(self_attn_later, num_attn_layers)
        
        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim,nhead=num_head)
        self.cross_attn_layers = get_clones(cross_attn_layer,num_attn_layers)
        
        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.nhead = num_head
        self.num_attn_layers = num_attn_layers

    def _alternating_attn(self, feat: torch.Tensor, pos_enc: torch.Tensor, pos_indexes: Tensor, hn: int):
        """
        Alternate self and cross attention with gradient checkpointing to save memory
        :param feat: image feature concatenated from left and right, [W,2HN,C]
        :param pos_enc: positional encoding, [W,HN,C]
        :param pos_indexes: indexes to slice positional encoding, [W,HN,C]
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        """

        global layer_idx
        # alternating
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            layer_idx = idx

            # checkpoint self attn
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn

            feat = checkpoint(create_custom_self_attn(self_attn), feat, pos_enc, pos_indexes)

            # add a flag for last layer of cross attention
            if idx == self.num_attn_layers - 1:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs)

                    return custom_cross_attn
            else:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs)

                    return custom_cross_attn

            feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:], pos_enc,
                                           pos_indexes)
        layer_idx = 0
        
        return feat


    
    def forward(self,feat_left,feat_right,pos_enc):
        
        bs, c, hn, w = feat_left.shape
        
        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # CxWxHxN -> CxWxHN -> WxHNxC
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None

        # concatenate left and right features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        # compute attention
        feat = self._alternating_attn(feat, pos_enc, pos_indexes, hn)

        return feat



class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.self_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, feat: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None):
        """
        :param feat: image feature [W,2HN,C]
        :param pos: pos encoding [2W-1,HN,C]
        :param pos_indexes: indexes to slice pos encoding [W,W]
        :return: updated image feature
        """
        feat2 = self.norm1(feat)

        # torch.save(feat2, 'feat_self_attn_input_' + str(layer_idx) + '.dat')

        feat2, attn_weight, _ = self.self_attn(query=feat2, key=feat2, value=feat2, pos_enc=pos,
                                               pos_indexes=pos_indexes)

        # addition: former is the 
        feat = feat + feat2

        return feat
    


class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, feat_left: Tensor, feat_right: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None):
        """
        :param feat_left: left image feature, [W,HN,C]
        :param feat_right: right image feature, [W,HN,C]
        :param pos: pos encoding, [2W-1,HN,C]
        :param pos_indexes: indexes to slicer pos encoding [W,W]
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)

        # torch.save(torch.cat([feat_left_2, feat_right_2], dim=1), 'feat_cross_attn_input_' + str(layer_idx) + '.dat')

        # update right features
        if pos is not None:
            pos_flipped = torch.flip(pos, [0])
        else:
            pos_flipped = pos
        feat_right_2 = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped,
                                       pos_indexes=pos_indexes)[0]

        feat_right = feat_right + feat_right_2

        # # update left features
        # # use attn mask for last layer
        # if last_layer:
        #     w = feat_left_2.size(0)
        #     attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)  # generate attn mask
        # else:
        #     attn_mask = None
        attn_mask = None

        # normalize again the updated right features
        feat_right_2 = self.norm2(feat_right)
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2,
                                                             attn_mask=attn_mask, pos_enc=pos,
                                                             pos_indexes=pos_indexes)

        # torch.save(attn_weight, 'cross_attn_' + str(layer_idx) + '.dat')

        feat_left = feat_left + feat_left_2

        # concat features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        return feat, raw_attn

if __name__=="__main__":
    left_feat = torch.randn(1,128,40,80).cuda()
    right_feat = torch.randn(1,128,40,80).cuda()
    
    
    sttm_complex = STTM_Complex(hidden_dim=128,num_attn_layers=4,num_head=4).cuda()
    
    feat = sttm_complex(left_feat,right_feat,None)

    print(feat.shape)
