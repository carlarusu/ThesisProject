# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Based on Pytorch's torch.nn.Transformer and Facebook's DETR.
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=0.0):
    # q           (B * nhead, tgt_len, head_dim)      [16, 100, 32]
    # kv          (B * nhead, src_len, head_dim)      [16, 672, 32]
    # attn_mask   (B * nhead, 1 or tgt_len, src_len)  [16, 1, 672]
    # out         (B * nhead, tgt_len, head_dim)      [16, 100, 32]
    
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    # [16, 100, 672]
    attn = torch.bmm(q, k.transpose(-2, -1))
    
    # attn mask will set -inf to attn positions that must be masked
    # mask is 0 by default so no masking takes place
    if attn_mask is not None:
        attn += attn_mask
    # [16, 100, 672]
        
    attn = F.softmax(attn, dim=-1)
    
    if dropout > 0.0:
        attn = F.dropout(attn, p=dropout)
    # [16, 100, 672]
        
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    # [16, 100, 672] * [16, 672, 32] -> [16, 100, 32]
    output = torch.bmm(attn, v)
    
    return output, attn
    

def _in_projection_packed(q, k, v, w, b=None):
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
            # q:        (B, *, in_features)         -> (..., E)
            # w:        (out_features, in_features) -> (E * 3, E)
            # b:        (out_features)              -> (E * 3)
            # lin_out:  (B, *, out_features)        -> (..., E * 3)
            # chunk_out:                            -> 3 * (..., E)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            # will concat q_out with k_out v_out
            #                            |
            #                            V
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        
        self.head_dim = d_model // nhead
        assert (self.head_dim * nhead == d_model), "d_model % nhead != 0"
        
        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model, d_model)))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        #                     Enc             Dec tgt         Dec mem
        # query, key, value:  [672, 2, 256]   [100, 2, 256]   [100, 2, 256], [672, 2, 256], [672, 2, 256]
        # attn_mask:          None            None            None
        # key_padding_mask:   [2, 672]        None            [2, 672]
        # output:             [672, 2, 256]   [100, 2, 256]   [100, 2, 256]
        
        # key_padding_mask: used to mask out padding positions after the end 
        #                   of the input sequence. It depends on the longest
        #                   sequence in the batch. Shape (B, src seq length)
        
        # attn_mask:        used in decoders to prevent attention to future
        #                   positions using a triangle mask. 
        #                   2D shape: (tgt seq length, src seq length)
        #                   3D shape: (B*nhead, tgt seq length, src seq length)
        
        # q:                (tgt seq length, B, C)
        # kv:               (src seq length, B, C)
        # out:  
        #   - attn_output           (tgt seq length, B, C)
        #   - attn_output_weights   (B, tgt seq length, C)
        
        tgt_len, batch_size, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        assert (embed_dim == self.d_model), f"expected hidden dim = {self.d_model}, but got {embed_dim}"
        assert (key.shape == value.shape), f"key shape {key.shape} does not match value shape {value.shape}"
        
        # compute in-projection
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        
        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, "wrong attn_mask type"
            
            if attn_mask.dim() == 2:
                assert (tgt_len, src_len) == attn_mask.shape, "wrong attn_mask shape"
                attn_mask = attn_mask.unsqueeze(0)
                # add artificial batch_size=1
            elif attn_mask.dim() == 3:
                assert (batch_size * self.nhead, tgt_len, src_len) == attn_mask.shape, "wrong attn_mask shape"
            else:
                assert False, "wrong attn_mask shape"
            
        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)
        
        # reshape q, k, v for multihead attention and make em batch first
        # q:    (tgt_len, B, C)->(tgt_len, B, nhead * head_dim)->
        #       (tgt_len, B * nhead, head_dim)->(B * nhead, tgt_len, head_dim)
        q = q.contiguous().view(tgt_len, batch_size * self.nhead, self.head_dim).transpose(0, 1)
        
        # kv:   (src_len, B, C)->(src_len, B, nhead * head_dim)->
        #       (src_len, B * nhead, head_dim)->(B * nhead, src_len, head_dim)
        # .view(-1, ...) lets python compute the first dim based on the other dims specified
        k = k.contiguous().view(-1, batch_size * self.nhead, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * self.nhead, self.head_dim).transpose(0, 1)
        
        # update source sequence length after adjustments
        src_len = k.shape[1]
        
        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, src_len), "wrong key_padding_mask shape"
            
            #[2, 672]
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
            #[2, 1, 1, 672]
            key_padding_mask = key_padding_mask.expand(-1, self.nhead, -1, -1)
            #[2, 8, 1, 672]
            # -1 means not changing the size of that dimension
            key_padding_mask = key_padding_mask.reshape(batch_size * self.nhead, 1, src_len)
            #[16, 1, 672]
            
            # attn_mask if not None: [16, 100, 672]
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
        
        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, self.dropout)
        #attn_output            [16, 100, 32]
        #attn_output_weights    [16, 100, 672]
        
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        #attn_output [16, 100, 32]->[100, 16, 32]->[100, 2, 256]
        
        #attn_output            [100, 2, 256]
        #self.out_proj.weight   [256, 256]
        #self.out_proj.bias     [256]
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        #attn_output [100, 2, 256]
        
        return attn_output, attn_output_weights


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, activation="relu", 
                 model_type='detr', class_emb=0):
        super().__init__()
        
        self.class_emb = class_emb

        self.model_type = model_type
        if self.model_type == 'detr'
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation)

            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # src.shape:            [2, 256, 32, 21]    [batch_size, hidden_dim, h, w]
        # mask.shape:           [2, 32, 21]         [batch_size, h, w]
        # query_embed.shape:    [100, 256]          [tgt_len, hidden_dim]
        # pos_embed.shape:      [2, 256, 32, 21]    [batch_size, hidden_dim, h, w]
    
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        
        if self.class_emb:
            class_embed = torch.load('models/class_embeddings_db')
            class_embed = class_embed.repeat(1, bs, 1).contiguous()
            
            if query_embed.get_device() != -1:
                class_embed = class_embed.cuda()
                
            query_embed = query_embed + class_embed
        
        # src.shape:            [32*21, 2, 256]
        # mask.shape:           [2, 32*21]
        # query_embed.shape:    [100, 2, 256]
        # pos_embed.shape:      [32*21, 2, 256]
        
        tgt = torch.zeros_like(query_embed)
        
        if self.model_type == 'detr':
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src
        
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        # tgt.shape:            [100, 2, 256]
        # memory.shape:         [672, 2, 256]->[2, 256, 672]->[2, 256, 32, 21]
        # hs.shape:             [3, 100, 2, 256]->[3, 2, 100, 256]
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        
        # src.shape:                    [672, 2, 256]
        # mask:                         None
        # src_key_padding_mask.shape:   [2, 672]
        # pos.shape:                    [672, 2, 256]
        
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            # output.shape:             [672, 2, 256]

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        # tgt.shape:                        [100, 2, 256]
        # memory.shape:                     [672, 2, 256]
        # tgt_mask:                         None
        # memory_mask:                      None
        # tgt_key_padding_mask:             None
        # memory_key_padding_mask.shape:    [2, 672]
        # pos.shape:                        [672, 2, 256]
        # query_pos.shape:                  [100, 2, 256]
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # output.shape:                 [100, 2, 256]
            intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            intermediate.pop()
            intermediate.append(output)
        
        return torch.stack(intermediate)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):        
        # src.shape:                    [672, 2, 256]
        # src_mask:                     None
        # src_key_padding_mask.shape:   [2, 672]
        # pos.shape:                    [672, 2, 256]
        
        q = k = self.with_pos_embed(src, pos)
        
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        #[672, 2, 256]
        src = src + self.dropout1(src2)
        #[672, 2, 256]
        src = self.norm1(src)
        #[672, 2, 256]
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        #[672, 2, 256]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        #[672, 2, 256]
        
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # tgt.shape:                        [100, 2, 256]
        # memory.shape:                     [672, 2, 256]
        # tgt_mask:                         None
        # memory_mask:                      None
        # tgt_key_padding_mask:             None
        # memory_key_padding_mask.shape:    [2, 672]
        # pos.shape:                        [672, 2, 256]
        # query_pos.shape:                  [100, 2, 256]
        
        q = k = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        #[100, 2, 256]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        #[100, 2, 256]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        #[100, 2, 256]
        
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    # args.hidden_dim = 256
    # args.dropout =.1
    # args.nheads = 8
    # args.dim_feedforward = 2048
    # args.enc_layers = 3
    # args.dec_layers = 3
    
    if args.vit224 or args.vit384 or args.vit640:
        model_type = 'vit'
    else:
        model_type = 'detr'
        
    if args.class_emb:
        class_emb = 1
    else: 
        class_emb = 0
    
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        model_type=model_type,
        class_emb=class_emb
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
