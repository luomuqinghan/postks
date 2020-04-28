#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/decoders/hgfu_rnn_decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.misc import sequence_mask

class DraftDecoder(nn.Module):
    """
    A HGFU GRU recurrent neural network decoder.
    Paper <<Towards Implicit Content-Introducing for Generative Short-Text
            Conversation Systems>>
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 tgt_field,
                 max_length,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0,
                 use_gpu=False):
        super(DraftDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout
        self.max_length = max_length
        self.use_gpu = use_gpu

        self.BOS = tgt_field.stoi[tgt_field.bos_token]
        self.EOS = tgt_field.stoi[tgt_field.eos_token]


        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size
        self.cue_input_size = self.hidden_size

        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size
            self.cue_input_size += self.feature_size

        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.rnn_input_size += self.memory_size
            self.cue_input_size += self.memory_size
            self.out_input_size += self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.Softmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         feature=None,
                         src_enc_inputs=None,
                         src_enc_outputs=None,
                         src_mask=None,
                         knowledge=None):
        """
        initialize_state
        """
        if self.feature_size is not None:
            assert feature is not None

        if self.attn_mode is not None:
            assert src_enc_outputs is not None

        if src_enc_inputs is not None and src_mask is None:
            max_len = src_enc_inputs[0].size(1)
            src_mask = sequence_mask(src_enc_inputs[1], max_len).eq(0)

        init_state = DecoderState(
            hidden=hidden,
            feature=feature,
            src_enc_inputs=src_enc_inputs[0],
            src_enc_outputs=src_enc_outputs,
            src_mask=src_mask,
            knowledge=knowledge)
        return init_state

    def decode(self, input, state, is_training=False):
        """
        decode
        """
        hidden = state.hidden
        input_list = []
        output_list = []
        output = Pack()

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        input_list.append(input)

        if self.feature_size is not None:
            feature = state.feature.unsqueeze(1)
            input_list.append(feature)

        if self.attn_mode is not None:
            weighted_context, attn = self.attention(query=hidden[-1].unsqueeze(1),
                                                    memory=state.src_enc_outputs,
                                                    mask=state.src_mask)
            input_list.append(weighted_context)
            output_list.append(weighted_context)
            output.add(attn=attn)

        rnn_input = torch.cat(input_list, dim=-1)
        _, new_hidden = self.rnn(rnn_input, hidden)

        output_list.append(new_hidden[-1].unsqueeze(1))
        out_input = torch.cat(output_list, dim=-1)

        prob = self.output_layer(out_input)
        log_prob = torch.log(prob+1e-10)

        state.hidden = new_hidden
        return log_prob, state, output

    def forward(self, inputs, state, is_training=False):
        """
        forward
        """
        logits = []
        predicts = []
        dec_outputs = []
        if inputs is None:
            dec_length = self.max_length
        else:
            dec_length = inputs.size(1)
        for i in range(dec_length):
            if i==0 or is_training:
                input = inputs[:,i]
            else:
                input = predict.squeeze(1)
            log_prob, state, _ = self.decode(input, state, is_training=is_training)
            predict = log_prob.argmax(2)
            logits.append(log_prob)
            predicts.append(predict)
            dec_outputs.append(state.hidden.transpose(0,1))
        logits = torch.cat(logits,dim=1)
        predicts= torch.cat(predicts,dim=1)
        dec_outputs = torch.cat(dec_outputs,dim=1)

        length = []
        for x in predicts.tolist():
            try:
                length.append(x.index(self.EOS))
            except ValueError:
                length.append(self.max_length-1)
        length = torch.tensor(length,dtype=torch.int32)
        if self.use_gpu:
            length = length.cuda()
        return logits, predicts, dec_outputs, length

class RNNDecoder(nn.Module):
    """
    A HGFU GRU recurrent neural network decoder.
    Paper <<Towards Implicit Content-Introducing for Generative Short-Text
            Conversation Systems>>
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 copy=False,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0,
                 concat=False):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout
        self.concat = concat
        self.copy = copy

        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size
        self.cue_input_size = self.hidden_size

        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size
            self.cue_input_size += self.feature_size

        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.rnn_input_size += self.memory_size
            self.cue_input_size += self.memory_size
            self.out_input_size += self.memory_size

        self.rnn_input_size += self.hidden_size
        self.cue_input_size += self.hidden_size
        self.out_input_size += self.hidden_size

        self.cue_attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        self.cue_rnn = nn.GRU(input_size=self.cue_input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=self.dropout if self.num_layers > 1 else 0,
                              batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        if self.concat:
            self.fc4 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        else:
            self.fc4 = nn.Linear(self.hidden_size * 3, 3)
        self.fc5 = nn.Linear(self.input_size+self.hidden_size*3, self.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.Softmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         feature=None,
                         src_enc_inputs=None,
                         src_enc_outputs=None,
                         src_mask=None,
                         cue_enc_inputs=None,
                         cue_enc_outputs=None,
                         cue_mask=None,
                         selected_cue_memory=None,
                         selected_cue_length=None,
                         selected_cue_mask=None,
                         knowledge=None,
                         draft_dec_outputs=None,
                         draft_length=None,
                         draft_mask=None,
                         draft_logits=None):
        """
        initialize_state
        """
        if self.feature_size is not None:
            assert feature is not None

        if self.attn_mode is not None:
            assert src_enc_outputs is not None

        if src_enc_inputs is not None and src_mask is None:
            max_len = src_enc_inputs[0].size(1)
            src_mask = sequence_mask(src_enc_inputs[1], max_len).eq(0)

        if cue_enc_inputs is not None and cue_mask is None:
            max_len = cue_enc_inputs[0].size(2)
            cue_mask = sequence_mask(cue_enc_inputs[1], max_len).eq(0)

        if selected_cue_length is not None and selected_cue_mask is None:
            max_len = selected_cue_memory.size(1)
            selected_cue_mask = sequence_mask(selected_cue_length, max_len).eq(0)

        if draft_length is not None and draft_mask is None:
            max_len = draft_dec_outputs.size(1)
            draft_mask = sequence_mask(draft_length, max_len).eq(0)

        init_state = DecoderState(
            hidden=hidden,
            feature=feature,
            src_enc_inputs=src_enc_inputs[0] if src_enc_inputs else None,
            src_enc_outputs=src_enc_outputs,
            src_mask=src_mask,
            cue_enc_outputs=cue_enc_outputs,
            cue_enc_inputs=cue_enc_inputs[0] if cue_enc_inputs else None,
            cue_mask=cue_mask,
            selected_cue_memory=selected_cue_memory,
            selected_cue_mask=selected_cue_mask,
            knowledge=knowledge,
            draft_dec_outputs=draft_dec_outputs,
            draft_mask=draft_mask,
            draft_logits=draft_logits)
        return init_state

    def decode(self, input, state, is_training=False):
        """
        decode
        """
        hidden = state.hidden
        rnn_input_list = []
        cue_input_list = []
        out_input_list = []
        local_cue_input_list = []
        output = Pack()

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        rnn_input_list.append(input)
        cue_input_list.append(state.knowledge)

        if self.feature_size is not None:
            feature = state.feature.unsqueeze(1)
            rnn_input_list.append(feature)
            cue_input_list.append(feature)

        if self.attn_mode is not None:
            weighted_context, attn = self.attention(query=hidden[-1].unsqueeze(1),
                                                    memory=state.src_enc_outputs,
                                                    mask=state.src_mask)
            rnn_input_list.append(weighted_context)
            cue_input_list.append(weighted_context)
            out_input_list.append(weighted_context)
            local_cue_input_list.append(weighted_context)
            output.add(attn=attn)

        draft_context, draft_attn = self.attention(query=hidden[-1].unsqueeze(1),
                                                   memory=state.draft_dec_outputs,
                                                   mask=state.draft_mask)
        rnn_input_list.append(draft_context)
        cue_input_list.append(draft_context)
        out_input_list.append(draft_context)
        local_cue_input_list.append(draft_context)
        output.add(draft_attn=draft_attn)

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        _, rnn_hidden = self.rnn(rnn_input, hidden)

        cue_input = torch.cat(cue_input_list, dim=-1)
        cue_output, cue_hidden = self.cue_rnn(cue_input, hidden)

        local_knowledge, local_cue_attn = self.cue_attention(query=rnn_hidden[-1].unsqueeze(1),
                                                             memory=state.selected_cue_memory,
                                                             mask=state.selected_cue_mask)
        output.add(local_cue_attn=local_cue_attn)
        local_cue_input_list.append(local_knowledge)
        local_cue_input = torch.cat(local_cue_input_list, dim=-1)
        local_cue_output, local_cue_hidden = self.cue_rnn(local_cue_input, hidden)
        h_y = self.tanh(self.fc1(rnn_hidden))
        h_cue = self.tanh(self.fc2(cue_hidden))
        local_h_cue = self.tanh(self.fc3(local_cue_hidden))
        if self.concat:
            new_hidden = self.fc4(torch.cat([h_y, h_cue, local_h_cue], dim=-1))
        else:
            k = F.softmax(self.fc4(torch.cat([h_y, h_cue, local_h_cue], dim=-1)), dim=-1)
            output.add(k=k)
            k = k.split(1, dim=-1)
            new_hidden = k[0] * h_y + k[1] * h_cue + k[2] * local_h_cue
        out_input_list.append(new_hidden[-1].unsqueeze(1))
        out_input = torch.cat(out_input_list, dim=-1)

        prob = self.output_layer(out_input)
        if self.copy:
            batch_size, sent_num, sent, _ = state.cue_enc_outputs.size()
            _, knowledge_attn = self.attention(query=hidden[-1].unsqueeze(1).repeat(sent_num,1,1),
                                               memory=state.cue_enc_outputs.view(batch_size*sent_num,sent,-1),
                                               mask=state.cue_mask.view(batch_size*sent_num,-1))
            knowledge_attn = state.cue_attn.unsqueeze(2) * knowledge_attn.squeeze(1).view(batch_size,sent_num,-1)
            knowledge_attn = knowledge_attn.view(batch_size,1,-1)
            output.add(knowledge_attn=knowledge_attn)
            p = F.softmax(self.fc5(torch.cat([input,new_hidden[-1].unsqueeze(1),weighted_context,draft_context,state.knowledge],dim=-1)),dim=-1)
            output.add(p=p)
            p = p.split(1,dim=2)
            prob = (p[0]*prob).scatter_add(2, state.src_enc_inputs.unsqueeze(1), p[1]*attn)
            prob = prob.scatter_add(2, state.cue_enc_inputs.view(batch_size,1,-1), p[2]*knowledge_attn)
        log_prob = torch.log(prob+1e-10)

        state.hidden = new_hidden
        return log_prob, state, output

    def forward(self, inputs, state):
        """
        forward
        """
        inputs, lengths = inputs
        batch_size, max_len = inputs.size()

        log_probs = inputs.new_zeros(
            size=(batch_size, max_len, self.output_size),
            dtype=torch.float)

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            log_prob, valid_state, _ = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:,:num_valid] = valid_state.hidden
            log_probs[:num_valid, i] = log_prob.squeeze(1)

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        log_probs = log_probs.index_select(0, inv_indices)
        draft_logits = state.draft_logits
        return log_probs, draft_logits, state
