#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/knowledge_seq2seq.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.decoders.lka_de_decoder import DraftDecoder, RNNDecoder
from source.utils.criterions import NLLLoss
from source.utils.misc import Pack
from source.utils.metrics import accuracy
from source.utils.metrics import attn_accuracy
from source.utils.metrics import perplexity
from source.modules.attention import Attention

class LkaDeSeq2seq(BaseModel):
    """
    KnowledgeSeq2Seq
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, padding_idx=None,
                 num_layers=1, bidirectional=True, attn_mode="mlp", attn_hidden_size=None, 
                 with_bridge=False, tie_embedding=False, dropout=0.0, use_gpu=False, use_bow=False,
                 use_kd=False, use_dssm=False, use_posterior=False, weight_control=False, 
                 use_pg=False, use_gs=False, gs_tau=1.0, concat=False, copy=False, kl_annealing=False, pretrain_epoch=0,
                 tgt_field=None,max_draft_len=None,pretrain_lr=None,lr=None):
        super(LkaDeSeq2seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.attn_hidden_size = attn_hidden_size
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_bow = use_bow
        self.use_dssm = use_dssm
        self.weight_control = weight_control
        self.use_kd = use_kd
        self.use_pg = use_pg
        self.use_gs = use_gs
        self.gs_tau = gs_tau
        self.use_posterior = use_posterior
        self.pretrain_epoch = pretrain_epoch
        self.baseline = 0
        self.copy = copy
        self.kl_annealing = kl_annealing
        self.pretrain_lr = pretrain_lr
        self.lr = lr

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size, padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  embedder=enc_embedder, num_layers=self.num_layers,
                                  bidirectional=self.bidirectional, dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
            knowledge_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size, padding_idx=self.padding_idx)
            knowledge_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                          embedding_dim=self.embed_size,
                                          padding_idx=self.padding_idx)

        self.knowledge_encoder = RNNEncoder(input_size=self.embed_size,
                                            hidden_size=self.hidden_size,
                                            embedder=knowledge_embedder,
                                            num_layers=self.num_layers,
                                            bidirectional=self.bidirectional,
                                            dropout=self.dropout)

        self.prior_attention = Attention(query_size=self.hidden_size,
                                         memory_size=self.hidden_size,
                                         hidden_size=self.hidden_size,
                                         mode="dot")

        self.posterior_attention = Attention(query_size=self.hidden_size,
                                             memory_size=self.hidden_size,
                                             hidden_size=self.hidden_size,
                                             mode="dot")

        self.draft_decoder = DraftDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                          output_size=self.tgt_vocab_size,
                                          tgt_field=tgt_field,
                                          max_length=max_draft_len,
                                          embedder=dec_embedder,
                                          num_layers=self.num_layers, attn_mode=self.attn_mode,
                                          memory_size=self.hidden_size, feature_size=None,
                                          dropout=self.dropout,
                                          use_gpu=use_gpu)

        self.decoder = RNNDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size, embedder=dec_embedder,
                                  num_layers=self.num_layers, attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size, feature_size=None,
                                  dropout=self.dropout, concat=concat)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        if self.use_bow:
            self.bow_output_layer = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_size, out_features=self.tgt_vocab_size),
                    nn.LogSoftmax(dim=-1))

        if self.use_dssm:
            self.dssm_project = nn.Linear(in_features=self.hidden_size,
                                          out_features=self.hidden_size)
            self.mse_loss = torch.nn.MSELoss(reduction='mean')

        if self.use_kd:
            self.knowledge_dropout = nn.Dropout()

        if self.padding_idx is not None:
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')
        self.kl_loss = torch.nn.KLDivLoss(size_average=True)

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, hidden=None, is_training=False):
        """
        encode
        """
        outputs = Pack()
        src_enc_inputs = inputs.src[0][:, 1:-1], inputs.src[1]-2
        src_enc_outputs, src_enc_hidden = self.encoder(src_enc_inputs, hidden)

        keywords_enc_inputs = inputs.keywords[0][:, 1:-1], inputs.keywords[1] - 2
        if self.with_bridge:
            src_enc_hidden = self.bridge(src_enc_hidden)

        # knowledge
        batch_size, sent_num, sent  = inputs.cue[0].size()
        tmp_len = inputs.cue[1]
        tmp_len[tmp_len > 0] -= 2
        cue_enc_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], tmp_len.view(-1)
        cue_enc_outputs, cue_enc_hidden = self.knowledge_encoder(cue_enc_inputs, hidden)
        cue_enc_outputs = cue_enc_outputs.view(batch_size, sent_num, sent-2, -1)
        cue_enc_hidden = cue_enc_hidden[-1].view(batch_size, sent_num, -1)
        # Attention
        weighted_cue, cue_attn = self.prior_attention(query=src_enc_hidden[-1].unsqueeze(1),
                                                      memory=cue_enc_hidden,
                                                      mask=inputs.cue[1].eq(0))
        cue_attn = cue_attn.squeeze(1)
        outputs.add(prior_attn=cue_attn)

        if self.use_posterior:
            _, keywords_enc_hidden = self.knowledge_encoder(keywords_enc_inputs, hidden)
            posterior_weighted_cue, posterior_attn = self.posterior_attention(
                # P(z|u,r)
                # query=torch.cat([dec_init_hidden[-1], tgt_enc_hidden[-1]], dim=-1).unsqueeze(1)
                # P(z|r)
                query=keywords_enc_hidden[-1].unsqueeze(1),
                memory=cue_enc_hidden,
                mask=inputs.cue[1].eq(0))
            posterior_attn = posterior_attn.squeeze(1)
            outputs.add(posterior_attn=posterior_attn)
            # Gumbel Softmax
            if self.use_gs:
                gumbel_attn = F.gumbel_softmax(torch.log(posterior_attn + 1e-10), self.gs_tau, hard=True)
                outputs.add(gumbel_attn=gumbel_attn)
                knowledge = torch.bmm(gumbel_attn.unsqueeze(1), cue_enc_hidden)
                indexs = gumbel_attn.max(-1)[1]
                cue_memory = torch.sum(gumbel_attn.view(batch_size,sent_num,1,1)*cue_enc_outputs, dim=1)
            else:
                knowledge = posterior_weighted_cue
                indexs = posterior_attn.max(dim=1)[1]
                cue_memory = torch.sum(posterior_attn.view(batch_size,sent_num,1,1)*cue_enc_outputs, dim=1)
            if self.use_bow:
                bow_logits = self.bow_output_layer(knowledge)
                outputs.add(bow_logits=bow_logits)
        elif is_training:
            if self.use_gs:
                gumbel_attn = F.gumbel_softmax(torch.log(cue_attn + 1e-10), self.gs_tau, hard=True)
                knowledge = torch.bmm(gumbel_attn.unsqueeze(1), cue_enc_hidden)
                indexs = gumbel_attn.max(-1)[1]
                cue_memory = torch.sum(gumbel_attn.view(batch_size,sent_num,1,1)*cue_enc_outputs, dim=1)
            else:
                indexs = cue_attn.max(dim=1)[1]
                knowledge = weighted_cue
                cue_memory = torch.sum(cue_attn.view(batch_size,sent_num,1,1)*cue_enc_outputs, dim=1)
        else:
            indexs = cue_attn.max(dim=1)[1]
            if self.use_gs:
                knowledge = cue_enc_hidden.gather(1, indexs.view(-1, 1, 1).repeat(1, 1, cue_enc_hidden.size(-1)))
                cue_memory = cue_enc_outputs.gather(1, indexs.view(-1, 1, 1, 1).repeat(1, 1, cue_enc_outputs.size(2), cue_enc_outputs.size(3))).squeeze(1)
            else:
                knowledge = weighted_cue
                cue_memory = torch.sum(cue_attn.view(batch_size, sent_num, 1, 1) * cue_enc_outputs, dim=1)

        cue_length = inputs.cue[1].gather(1, indexs.view(-1,1)).squeeze(1)
        outputs.add(indexs=indexs)
        if 'index' in inputs.keys():
            outputs.add(attn_index=inputs.index)

        draft_init_state = self.draft_decoder.initialize_state(
            hidden=src_enc_hidden,
            src_enc_inputs=src_enc_inputs,
            src_enc_outputs=src_enc_outputs,
            knowledge=knowledge)
        keywords_dec_inputs = inputs.keywords[0][:, 1:]
        draft_logits, _, draft_dec_outputs, draft_length = self.draft_decoder(keywords_dec_inputs,draft_init_state, is_training=self.training)
        if self.copy:
            dec_init_state = self.decoder.initialize_state(
                hidden=src_enc_hidden,
                src_enc_inputs=src_enc_inputs,
                src_enc_outputs=src_enc_outputs,
                cue_enc_inputs=cue_enc_inputs,
                cue_enc_outputs=cue_enc_outputs.view(batch_size,sent_num,sent-2,-1),
                knowledge=knowledge,
                selected_cue_memory=cue_memory,
                selected_cue_length=cue_length,
                draft_dec_outputs=draft_dec_outputs,
                draft_length=draft_length,
                draft_logits=draft_logits)
        else:
            dec_init_state = self.decoder.initialize_state(
                hidden=src_enc_hidden,
                src_enc_outputs=src_enc_outputs,
                knowledge=knowledge,
                selected_cue_memory=cue_memory,
                selected_cue_length=cue_length,
                draft_dec_outputs=draft_dec_outputs,
                draft_length=draft_length,
                draft_logits=draft_logits)
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, draft_inputs, dec_inputs, hidden=None, is_training=False):
        """
        forward
        """
        outputs, dec_init_state = self.encode(
                enc_inputs, hidden, is_training=is_training)
        log_probs, draft_logits, _ = self.decoder(dec_inputs, dec_init_state)
        outputs.add(draft_logits=draft_logits)
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, draft_target, target, epoch=-1):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # test begin
        # nll = self.nll(torch.log(outputs.posterior_attn+1e-10), outputs.attn_index)
        # loss += nll
        # attn_acc = attn_accuracy(outputs.posterior_attn, outputs.attn_index)
        # metrics.add(attn_acc=attn_acc)
        # metrics.add(loss=loss)
        # return metrics
        # test end
        draft_logits = outputs.draft_logits[:,:draft_target.size(1)].contiguous()
        draft_nll_loss = self.nll_loss(draft_logits, draft_target)
        draft_num_words = draft_target.ne(self.padding_idx).sum().item()
        draft_acc = accuracy(draft_logits, draft_target, padding_idx=self.padding_idx)
        metrics.add(draft_nll=(draft_nll_loss, draft_num_words), draft_acc=draft_acc)
        loss += draft_nll_loss

        logits = outputs.logits
        scores = -self.nll_loss(logits, target, reduction=False)
        nll_loss = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll_loss, num_words), acc=acc)

        if self.use_posterior:
            kl_loss = self.kl_loss(torch.log(outputs.prior_attn + 1e-10),
                                   outputs.posterior_attn.detach())
            metrics.add(kl=kl_loss)

            if self.kl_annealing:
                loss += min(epoch / self.pretrain_epoch, 1) * kl_loss
                loss += nll_loss
            else:
                if self.use_bow:
                    bow_logits = outputs.bow_logits
                    bow_labels = target[:, :-1]
                    bow_logits = bow_logits.repeat(1, bow_labels.size(-1), 1)
                    bow = self.nll_loss(bow_logits, bow_labels)
                    loss += bow
                    metrics.add(bow=bow)

                if epoch == -1 or epoch > self.pretrain_epoch:
                    loss += nll_loss
                    loss += kl_loss

            if self.use_dssm:
                mse = self.mse_loss(outputs.dssm, outputs.reply_vec.detach())
                loss += mse
                metrics.add(mse=mse)
                pos_logits = outputs.pos_logits
                pos_target = torch.ones_like(pos_logits)
                neg_logits = outputs.neg_logits
                neg_target = torch.zeros_like(neg_logits)
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, pos_target, reduction='none')
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, neg_target, reduction='none')
                loss += (pos_loss + neg_loss).mean()
                metrics.add(pos_loss=pos_loss.mean(), neg_loss=neg_loss.mean())

            if self.use_pg:
                posterior_probs = outputs.posterior_attn.gather(1, outputs.indexs.view(-1, 1))
                reward = -perplexity(logits, target, self.weight, self.padding_idx) * 100
                pg_loss = -(reward.detach() - self.baseline) * posterior_probs.view(-1)
                pg_loss = pg_loss.mean()
                loss += pg_loss
                metrics.add(pg_loss=pg_loss, reward=reward.mean())

            if 'attn_index' in outputs:
                attn_acc = attn_accuracy(outputs.posterior_attn, outputs.attn_index)
                metrics.add(attn_acc=attn_acc)
        else:
            loss += nll_loss

        metrics.add(loss=loss)
        return metrics, scores

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=False, epoch=-1):
        """
        iterate
        """
        enc_inputs = inputs
        draft_inputs = inputs.keywords[0][:, :-1], inputs.keywords[1] - 1
        draft_target = inputs.keywords[0][:, 1:]
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        target = inputs.tgt[0][:, 1:]

        outputs = self.forward(enc_inputs, draft_inputs, dec_inputs, is_training=is_training)
        metrics, scores = self.collect_metrics(outputs, draft_target, target, epoch=epoch)

        loss = metrics.loss
        if torch.isnan(loss):
            # raise ValueError("nan loss encountered")
            return metrics, scores
        if is_training:
            if epoch <= self.pretrain_epoch:
                optimizer.param_groups[0]['lr'] = self.pretrain_lr
            else:
                optimizer.param_groups[0]['lr'] = self.lr
            if self.use_pg:
                self.baseline = 0.99 * self.baseline + 0.01 * metrics.reward.item()
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics, scores
