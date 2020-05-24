from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules

from onmt.Trainer import Statistics


class TrainerMultimodal(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            train_img_feats: training global image features.
            valid_img_feats: validation global image features.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 train_attr=None, valid_attr=None,
                 train_img_feats=None, valid_img_feats=None,
                 train_img_mask=None, valid_img_mask=None,
                 train_feat_indices=None,
                 multimodal_model_type=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.train_attr = train_attr
        self.valid_attr = valid_attr
        self.train_img_feats = train_img_feats
        self.valid_img_feats = valid_img_feats
        self.train_img_mask = train_img_mask
        self.valid_img_mask = valid_img_mask
        self.train_feat_indices = train_feat_indices
        self.multimodal_model_type = multimodal_model_type
        self.progress_step = 0

        assert(not self.train_img_feats is None), \
                'Must provide training image features!'
        assert(not self.valid_img_feats is None), \
                'Must provide validation image features!'
        assert(not self.train_img_mask is None), \
                'Must provide training image mask!'
        assert(not self.valid_img_mask is None), \
                'Must provide validation image mask!'
        assert(not self.train_attr is None), \
                'Must provide training image attributes!'
        assert(not self.valid_attr is None), \
                'Must provide validation image attributes!'
        assert(self.multimodal_model_type in ['generator', 'bank', 'bank+generator', 'imgw', 'dcap']), \
                'Invalid multimodal model type: %s!'%(self.multimodal_model_type)

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            self.progress_step,
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                    self.progress_step += 1

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # extract indices for all entries in the mini-batch
            idxs = batch.indices.cpu().data.numpy()
            # load image features for this minibatch into a pytorch Variable
            img_feats = torch.from_numpy(self.valid_img_feats[idxs])
            img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
            img_mask = torch.from_numpy(self.valid_img_mask[idxs])
            img_mask = torch.autograd.Variable(img_mask, requires_grad=False)
            img_attr = torch.from_numpy(self.valid_attr[idxs])
            img_attr = torch.autograd.Variable(img_attr, requires_grad=False)
            if next(self.model.parameters()).is_cuda:
                img_feats = img_feats.cuda()
                img_mask = img_mask.cuda()
                img_attr = img_attr.cuda()
            else:
                img_feats = img_feats.cpu()
                img_mask = img_mask.cpu()
                img_attr = img_attr.cpu()
            # F-prop through the model.
            if 'bank' in self.multimodal_model_type \
                    or 'dcap' in self.multimodal_model_type \
                    or 'imgw' in self.multimodal_model_type:
                outputs, attns, _ = self.model(src, tgt, src_lengths, img_attr=img_attr,
                                               img_feats=img_feats, img_mask=img_mask)
            else:
                outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            if 'generator' in self.multimodal_model_type:
                batch_stats = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, attns, img_feats=img_feats)
            else:
                batch_stats = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        return '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, valid_stats.accuracy(), valid_stats.ppl(), epoch)

    def drop_checkpoint_with_bleu(self, opt, epoch, fields, valid_stats, bleu):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_bleu_%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, bleu, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))


    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            # extract indices for all entries in the mini-batch
            idxs = batch.indices.cpu().data.numpy()
            if self.train_feat_indices is not None:
                idxs = self.train_feat_indices[idxs]
            # load image features for this minibatch into a pytorch Variable
            img_attr = torch.from_numpy(self.train_attr[idxs])
            img_attr = torch.autograd.Variable(img_attr, requires_grad=False)
            img_feats = torch.from_numpy(self.train_img_feats[idxs])
            img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
            img_mask = torch.from_numpy(self.train_img_mask[idxs])
            img_mask = torch.autograd.Variable(img_mask, requires_grad=False)
            if next(self.model.parameters()).is_cuda:
                img_attr = img_attr.cuda()
                img_feats = img_feats.cuda()
                img_mask = img_mask.cuda()
            else:
                img_attr = img_attr.cpu()
                img_feats = img_feats.cpu()
                img_mask = img_mask.cpu()
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                if 'bank' in self.multimodal_model_type \
                        or 'dcap' in self.multimodal_model_type\
                        or 'imgw' in self.multimodal_model_type:
                    outputs, attns, dec_state = \
                        self.model(src, tgt, src_lengths, dec_state, img_attr=img_attr,
                                   img_feats=img_feats, img_mask=img_mask)
                else:
                    outputs, attns, dec_state = \
                        self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                if 'generator' in self.multimodal_model_type:
                    batch_stats = self.train_loss.sharded_compute_loss(
                            batch, outputs, attns, j,
                            trunc_size, self.shard_size, normalization,
                            img_feats=img_feats)
                else:
                    batch_stats = self.train_loss.sharded_compute_loss(
                            batch, outputs, attns, j,
                            trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()
