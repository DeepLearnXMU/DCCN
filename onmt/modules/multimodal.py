import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable
import onmt.io
from onmt.Utils import aeq
from onmt.modules.Transformer import TransformerEncoder
from onmt.modules.Transformer import TransformerDecoder
from onmt.modules.Transformer import TransformerDecoderLayer
from onmt.modules.Transformer import TransformerDecoderState
from onmt.modules.CapMultiHeadedAttn import CapsuleMultiHeadedAttn


class MultiModalNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, bridge, decoder, multigpu=False, imgw=False, num_capsules=0,
                 num_regions=0, dcap=False):
        self.multigpu = multigpu
        super(MultiModalNMTModel, self).__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder
        self.imgw = imgw
        self.num_capsules = num_capsules
        self.num_regions = num_regions
        self.dcap = dcap

    def objtransemb(self, img_feats):
        # for objects
        # B 50 400 1
        batchsize = img_feats.size(0)
        regionnum = img_feats.size(1)
        attr_srcvoc = img_feats[:, :, :, 0:1].long().view(-1, 1600, 1)
        # B 50 400 embsize
        attr_emb = self.encoder.embeddings.make_embedding[0](attr_srcvoc)
        attr_emb = attr_emb.view(batchsize, regionnum, 1600, -1)
        # B 50 400 1
        attr_prob = img_feats[:, :, :, 1:]
        img_feats = attr_emb * attr_prob
        return img_feats.sum(2)

    def forward(self, src, tgt, lengths, dec_state=None, img_attr=None, img_feats=None, img_mask=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        assert img_feats is not None
        assert img_attr is not None
        # for attributes

        img_attr = self.objtransemb(img_attr)

        tgt = tgt[:-1]  # exclude last target from inputs
        if self.imgw:
            enc_final, memory_bank = self.encoder(src, img_feats, lengths)
            # expand indices to account for image "word"
            src = torch.cat([src[0:1, :, :], src], dim=0)
        else:
            enc_final, memory_bank = self.encoder(src, lengths)
            
        if self.bridge is not None:
            memory_bank = self.bridge(memory_bank, img_feats, src.size(0))
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)

        if self.dcap:
            decoder_outputs, dec_state, attns = \
                self.decoder(tgt, memory_bank,
                             enc_state if dec_state is None
                             else dec_state, img=img_feats, attr=img_attr, img_mask=img_mask,
                             memory_lengths=lengths)
        else:
            decoder_outputs, dec_state, attns = \
                 self.decoder(tgt, memory_bank,
                             enc_state if dec_state is None
                             else dec_state,
                             memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class MultiModalMemoryBankGate(nn.Module):
    def __init__(self, bank_size, img_feat_size, add=0):
        super(MultiModalMemoryBankGate, self).__init__()
        self.bank_to_gate = nn.Linear(
            bank_size, bank_size, bias=False)
        self.feat_to_gate = nn.Linear(
            img_feat_size, bank_size, bias=True)
        #nn.init.constant_(self.feat_to_gate.bias, 1.0) # newer pytorch
        nn.init.constant(self.feat_to_gate.bias, 1.0)
        self.add = add

    def forward(self, bank, img_feats, n_time):
        feat_to_gate = self.feat_to_gate(img_feats)
        feat_to_gate = feat_to_gate.expand(n_time, -1, -1)
        bank_to_gate = self.bank_to_gate(bank)
        gate = F.sigmoid(feat_to_gate + bank_to_gate) + self.add
        gate = gate / (1. + self.add)
        return bank * gate


class MultiModalGenerator(nn.Module):
    def __init__(self, old_generator, img_feat_size, add=0, use_hidden=False):
        super(MultiModalGenerator, self).__init__()
        self.linear = old_generator[0]
        self.vocab_size = self.linear.weight.size(0)
        self.gate = nn.Linear(img_feat_size, self.vocab_size, bias=True)
        #nn.init.constant_(self.gate.bias, 1.0) # newer pytorch
        nn.init.constant(self.gate.bias, 1.0)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.add = add
        self.use_hidden = use_hidden
        if use_hidden:
            self.hidden_to_gate = nn.Linear(
                self.linear.weight.size(1), self.vocab_size, bias=False)

    def forward(self, hidden, img_feats, n_time):
        proj = self.linear(hidden)
        pre_sigmoid = self.gate(img_feats)
        if self.use_hidden:
            pre_sigmoid = pre_sigmoid.repeat(n_time, 1)
            hidden_to_gate = self.hidden_to_gate(hidden)
            gate = F.sigmoid(pre_sigmoid + hidden_to_gate) + self.add
        else:
            gate = F.sigmoid(pre_sigmoid) + self.add
            gate = gate.repeat(n_time, 1)
        gate = gate / (1. + self.add)
        return self.logsoftmax(proj * gate)


class MultiModalLossCompute(onmt.Loss.NMTLossCompute):
    def _compute_loss(self, batch, output, target, img_feats):
        scores = self.generator(self._bottle(output), img_feats, output.size(0))

        ### Copypasta from superclass
        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)
        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats
        ### Copypasta ends


class MultiModalTransformerEncoder(TransformerEncoder):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
       num_layers (int): number of encoder layers
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    """
    def __init__(self, num_layers, hidden_size, img_feat_size,
                 dropout, embeddings):
        super(MultiModalTransformerEncoder, self).__init__(
            num_layers, hidden_size, dropout, embeddings)
        img_feat_size = 2048
        self.img_to_emb = nn.Linear(img_feat_size, hidden_size, bias=True)

    def forward(self, input, img_feats, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)
        emb = self.embeddings(input)
        img_feats = torch.mean(img_feats, dim=1, keepdim=True).permute(1, 0, 2)
        img_emb = self.img_to_emb(img_feats)

        # prepend image "word"
        emb = torch.cat([img_emb, emb], dim=0)
        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)

        # expand mask to account for image "word"
        words = torch.cat([words[:, 0:1], words], dim=1)

        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)  #padding mask

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return Variable(emb.data), out.transpose(0, 1).contiguous()


class CapsuleDecoderLayer(TransformerDecoderLayer):
    def __init__(self, size, dropout, num_iterations, num_capsules, num_regions, head_count=8, hidden_size=2048):
        super(CapsuleDecoderLayer, self).__init__(size, dropout, head_count, hidden_size)
        self.context_attn = CapsuleMultiHeadedAttn(
            head_count, size, num_iterations, num_capsules, num_regions, dropout=dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, img=None, attr=None, img_mask=None,
                cap_mask=None, previous_input=None):
        # Args Checks
        assert img is not None
        assert attr is not None
        input_batch, input_len, _ = inputs.size()
        if previous_input is not None:
            pi_batch, _, _ = previous_input.size()
            aeq(pi_batch, input_batch)
        contxt_batch, contxt_len, _ = memory_bank.size()
        aeq(input_batch, contxt_batch)

        src_batch, t_len, s_len = src_pad_mask.size()
        tgt_batch, t_len_, t_len__ = tgt_pad_mask.size()
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        # aeq(t_len, t_len_, t_len__, input_len)
        aeq(s_len, contxt_len)
        # END Args Checks

        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None
        query, attn = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask)
        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm, img, attr,
                                      img_mask=img_mask, cap_mask=cap_mask,
                                      mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)

        # CHECKS
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(contxt_batch, output_batch)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(contxt_len, s_len_)
        aeq(input_len, t_len_)
        # END CHECKS

        return output, attn, all_input


class CapsuleTransformerDecoder(TransformerDecoder):
    def __init__(self, num_layers, hidden_size, attn_type, copy_attn, dropout, embeddings,
                 num_iterations, num_capsules, num_regions):
        super(CapsuleTransformerDecoder, self).__init__(num_layers, hidden_size, attn_type,
                 copy_attn, dropout, embeddings)

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, dropout)
             for _ in range(num_layers-1)])
        self.transformer_layers.append(CapsuleDecoderLayer
                                       (hidden_size, dropout, num_iterations, num_capsules, num_regions))

    def forward(self, tgt, memory_bank, state, img, attr, img_mask, memory_lengths=None):
        """
         See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        # CHECKS

        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)

        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(tgt_batch, memory_batch, src_batch, tgt_batch)

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        # END CHECKS

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)
        cap_mask = tgt_words.data.eq(padding_idx)

        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]

            if i == self.num_layers-1:
                output, attn, all_input = self.transformer_layers[i](output, src_memory_bank,
                                                                     src_pad_mask, tgt_pad_mask,
                                                                     img=img, attr=attr, img_mask=img_mask,
                                                                     cap_mask=cap_mask,
                                                                     previous_input=prev_layer_input)
            else:
                output, attn, all_input = self.transformer_layers[i](output, src_memory_bank,
                                                                     src_pad_mask, tgt_pad_mask,
                                                                     previous_input=prev_layer_input)
            saved_inputs.append(all_input)

        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state = state.update_state(tgt, saved_inputs)
        return outputs, state, attns







