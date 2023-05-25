import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
from funasr.modules.attention import MultiHeadedAttentionSANM
from funasr.export.models.modules.multihead_att import MultiHeadedAttentionSANM as MultiHeadedAttentionSANM_export
from funasr.export.models.modules.encoder_layer import EncoderLayerSANM as EncoderLayerSANM_export
from funasr.modules.positionwise_feed_forward import PositionwiseFeedForward
from funasr.export.models.modules.feedforward import PositionwiseFeedForward as PositionwiseFeedForward_export

class SANMEncoder(nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=560,
        model_name='encoder',
        onnx: bool = True,
    ):
        super().__init__()
        self.embed = model.embed
        self.model = model
        self.feats_dim = feats_dim
        self._output_size = model._output_size

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        if hasattr(model, 'encoders0'):
            for i, d in enumerate(self.model.encoders0):
                if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                    d.self_attn = MultiHeadedAttentionSANM_export(d.self_attn)
                if isinstance(d.feed_forward, PositionwiseFeedForward):
                    d.feed_forward = PositionwiseFeedForward_export(d.feed_forward)
                self.model.encoders0[i] = EncoderLayerSANM_export(d)

        for i, d in enumerate(self.model.encoders):
            if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                d.self_attn = MultiHeadedAttentionSANM_export(d.self_attn)
            if isinstance(d.feed_forward, PositionwiseFeedForward):
                d.feed_forward = PositionwiseFeedForward_export(d.feed_forward)
            self.model.encoders[i] = EncoderLayerSANM_export(d)
        
        self.model_name = model_name
        self.num_heads = model.encoders[0].self_attn.h
        self.hidden_size = model.encoders[0].self_attn.linear_out.out_features

        import os
        self.fp16 = float(os.environ.get('FP16', False))
        self.export_fp16 = float(os.environ.get('EXPORT_FP16', False))
        if self.export_fp16:
            self.model.encoders0.half()
            self.model.encoders.half()
            self.model.after_norm.half()

    
    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0
        
        return mask_3d_btd, mask_4d_bhlt

    # """
    def forward(self,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                ):
        speech = speech * self._output_size ** 0.5
        mask = self.make_pad_mask(speech_lengths)
        mask = self.prepare_mask(mask)
        if self.embed is None:
            xs_pad = speech
        else:
            xs_pad = self.embed(speech)

        if self.fp16: xs_pad = xs_pad / self.fp16
        if self.export_fp16: xs_pad = xs_pad.half(); mask = tuple([i.half() for i in mask])
        # import pdb; pdb.set_trace()
        encoder_outs = self.model.encoders0(xs_pad, mask)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # import pdb; pdb.set_trace()
        encoder_outs = self.model.encoders(xs_pad, mask)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # import pdb; pdb.set_trace()
        xs_pad = self.model.after_norm(xs_pad)
        if self.export_fp16: xs_pad = xs_pad.float()

        return xs_pad, speech_lengths
    """
    # MASK
    def forward(self,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                ):
        speech = speech * self._output_size ** 0.5
        mask = self.make_pad_mask(speech_lengths)
        mask = self.prepare_mask(mask)
        if self.embed is None:
            xs_pad = speech
        else:
            xs_pad = self.embed(speech)

        if self.fp16: xs_pad = xs_pad / self.fp16
        if self.export_fp16: xs_pad = xs_pad.half(); mask = tuple([i.half() for i in mask])
        # import pdb; pdb.set_trace()
        encoder_outs = self.model.encoders0(xs_pad, mask[0], mask[1])
        xs_pad = encoder_outs[0]

        # import pdb; pdb.set_trace()
        encoder_outs = self.model.encoders(xs_pad, mask[0], mask[1])
        xs_pad = encoder_outs[0]

        # import pdb; pdb.set_trace()
        xs_pad = self.model.after_norm(xs_pad)
        if self.export_fp16: xs_pad = xs_pad.float()

        return xs_pad, speech_lengths
    """

    def get_output_size(self):
        return self.model.encoders[0].size

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return (feats)

    def get_input_names(self):
        return ['feats']

    def get_output_names(self):
        return ['encoder_out', 'encoder_out_lens', 'predictor_weight']

    def get_dynamic_axes(self):
        return {
            'feats': {
                1: 'feats_length'
            },
            'encoder_out': {
                1: 'enc_out_length'
            },
            'predictor_weight':{
                1: 'pre_out_length'
            }

        }
