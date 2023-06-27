import logging
import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.export.models.encoder.sanm_encoder import SANMEncoder as SANMEncoder_export
from funasr.export.models.encoder.sanm_encoder import SANMEncoderChunkOpt as SANMEncoderChunkOpt_export
from funasr.export.models.encoder.conformer_encoder import ConformerEncoder as ConformerEncoder_export
from funasr.models.predictor.cif import CifPredictorV2, CifPredictorV3
from funasr.export.models.predictor.cif import CifPredictorV2 as CifPredictorV2_export
from funasr.export.models.predictor.cif import CifPredictorV3 as CifPredictorV3_export
from funasr.models.decoder.sanm_decoder import ParaformerSANMDecoder
from funasr.models.decoder.transformer_decoder import ParaformerDecoderSAN
from funasr.export.models.decoder.sanm_decoder import ParaformerSANMDecoder as ParaformerSANMDecoder_export
from funasr.export.models.decoder.transformer_decoder import ParaformerDecoderSAN as ParaformerDecoderSAN_export


class Paraformer(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
            model_name='model',
            **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        if isinstance(model.encoder, SANMEncoder):
            self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        elif isinstance(model.encoder, ConformerEncoder):
            self.encoder = ConformerEncoder_export(model.encoder, onnx=onnx)
        if isinstance(model.predictor, CifPredictorV2):
            self.predictor = CifPredictorV2_export(model.predictor)
        if isinstance(model.decoder, ParaformerSANMDecoder):
            self.decoder = ParaformerSANMDecoder_export(model.decoder, onnx=onnx)
        elif isinstance(model.decoder, ParaformerDecoderSAN):
            self.decoder = ParaformerDecoderSAN_export(model.decoder, onnx=onnx)
        
        self.feats_dim = feats_dim
        self.model_name = model_name

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)
        
    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ):
        # a. To device
        batch = {"speech": speech, "speech_lengths": speech_lengths}
        # batch = to_device(batch, device=self.device)
    
        enc, enc_len = self.encoder(**batch)
        mask = self.make_pad_mask(enc_len)[:, None, :]
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
        pre_token_length = pre_token_length.floor().type(torch.int32)

        decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        # sample_ids = decoder_out.argmax(dim=-1)

        return decoder_out, pre_token_length

    def get_dummy_inputs(self):
        speech = torch.randn(2, 30, self.feats_dim)
        speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
        return (speech, speech_lengths)

    def get_dummy_inputs_txt(self, txt_file: str = "/mnt/workspace/data_fbank/0207/12345.wav.fea.txt"):
        import numpy as np
        fbank = np.loadtxt(txt_file)
        fbank_lengths = np.array([fbank.shape[0], ], dtype=np.int32)
        speech = torch.from_numpy(fbank[None, :, :].astype(np.float32))
        speech_lengths = torch.from_numpy(fbank_lengths.astype(np.int32))
        return (speech, speech_lengths)

    def get_input_names(self):
        return ['speech', 'speech_lengths']

    def get_output_names(self):
        return ['logits', 'token_num']

    def get_dynamic_axes(self):
        return {
            'speech': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'speech_lengths': {
                0: 'batch_size',
            },
            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
        }


class BiCifParaformer(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
            model_name='model',
            **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        if isinstance(model.encoder, SANMEncoder):
            self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        elif isinstance(model.encoder, ConformerEncoder):
            self.encoder = ConformerEncoder_export(model.encoder, onnx=onnx)
        else:
            logging.warning("Unsupported encoder type to export.")
        if isinstance(model.predictor, CifPredictorV3):
            self.predictor = CifPredictorV3_export(model.predictor)
        else:
            logging.warning("Wrong predictor type to export.")
        if isinstance(model.decoder, ParaformerSANMDecoder):
            self.decoder = ParaformerSANMDecoder_export(model.decoder, onnx=onnx)
        elif isinstance(model.decoder, ParaformerDecoderSAN):
            self.decoder = ParaformerDecoderSAN_export(model.decoder, onnx=onnx)
        else:
            logging.warning("Unsupported decoder type to export.")
        
        self.feats_dim = feats_dim
        self.model_name = model_name

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)
        
    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ):
        # a. To device
        batch = {"speech": speech, "speech_lengths": speech_lengths}
        # batch = to_device(batch, device=self.device)
    
        enc, enc_len = self.encoder(**batch)
        mask = self.make_pad_mask(enc_len)[:, None, :]
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
        pre_token_length = pre_token_length.round().type(torch.int32)

        decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        
        # get predicted timestamps
        us_alphas, us_cif_peak = self.predictor.get_upsample_timestmap(enc, mask, pre_token_length)

        return decoder_out, pre_token_length, us_alphas, us_cif_peak

    def get_dummy_inputs(self):
        speech = torch.randn(2, 30, self.feats_dim)
        speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
        return (speech, speech_lengths)

    def get_dummy_inputs_txt(self, txt_file: str = "/mnt/workspace/data_fbank/0207/12345.wav.fea.txt"):
        import numpy as np
        fbank = np.loadtxt(txt_file)
        fbank_lengths = np.array([fbank.shape[0], ], dtype=np.int32)
        speech = torch.from_numpy(fbank[None, :, :].astype(np.float32))
        speech_lengths = torch.from_numpy(fbank_lengths.astype(np.int32))
        return (speech, speech_lengths)

    def get_input_names(self):
        return ['speech', 'speech_lengths']

    def get_output_names(self):
        return ['logits', 'token_num', 'us_alphas', 'us_cif_peak']

    def get_dynamic_axes(self):
        return {
            'speech': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'speech_lengths': {
                0: 'batch_size',
            },
            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
            'us_alphas': {
                0: 'batch_size',
                1: 'alphas_length'
            },
            'us_cif_peak': {
                0: 'batch_size',
                1: 'alphas_length'
            },
        }


class ParaformerOnline_encoder_predictor(nn.Module):
    """
    Author: Speech Lab, Alibaba Group, China
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
            model_name='model',
            **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        if isinstance(model.encoder, SANMEncoder):
            self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        elif isinstance(model.encoder, SANMEncoderChunkOpt):
            self.encoder = SANMEncoderChunkOpt_export(model.encoder, onnx=onnx)
        elif isinstance(model.encoder, ConformerEncoder):
            self.encoder = ConformerEncoder_export(model.encoder, onnx=onnx)
        if isinstance(model.predictor, CifPredictorV2):
            self.predictor = CifPredictorV2_export(model.predictor)

        self.feats_dim = feats_dim
        self.model_name = model_name

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ):
        # a. To device
        batch = {"speech": speech, "speech_lengths": speech_lengths, "online": True}
        batch.pop('online')  # Think: xw
        # batch = to_device(batch, device=self.device)

        enc, enc_len = self.encoder(**batch)
        mask = self.make_pad_mask(enc_len)[:, None, :]
        alphas, _ = self.predictor.forward_cnn(enc, mask)

        return enc, enc_len, alphas

    def get_dummy_inputs(self):
        speech = torch.randn(2, 30, self.feats_dim)
        speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
        return speech, speech_lengths

    def get_input_names(self):
        return ['speech', 'speech_lengths']

    def get_output_names(self):
        return ['enc', 'enc_len', 'alphas']

    def get_dynamic_axes(self):
        return {
            'speech': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'speech_lengths': {
                0: 'batch_size',
            },
            'enc': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'enc_len': {
                0: 'batch_size',
            },
            'alphas': {
                0: 'batch_size',
                1: 'feats_length'
            },
        }


class Paraformer_decoder(nn.Module):
    """
    Author: Speech Lab, Alibaba Group, China
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
            model_name='model',
            **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        if isinstance(model.decoder, ParaformerSANMDecoder):
            self.decoder = ParaformerSANMDecoder_export(model.decoder, onnx=onnx)
        elif isinstance(model.decoder, ParaformerDecoderSAN):
            self.decoder = ParaformerDecoderSAN_export(model.decoder, onnx=onnx)

        self.feats_dim = feats_dim
        self.model_name = model_name
        self.enc_size = model.encoder._output_size

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

    def forward(
            self,
            enc: torch.Tensor,
            enc_len: torch.Tensor,
            acoustic_embeds: torch.Tensor,
            acoustic_embeds_len: torch.Tensor,
    ):
        decoder_out, _ = self.decoder(enc, enc_len, acoustic_embeds, acoustic_embeds_len)
        decoder_out = torch.log_softmax(decoder_out, dim=-1)

        return decoder_out, acoustic_embeds_len

    def get_dummy_inputs(self, ):
        enc = torch.randn(2, 30, self.enc_size).type(torch.float32)
        enc_len = torch.tensor([6, 30], dtype=torch.int32)
        acoustic_embeds = torch.randn(2, 10, self.enc_size).type(torch.float32)
        acoustic_embeds_len = torch.tensor([5, 10], dtype=torch.int32)
        return enc, enc_len, acoustic_embeds, acoustic_embeds_len

    def get_input_names(self):

        return ['enc', 'enc_len', 'acoustic_embeds', 'acoustic_embeds_len']

    def get_output_names(self):

        return ['logits', 'pre_token_length']

    def get_dynamic_axes(self):
        return {
            'enc': {
                0: 'batch_size',
                1: 'enc_length'
            },
            'acoustic_embeds': {
                0: 'batch_size',
                1: 'token_length'
            },
            'enc_len': {
                0: 'batch_size',
            },
            'acoustic_embeds_len': {
                0: 'batch_size',
            },
            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
        }

