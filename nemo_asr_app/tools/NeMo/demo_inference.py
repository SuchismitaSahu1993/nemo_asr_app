import os
import numpy as np
import scipy.io.wavfile as wave
import torch

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import post_process_predictions
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from ruamel.yaml import YAML

def offline_inference(config, encoder, decoder, audio_file,
                      lm_path=None, beam_width=200, alpha=3, beta=0.1):
  MODEL_YAML = config
  CHECKPOINT_ENCODER = encoder
  CHECKPOINT_DECODER = decoder
  sample_rate, signal = wave.read(audio_file)

  # get labels (vocab)
  yaml = YAML(typ="safe")
  with open(MODEL_YAML) as f:
    model_definition = yaml.load(f)
  labels = model_definition['labels']

  # build neural factory and neural modules
  neural_factory = nemo.core.NeuralModuleFactory(
    placement=nemo.core.DeviceType.GPU,
    backend=nemo.core.Backend.PyTorch)

  # AudioDataLayer
  class AudioDataLayer(DataLayerNM):
    @property
    def output_ports(self):
        return {
          'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
          'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = np.reshape(signal.astype(np.float32)/32768., [1, -1])
        self.signal_shape = np.expand_dims(self.signal.size, 0).astype(np.int64)
        self.output = True

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self

  # Instantiate necessary neural modules
  data_layer = AudioDataLayer(sample_rate=model_definition['sample_rate'])

  data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
    **model_definition['AudioToMelSpectrogramPreprocessor'])

  jasper_encoder = nemo_asr.JasperEncoder(
    feat_in=model_definition['AudioToMelSpectrogramPreprocessor']['features'],
    **model_definition['JasperEncoder'])

  jasper_decoder = nemo_asr.JasperDecoderForCTC(
    feat_in=model_definition['JasperEncoder']['jasper'][-1]['filters'],
    num_classes=len(model_definition['labels']))

  greedy_decoder = nemo_asr.GreedyCTCDecoder()

  # load model
  jasper_encoder.restore_from(CHECKPOINT_ENCODER)
  jasper_decoder.restore_from(CHECKPOINT_DECODER)

  # Define inference DAG
  audio_signal, audio_signal_len = data_layer()
  processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)
  encoded, encoded_len = jasper_encoder(audio_signal=processed_signal,
                                        length=processed_signal_len)
  log_probs = jasper_decoder(encoder_output=encoded)
  predictions = greedy_decoder(log_probs=log_probs)

  inf_array = [
    audio_signal,
    processed_signal,
    encoded,
    log_probs,
    predictions]

  # language model
  if lm_path:
    beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
      vocab=labels,
      beam_width=beam_width,
      alpha=alpha,
      beta=beta,
      lm_path=lm_path,
      num_cpus=max(os.cpu_count(), 1))
    beam_predictions = beam_search_with_lm(log_probs=log_probs,
                                           log_probs_length=encoded_len)
    inf_array.append(beam_predictions)

  # inference
  data_layer.set_signal(signal)
  tensors = neural_factory.infer(inf_array, verbose=False)

  # results
  audio = tensors[0][0][0].cpu().numpy()
  features = tensors[1][0][0].cpu().numpy()
  encoded_features = tensors[2][0][0].cpu().numpy(),
  probs = tensors[3][0][0].cpu().numpy()
  preds = tensors[4][0]
  transcript = post_process_predictions([preds], labels)

  if lm_path:
      beam_preds = tensors[5][0][0][0][1]
      return transcript, probs, beam_preds
  else:
        return transcript, probs
