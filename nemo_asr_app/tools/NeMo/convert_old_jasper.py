import argparse
from omegaconf import DictConfig
import pytorch_lightning as pl
from ruamel.yaml import YAML
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel

parser = argparse.ArgumentParser(description="Converts old Jasper/QuartzNet models to NeMo v1.0beta")
parser.add_argument(
    "--config_path", default=None, required=True, help="Path to model config (NeMo v1.0beta)"
)
parser.add_argument(
    "--encoder_ckpt", default=None, required=True, help="Encoder checkpoint path"
)
parser.add_argument(
    "--decoder_ckpt", default=None, required=True, help="Decoder checkpoint path"
)
parser.add_argument(
    "--output_path", default=None, required=True, help="Output checkpoint path (should be .nemo)"
)

args = parser.parse_args()

yaml = YAML(typ='safe')
with open(args.config_path) as f:
    params = yaml.load(f)

asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']))

asr_model.encoder.load_state_dict(torch.load(args.encoder_ckpt))
asr_model.decoder.load_state_dict(torch.load(args.decoder_ckpt))
print("========= Loaded Old Checkpoint =========")

# Brief sanity check
# files = ['/data/librispeech/LibriSpeech/dev-clean-wav/2803-154328-0000.wav']
# for fname, transcription in zip(files, asr_model.transcribe(paths2audio_files=files)):
#    print(f"Audio in {fname} was recognized as: {transcription}")

asr_model.save_to(args.output_path)
