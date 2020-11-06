# Copyright (c) 2019 NVIDIA Corporation
# python create_ezdi_dataset.py --path="/raid/datasets/asr/data/example_data/healthcare/" --outfile="ezdi_full.json"
import sys
import os
import subprocess
import json
import argparse
from tools.filetools import file_exists
from tools.transcript_tools import normalize
from striprtf.striprtf import rtf_to_text

manifests = []

def convert_to_dataset(path, outfile):
  apath = os.path.join(path, "Audio")
  tpath = os.path.join(path, "Documents")

  for text_filename in os.listdir(tpath):
    text_file = os.path.join(tpath, text_filename)
    assert(file_exists(text_file))

    print('Processing: {0}'.format(text_file))
    with open(text_file, 'r' ) as text:
      data = text.read().replace('\n','')
    data = rtf_to_text(data)

    wav_dir = os.path.join(path, "wavs")
    os.system("mkdir -p {0}".format(wav_dir))

    org_wav = os.path.join(apath, text_filename.replace(".rtf",".wav"))
    new_wav = os.path.join(path,"wavs", text_filename.replace(".rtf",".wav"))

    try:
      if not os.path.exists(new_wav):
        subprocess.check_output("sox -v 0.98 {0} -c 1 -r 16000 {1}".format(
          org_wav, new_wav), shell=True)
      duration = subprocess.check_output("soxi -D {0}".format(new_wav),
                                         shell=True)
      entry = {}
      entry['audio_filepath'] = new_wav
      entry['duration'] = float(duration)
      entry['text'] = normalize(data.rstrip('\n'))
      manifests.append(entry)
    except:
      print("SOMETHING WENT WRONG - IGNORING ENTRY")

  manifest_file = os.path.join(path, outfile)
  print("Saving dataset to {}".format(manifest_file))
  with open(manifest_file, 'w+') as fout:
    for m in manifests:
      fout.write(json.dumps(m) + '\n')
  print('Done!')


def main():
  parser = argparse.ArgumentParser(description='Build NeMo ready dataset from tsv and mp3 files')
  parser.add_argument('--path', type=str, required=True,
                      help='Path to directory with Audio and Document folders')
  parser.add_argument('--outfile', type=str, required=True,
                      help='Output dataset (.json) path')
  args = parser.parse_args()

  convert_to_dataset(args.path, args.outfile)

if __name__ == "__main__":
    main()
