# Copyright (c) 2019 NVIDIA Corporation
import sys
import os
import pandas as pd
import subprocess
import json
import argparse
import unidecode
from tools.filetools import file_exists

def normalize_str(txt) -> str:
    valid_chars = (" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                   "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
                   "x", "y", "z", "'",
                   "ß","ä","ö","ü")
    #new_txt = txt.lower().strip()
    new_txt = unidecode.unidecode(txt.lower().strip())
    res_arr = []
    for c in new_txt:
        if c in valid_chars:
            res_arr.append(c)
        else:
            # remove accent and see if it is valid
            non_accent_c = unidecode.unidecode(c)
            if non_accent_c in valid_chars:
                res_arr.append(non_accent_c)
            # a character we don't know
            else:
                res_arr.append(' ')
    res = ''.join(res_arr).strip()
    return ' '.join(res.split())

def tsv_to_dataset(path, tsv_files, manifest_file):
  manifests = []
  for tfile in tsv_files:
    tsv_file = os.path.join(path, tfile)
    assert(file_exists(tsv_file))
    print('Processing: {0}'.format(tsv_file))
    dt = pd.read_csv(tsv_file, sep='\t')
    for index, row in dt.iterrows():
      try:
        entry = {}
        wav_dir = os.path.join(path, "wavs")
        os.system("mkdir -p {0}".format(wav_dir))

        mp3_file = os.path.join(path, "clips", row['path'])
        wav_file = os.path.join(wav_dir, row['path'].replace(".mp3",".wav"))

        if not os.path.exists(wav_file):
          subprocess.check_output("sox -v 0.98 {0} -c 1 -r 16000 {1}".format(
            mp3_file, wav_file), shell=True)
        duration = subprocess.check_output("soxi -D {0}".format(wav_file),
                                           shell=True)
        entry['audio_filepath'] = wav_file
        entry['duration'] = float(duration)
        entry['text'] = normalize_str(row['sentence'])
        manifests.append(entry)
      except:
        print("SOMETHING WENT WRONG - IGNORING ENTRY")

  manifest_file = os.path.join(path, manifest_file)
  print("Saving dataset to {}".format(manifest_file))
  with open(manifest_file, 'w') as fout:
    for m in manifests:
      fout.write(json.dumps(m) + '\n')
  print('Done!')

def main():
  parser = argparse.ArgumentParser(description='Build NeMo ready dataset from tsv and mp3 files')
  parser.add_argument('--path', type=str, required=True,
                      help='Directory of dataset files')
  parser.add_argument('--tsv_files', type=str, required=True,
                      help='List of tsv files to convert')
  parser.add_argument('--output', type=str, required=True,
                      help='Output dataset (.json) filename')
  args = parser.parse_args()

  tsvs=args.tsv_files.split(",")
  tsv_to_dataset(args.path, tsvs, args.output)

if __name__ == "__main__":
    main()
