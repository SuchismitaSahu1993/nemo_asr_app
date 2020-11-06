# Copyright (c) 2019 NVIDIA Corporation
import sys
import os
import pandas as pd
import subprocess
import json
import codecs
import unidecode
import argparse
from num2words import num2words
from typing import List
from tools.filetools import file_exists

#alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
alphabet = (" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "á", "é", "í", "ó", "ú", "ñ", "ü")

def nums2strings(txt):
  """Convert digits to words"""
  words = txt.split()
  for i, word in enumerate(words):
    if any(i.isdigit() for i in word):
      if word.isdigit():
        # digits only
        num =  num2words(word, lang="es").replace("-"," ").replace(",","")
        words[i] = num
      else:
        # digits with letters, e.g. 3d
        # identify letters/digits and split
        numbers = [x for x in word if x.isdigit()]
        numbers = ''.join(numbers)
        chars = [x for x in word if not x.isdigit()]
        chars = ''.join(chars)
        num_tmp =  num2words(numbers, lang="es").replace("-"," ").replace(",","")
        words[i] = num_tmp + " " + chars
  return ' '.join(words)

def remove_non_vocab_chars(txt: str, valid_chars: List[str]) -> str:
    res_arr = []
    for c in txt:
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

def process_transcript(txt: str, vocab: List[str]) -> str:
    # Lowercase
    txt = txt.strip().lower()
    # Convert numbers to strings
    txt = nums2strings(txt)
    # Remove every character which is not in vocabluary
    txt = remove_non_vocab_chars(txt, valid_chars=vocab)
    return txt

def tsv_to_dataset(path, tsv_files, manifest_file):
  min_duration = 10000.0
  max_duration = 0.0
  total_duration = 0.0

  manifests = []
  for tfile in tsv_files:
    tsv_file = os.path.join(path, tfile)
    assert(file_exists(tsv_file))
    print('Processing: {0}'.format(tsv_file))
    dt = pd.read_csv(tsv_file, sep='\t', encoding='utf8')
    for index, row in dt.iterrows():
      try:
        entry = {}
        # audio processing
        wav_dir = os.path.join(path, "wavs")
        os.system("mkdir -p {0}".format(wav_dir))
        mp3_file = os.path.join(path, "clips", row['path'])
        wav_file = os.path.join(wav_dir, row['path'].replace(".mp3",".wav"))
        if not os.path.exists(wav_file):
            subprocess.check_output("sox {0} -c 1 -r 16000 {1}".format(mp3_file, wav_file), shell=True)
        duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)
        duration = float(duration)
        if duration > max_duration:
          max_duration = duration
        if duration < min_duration:
          min_duration = duration
        total_duration += duration

        entry['audio_filepath'] = wav_file
        entry['duration'] = duration

        # text processing
        new_text = process_transcript(row['sentence'], vocab=alphabet)
        entry['text'] = new_text
        manifests.append(entry)
      except:
        print("SOMETHING WENT WRONG - IGNORING ENTRY")

  manifest_file = os.path.join(path, manifest_file)
  print("Saving dataset to {}".format(manifest_file))
  with codecs.open(manifest_file, 'w', encoding='utf-8') as fout:
    for m in manifests:
      fout.write(json.dumps(m, ensure_ascii=False) + '\n')
  total_hrs = total_duration / 3600
  print(f'Processed {tsv_files} to {manifest_file}. Total duration {total_hrs} Hrs, min_duration {min_duration} secs, max_duration {max_duration} secs')
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
