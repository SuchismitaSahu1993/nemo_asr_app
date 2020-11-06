# Copyright (c) 2019 NVIDIA Corporation
import sys
import os
import pandas as pd
import subprocess
import json
import codecs
import unidecode
import argparse
from tools.filetools import file_exists
from multiprocessing import Pool, cpu_count
import numpy as np

def normalize_str(txt) -> str:
    # vocabulary
    valid_chars = (" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "á", "é", "í", "ó", "ú", "ñ", "ü")
    # lowercase
    new_txt = txt.lower().strip()
    # no digits exist in MCV_ES

    # remove characters not in vocabulary
    res_arr = []
    for c in new_txt:
        if c in valid_chars:
            res_arr.append(c)
        else:
            # remove accent and see if it is valid
            non_accent_c = unidecode.unidecode(c)
            if non_accent_c in valid_chars:
                res_arr.append(non_accent_c)
            else:
                # a character we don't know
                res_arr.append(' ')
    res = ''.join(res_arr).strip()
    return ' '.join(res.split())

def process_df_row(args):
    try:
        row, path, wav_dir = args
        entry = {}
        mp3_file = os.path.join(path, "clips", row['path'])
        wav_file = os.path.join(wav_dir, row['path'].replace(".mp3",".wav"))

        if not os.path.exists(wav_file):
            subprocess.check_output("sox {0} -c 1 -r 16000 {1}".format(mp3_file, wav_file), shell=True)
        duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)
        duration = float(duration)

        entry['audio_filepath'] = wav_file
        entry['duration'] = duration
        entry['text'] = normalize_str(row['sentence'])
        return (entry, duration)
    except:
        print("SOMETHING WENT WRONG - IGNORING ENTRY")
        return (None, None)

def tsv_to_dataset(path, tsv_files, manifest_file):
    wav_dir = os.path.join(path, "wavs")
    os.system("mkdir -p {0}".format(wav_dir))

    tsv_rows = []
    for tfile in tsv_files:
        tsv_file = os.path.join(path, tfile)
        assert(file_exists(tsv_file))
        print('Processing: {0}'.format(tsv_file))
        dt = pd.read_csv(tsv_file, sep='\t', encoding='utf8')
        #creating payload for parallel functions, combining an individual dataframe row with path and wave_dir in a tuple
        rows_list = [(row,path,wav_dir) for _, row in dt.iterrows()]
        tsv_rows.extend(rows_list)

    with Pool(cpu_count()) as p:
        outputs = p.map(process_df_row, tsv_rows)

    manifests = [entry for entry,_ in outputs if entry is not None] #unpack manifests
    durations = [duration for _,duration in outputs if duration is not None] #unpack durations
    total_duration = np.sum(durations)
    min_duration = np.min(durations)
    max_duration = np.max(durations)

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
