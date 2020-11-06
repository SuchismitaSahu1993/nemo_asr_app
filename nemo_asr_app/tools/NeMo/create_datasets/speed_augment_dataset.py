# Copyright (c) 2019 NVIDIA Corporation
import sys
import os
import subprocess
import json
import codecs
import unidecode
import argparse
from itertools import chain
from multiprocessing import Pool, cpu_count
import numpy as np

def speed_change(data_org, speed):
    entry = {}
    audio_file = data_org['audio_filepath']
    new_audio = audio_file.split(".wav")[0] +"_" + str(speed) + ".wav"
    if not os.path.exists(new_audio):
        subprocess.check_output("sox {0} {1} speed {2}".format(audio_file, new_audio, speed), shell=True)
    duration = subprocess.check_output("soxi -D {0}".format(new_audio), shell=True)
    entry['audio_filepath'] = new_audio
    entry['duration'] = float(duration)
    entry['text'] = data_org['text']
    return entry

def augment_single_file(line):
    try:
        #original
        data_org = json.loads(line)
        # fast
        fast = speed_change(data_org, 1.1)
        # slow
        slow = speed_change(data_org, 0.9)
        #return tuple of lists - 3 data entries ([original, fast, slow],[original_duration,fast_duration,slow_duration])
        return ([data_org, fast, slow],[ data_org['duration'], fast['duration'], slow['duration'] ])
    except:
        print("SOMETHING WENT WRONG - IGNORING ENTRY")
        return ([None,None]) #we will filter out these entries where errors occured.

def dataset_augment(dataset_in, dataset_out):
    print('Augmenting: {0}'.format(dataset_in))
    with open(dataset_in, "r") as a_file:
        datafiles = [line for line in a_file]

    with Pool(cpu_count()) as p:
        outputs = p.map(augment_single_file, datafiles)

    #these will be lists of lists after unpacking using itertools.chain to flatten them into a normal list
    #also filters out None values after flattening the list
    manifests = list(chain.from_iterable([entry for entry,_ in outputs if entry is not None])) #unpack manifests
    durations = list(chain.from_iterable([duration for _,duration in outputs if duration is not None])) #unpack durations
    total_duration = np.sum(durations)
    min_duration = np.min(durations)
    max_duration = np.max(durations)

    print("Saving dataset to {}".format(dataset_out))
    with codecs.open(dataset_out, 'w', encoding='utf-8') as fout:
        for m in manifests:
            fout.write(json.dumps(m, ensure_ascii=False) + '\n')
    total_hrs = total_duration / 3600
    print(f'Processed {dataset_in} to {dataset_out}. Total duration {total_hrs} Hrs, min_duration {min_duration} secs, max_duration {max_duration} secs')
    print('Done!')

def main():
  parser = argparse.ArgumentParser(description='NeMo dataset speed augmentation 0.9 and 1.1')
  parser.add_argument('--dataset_in', type=str, required=True,
                      help='Original NeMo dataset to augment (.json)')
  parser.add_argument('--dataset_out', type=str, required=True,
                      help='Speed augmented output dataset (.json)')
  args = parser.parse_args()

  dataset_augment(args.dataset_in, args.dataset_out)

if __name__ == "__main__":
    main()
