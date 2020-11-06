import sys
import os
import subprocess
import json
import argparse
from tools.transcript_tools import remove_punct
from tools.filetools import file_exists

def create_dataset(path, data, output_file):
  if file_exists(output_file):
    print("Output file {} already exists.".format(output_file))
    return

  manifests = []
  for key, value in data.items():
    try:
      entry = {}
      wav_file = os.path.join(path, "wavs", key)
      duration = subprocess.check_output("soxi -D {0}".format(wav_file),
                                         shell=True)

      entry['audio_filepath'] = wav_file
      entry['duration'] = float(duration)
      entry['text'] = remove_punct(value['clean'].lower())
      manifests.append(entry)
    except:
      print("SOMETHING WENT WRONG - IGNORING ENTRY")

  print("Saving dataset to {}".format(output_file))
  with open(output_file, 'w') as fout:
    for m in manifests:
      fout.write(json.dumps(m) + '\n')
  print('Done!')

def main():
  parser = argparse.ArgumentParser(description='Build NeMo ready dataset')
  parser.add_argument('--path', type=str, required=True,
                      help='Directory of dataset files with wavs folder.')
  parser.add_argument('--org_json', type=str, required=True,
                      help='Path of original json file to convert')
  parser.add_argument('--output', type=str, required=True,
                      help='Path of output dataset (.json) filename')
  args = parser.parse_args()

  assert(file_exists(args.org_json))
  print('Processing: {0}'.format(args.org_json))
  print("This can take some time...")
  with open(args.org_json) as json_file:
    data = json.load(json_file)
  create_dataset(args.path, data, args.output)

if __name__ == "__main__":
    main()
