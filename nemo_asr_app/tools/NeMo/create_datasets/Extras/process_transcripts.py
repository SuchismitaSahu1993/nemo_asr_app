import json
import unidecode
import inflect
from argparse import ArgumentParser
from typing import List
parser = ArgumentParser()
parser.add_argument("--manifests_in",  type=str, required=True, nargs="*", help="path to input manifests")
parser.add_argument("--manifests_out", type=str, required=True, nargs="*", help="path to output manifests")


args = parser.parse_args()
alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

p = inflect.engine()


def nums2strings(txt: str) -> str:
    return txt


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


def process_json(path2json_in: str, path2json_out: str):
    min_duration = 10000.0
    max_duration = 0.0
    total_duration = 0.0
    with open(path2json_out, 'w') as tgt_f:
        with open(path2json_in, 'r') as src_f:
            for line in src_f:
                record = json.loads(line)
                new_record = {}
                for key, value in record.items():
                    if key == "text":
                        new_record['raw_text'] = value
                        new_text = process_transcript(value, vocab=alphabet)
                        new_record['text'] = new_text
                    else:
                        new_record[key] = value
                if new_record['text'] != '':
                    duration = new_record['duration']
                    if duration > max_duration:
                        max_duration = duration
                    if duration < min_duration:
                        min_duration = duration
                    total_duration += duration
                    tgt_f.write(json.dumps(new_record, ensure_ascii=False) + '\n')
    print(f'Processed {path2json_in} to {path2json_out}. Total duration {total_duration}, min_duration {min_duration}, max_duration {max_duration}')
    return total_duration, min_duration, max_duration


def main():
    if len(args.manifests_in) != len(args.manifests_out):
        raise ValueError('Input and output manifest names must contain the same numbers of filenames')
    total_duration, min_duration, max_duration = 0, 100000.0, 0
    for in_manifest, out_manifest in zip(args.manifests_in, args.manifests_out):
        print(f'Processing pair: ({in_manifest} => {out_manifest})')
        _total_duration, _min_duration, _max_duration = process_json(in_manifest, out_manifest)
        total_duration += _total_duration
        min_duration = min(min_duration, _min_duration)
        max_duration = max(max_duration, _max_duration)
    print(f'Combined total_duration {total_duration}, min_duration {min_duration}, max_duration {max_duration}')


if __name__ == '__main__':
    main()
