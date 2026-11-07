"""Inspect expected splits and shape consistency of word-aligned datasets."""
import argparse
from pathlib import Path
import pickle
import numpy as np


def check(path):
    with Path(path).open('rb') as f:
        data = pickle.load(f, encoding='latin1')
    if not isinstance(data, dict):
        raise ValueError('Expected dict with train/dev(or valid)/test splits')
    for wanted in ('train','dev','test'):
        actual = 'valid' if wanted=='dev' and 'dev' not in data else wanted
        examples = data.get(actual)
        if not examples:
            raise ValueError(f'Missing or empty split {wanted!r}')
        for j,sample in enumerate(examples[:min(5,len(examples))]):
            try:
                (words,vision,audio), label, sample_id = sample
            except (TypeError,ValueError) as e:
                raise ValueError(f'Invalid tuple structure at {actual}[{j}]') from e
            vision=np.asarray(vision); audio=np.asarray(audio)
            if isinstance(words,(str,bytes)) or vision.ndim!=2 or audio.ndim!=2:
                raise ValueError(f'Expected word list and rank-2 AV arrays at {actual}[{j}]')
            if not (len(words)==len(vision)==len(audio)):
                raise ValueError(f'Unaligned sequence lengths at {actual}[{j}]')
            if np.asarray(label).size!=1:
                raise ValueError(f'Non-scalar sentiment label at {actual}[{j}]')
        sample=examples[0]
        print(f'{actual}: {len(examples)} examples; audio_dim={np.asarray(sample[0][2]).shape[-1]}, visual_dim={np.asarray(sample[0][1]).shape[-1]}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    args=parser.parse_args()
    check(args.path)
