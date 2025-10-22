#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database
from images_framework.src.composite import Composite
from src.students_landmarks import StudentsLandmarks


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns-file', '-a', dest='anns_file', required=True,
                        help='Ground truth annotations file.')
    args, unknown = parser.parse_known_args()
    anns_file = args.anns_file
    return unknown, anns_file


def load_annotations(anns_file):
    """
    Load ground truth annotations according to each database.
    """
    print('Open annotations file: ' + str(anns_file))
    if os.path.isfile(anns_file):
        pos = anns_file.rfind('/') + 1
        path = anns_file[:pos]
        file = anns_file[pos:]
        db = file[:file.find('_ann')]
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        with open(anns_file, 'r', encoding='utf-8') as ifs:
            lines = ifs.readlines()
            anns = []
            for i in tqdm(range(len(lines)), file=sys.stdout):
                parts = lines[i].strip().split(';')
                if parts[0] == '@':
                    db = parts[1]
                if parts[0] == '#' or parts[0] == '@':
                    continue
                idx = next((idx for idx, subset in enumerate(datasets) if db in subset), None)
                if idx is None:
                    raise ValueError('Database does not exist')
                seq = Database.__subclasses__()[idx]().load_filename(path, db, lines[i])
                if len(seq.images) == 0:
                    continue
                anns.append(seq)
        ifs.close()
    else:
        raise ValueError('Annotations file does not exist')
    return anns


def main():
    """
    Students landmarks train script.
    """
    unknown, anns_file = parse_options()
    # Load computer vision components
    composite = Composite()
    sa = StudentsLandmarks('')
    composite.add(sa)

    composite.parse_options(unknown)
    anns = load_annotations(anns_file)
    composite.load(Modes.TRAIN)
    anns_train, anns_valid = train_test_split(anns, test_size=0.1, random_state=1, shuffle=True)
    composite.train(anns_train, anns_valid)


if __name__ == '__main__':
    main()
