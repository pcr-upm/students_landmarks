#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import copy
import numpy as np
import importlib.util
from tqdm import tqdm
from pathlib import Path
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database
from images_framework.src.composite import Composite
from images_framework.src.viewer import Viewer
from src.students_landmarks import StudentsLandmarks


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns-file', '-a', dest='anns_file', required=True,
                        help='Ground truth annotations file.')
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    parser.add_argument('--save-file', '-f', dest='save_file', action="store_true",
                        help='Save experiments in a text file.')
    parser.add_argument('--save-image', '-i', dest='save_image', action="store_true",
                        help='Save processed images.')
    args, unknown = parser.parse_known_args()
    anns_file = args.anns_file
    show_viewer = args.show_viewer
    save_file = args.save_file
    save_image = args.save_image
    return unknown, anns_file, show_viewer, save_file, save_image


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
    Students landmarks database script.
    """
    unknown, anns_file, show_viewer, save_file, save_image = parse_options()
    # Load computer vision components
    composite = Composite()
    sa = StudentsLandmarks('')
    composite.add(sa)

    composite.parse_options(unknown)
    anns = load_annotations(anns_file)
    composite.load(Modes.TEST)
    spec = importlib.util.find_spec('images_framework')
    output_path = os.path.join('images_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    if show_viewer:
        viewer = Viewer('images_viewer')
    if save_file:
        ofs = open(output_path+'/results.txt', 'w', encoding='utf-8')
    if save_image:
        viewer = Viewer('images_save')
        dirname = os.path.join(output_path, 'images/')
        Path(dirname).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        for img_pred in pred.images:
            for obj_pred in img_pred.objects:
                obj_pred.clear()
                if obj_pred.bb == (-1, -1, -1, -1):
                    if all(np.array_equal(contour, np.array([[[-1, -1]], [[-1, -1]], [[-1, -1]]])) for contour in obj_pred.multipolygon):
                        obj_pred.bb = cv2.boundingRect(np.array([pt for contour in obj_pred.multipolygon for pt in contour]))
                        obj_pred.bb = (obj_pred.bb[0], obj_pred.bb[1], obj_pred.bb[0] + obj_pred.bb[2], obj_pred.bb[1] + obj_pred.bb[3])
                    else:
                        raise ValueError('Cannot perform alignment due to undefined object location')
        composite.process(anns[i], pred)
        if show_viewer:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.show(0)
        if save_file:
            composite.evaluate(ofs, anns[i], pred)
        if save_image:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.save(dirname)
            composite.save(dirname, pred)
    if save_file:
        ofs.close()


if __name__ == '__main__':
    main()
