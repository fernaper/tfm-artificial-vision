import cv2
import numpy as np
import pathlib
import random
import requests

from os.path import join

from tfm_core import config
from tfm_core.dnn import utilities


def main(folder='binary_dataset', model='resnet'):
    data_dir = pathlib.Path(join(config.DATA_PATH, folder))
    images = list(data_dir.glob('*/*'))

    random.shuffle(images)

    class_names = np.array([item.name for item in data_dir.glob('*')])

    false_positive_by_class = {c:0 for c in class_names}

    correct = 0
    total = 0

    for image_path in images:
        frame = cv2.imread(image_path.as_posix())

        predictions = utilities.send_frame_serving_tf(frame, model=model)
        #img = predictions
        #img = np.asarray(img, dtype=np.float32)

        correct_class_index = np.where(class_names == image_path.parent.name)[0][0]
        detected_class_index = np.where(predictions == np.amax(predictions))[0][0]

        if detected_class_index == correct_class_index:
            correct +=1
        else:
            false_positive_by_class[class_names[detected_class_index]] += 1

        total += 1

        print('Conf: {0:.2f}'.format(max(predictions)),
              'Acc: {0:.4f}'.format(float(correct)/total), '|',
              ''.join(['{}->{};'.format(k.replace('_',''),v) for k, v in false_positive_by_class.items()]),
              end='\r')

        cv2.imshow('Image', frame)
        cv2.waitKey(1)
    
    print('\nAccuracy:', float(correct)/total)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', default='dataset',
        help='Testing dataset (default dataset)')

    parser.add_argument('-m', '--model', default='resnet',
        help='Detecting model (default resnet)')

    args = parser.parse_args()

    main(folder=args.dataset, model=args.model)
