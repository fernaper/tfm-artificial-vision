import cv2
import numpy as np
import pathlib
import requests

from os.path import join

from tfm_core import config
from tfm_core.dnn import utilities


def main(folder='binary_dataset'):
    data_dir = pathlib.Path(join(config.DATA_PATH, folder))
    images = list(data_dir.glob('*/*'))

    class_names = np.array([item.name for item in data_dir.glob('*')])

    correct = 0
    total = 0

    for image_path in images:
        frame = cv2.imread(image_path.as_posix())

        predictions = utilities.send_frame_serving_tf(frame)
        #img = predictions
        #img = np.asarray(img, dtype=np.float32)

        correct_class_index = np.where(class_names == image_path.parent.name)[0][0]
        detected_class_index = np.where(predictions == np.amax(predictions))[0][0]

        if detected_class_index == correct_class_index:
            correct +=1

        total += 1

        print('Accuracy:', float(correct)/total, end='\r')

        cv2.imshow('Image', frame)
        cv2.waitKey(1)
    
    print('\nAccuracy:', float(correct)/total)


if __name__ == "__main__":
    main()