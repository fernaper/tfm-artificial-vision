import base64
import cv2
import json
import numpy as np
import requests

from os.path import join

from tfm_core import config


def main():
    image_path = join(config.DATA_PATH, 'blob2.jpg')
    frame = cv2.imread(image_path)

    data = json.dumps({"instances": [frame.tolist()]})

    headers = {"signature_name": "predict", "content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/resnet:predict', data=data, headers=headers)

    response = json.loads(json_response.text)

    print(response)

    predictions = json.loads(json_response.text)['predictions']
    img = predictions

    img = np.asarray(img, dtype=np.float32)

    cv2.imshow('Info', frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()