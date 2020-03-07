import cv2

from datetime import datetime
from glob import glob
from os.path import join, isdir

from cv2_tools.Management import ManagerCV2
from tfm_core import config


def save_images_by_class(output_path, selected_class, sub_images):
    for i, sub_image in enumerate(sub_images):
        file_name = '{selected_class}_{selection}.jpg'.format(
            selected_class=selected_class,
            selection=i
        )

        image_path = join(output_path, file_name)
        cv2.imwrite(image_path, sub_image)


def process_frame(frame, classes_to_sub_images, limit_per_class, width=64, height=64, extra_info='video'):
    frame_copy = frame.copy()

    # Select ROI
    for current_class in classes_to_sub_images.keys():
        # Skip completed classes
        if len(classes_to_sub_images[current_class]) >= limit_per_class:
            continue

        msg = '{} Select next: {}'.format(extra_info, current_class)
        print('-'*100)
        print(msg + ' (press c to skip class)')
        print('-'*100)
        while True:
            x,y,w,h = cv2.selectROI('Frame', frame_copy, True, False)

            if not any([x,y,w,h]):
                break

            classes_to_sub_images[current_class].append(frame[y:y+h, x:x+w])

            frame_copy = cv2.rectangle(frame_copy, (x,y), (x+w,y+h), (255,120,0), 2) 

            #min_width = min(min_width, w)
            #min_height = min(min_height, h)

    return classes_to_sub_images


def main(video_path, images_output_path):
    classes_to_sub_images = {
        'car': [],
        'motorbike': [],
        'pickup': [],
        #'truck':[],
        #'van':[]
    }

    frames_to_skip = 50
    limit_per_class = 100

    if isdir(video_path):
        for frame_path in glob(join(video_path, '*.jpg')):
            frame = cv2.imread(frame_path)
            extra_info = '(Frame: {})'.format(frame_path.split('/')[-1])
            classes_to_sub_images = process_frame(frame, classes_to_sub_images, limit_per_class, extra_info=extra_info)

    else:
        manager_cv2 = ManagerCV2(cv2.VideoCapture(video_path))
        manager_cv2.add_keystroke(27, 1, exit=True) # Exit when `Esc`
        manager_cv2.add_keystroke(ord(' '), 1, 'action')

        #min_width, min_height = 1000, 1000 # dummy values
        #nearest_power_of_2 = lambda x: 2**(x-1).bit_length()

        for frame in manager_cv2:
            if manager_cv2.count_frames % frames_to_skip != 0:
                continue

            extra_info = '(Frame: {})'.format(manager_cv2.count_frames)
            classes_to_sub_images = process_frame(frame, classes_to_sub_images, limit_per_class, extra_info=extra_info)

    # Change min width and height to the nearest power of 2 (bigger or equal than the value)
    #min_width = nearest_power_of_2(min_width)
    #min_height = nearest_power_of_2(min_height)

    min_width = 64
    min_height = 64

    for current_class, images in classes_to_sub_images.items():
        save_images_by_class(
            images_output_path,
            current_class,
            [cv2.resize(x, (min_width, min_height)) for x in images]
        )


if __name__ == "__main__":
    main(join(config.VIDEOS_PATH, 'cars1.mp4'),
         join(config.DATA_PATH, 'dataset'))

    #main(join(config.DATA_PATH, 'NotProcessed'),
    #     join(config.DATA_PATH, 'dataset2'))
