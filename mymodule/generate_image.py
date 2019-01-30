#!python3.6.6
import os
import re
import random
import shutil
import cv2

def video2frames(root='.', video_file='video.mov', image_dir='image_dir', image_file='img_frame', extension='.jpg', interval=0, file_name='aaa'):
    # Make the directory if it doesn't exist.
    if not os.path.exists(os.path.join(root,image_dir)):
        os.makedirs(os.path.join(root,image_dir))
    # Video to frames
    i = 0
    cap = cv2.VideoCapture(os.path.join(root, video_file))
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?
            break
        if (i % (interval+1)) is 0 :
#            image_file_frame = image_file + str(i) + extension
            image_file_frame = file_name + '-' + str(i) + extension
            save_path = os.path.join(root,image_dir,image_file_frame)
            cv2.imwrite(save_path, frame)  # Save a frame
            print('Save', save_path)
        i += 1

    cap.release()  # When everything done, release the capture
