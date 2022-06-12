import numpy as np
import time
import cv2

def stream_video(stream_time = 1):
    #1 sec = 26 frames
    video = cv2.VideoCapture(0)

    if (video.isOpened() == False):
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('Temp/video.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)

    start_time = time.time()
    secs = time.time() - start_time
    while secs < stream_time:
        secs = time.time() - start_time
        ret, frame = video.read()

        if ret == True:
            result.write(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    video.release()
    result.release()
    cv2.destroyAllWindows()
    images = conv_vid_to_img_array()
    return images

def conv_vid_to_img_array(vid = "Temp/video.avi"):
    cap = cv2.VideoCapture(vid)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frameCount)

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    #shows the 10th image in the video
    # cv2.namedWindow('frame 10')
    # cv2.imshow('frame 10', buf[9])
    #
    # cv2.waitKey(0)
    return buf

