# Mon script OpenCV : Video_processing
#
import numpy as np
import cv2
from matplotlib import pyplot as plt




#Method of
def imgproc(imgc):
    return imgc


#Import video file on format .mp4
cap = cv2.VideoCapture('data/jurassicworld.mp4')
#
while (True):
    #Take each frame
    ret, frame = cap.read()

    # Capture frame-by-frame
    if ret == True:
        #color conversion of each frame using flag COLOR_BGR2GRAY to convert BGR to HSV
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #NameError: name 'frame_processing' is not defined
        #gray = frame_processing(gray)

        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(img, -1, kernel)
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
        plt.xticks([]), plt.yticks([])
        plt.show()

        # Display the resulting frame in gray
        cv2.imshow('frame', gray)
        #Display an image in a windows. The window automatically fits to the image size.
        #cv2.imshow('MavideoAvant', frame)
        #cv2.imshow('MavideoApres', img)
    else:
        print('video ended')
        break

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()