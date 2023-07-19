
# ---------------    Face Recognition using OpenCV and HaarCascade    ---------------
import cv2

#----------    Loadding the cascades    ----------
# We load the cascade for the face.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# We load the cascade for the eyes.     
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')      


#----------    Defining a function that will do the detections    ----------
# We create a function that takes as input the image in black and white (gray) and the original image (frame), 
    # and that will return the same image with the detector rectangles. 
def detect(gray, frame): 
    # We apply the detectMultiScale method from the face_cascade to locate one or several faces in the image.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w]       # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w]     # We get the region of interest in the colored image.
        # We apply the detectMultiScale method to locate one or several eyes in the image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        # For each detected eye:
        for (ex, ey, ew, eh) in eyes:
            # We paint a rectangle around the eyes, but inside the referential of the face.
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
    return frame # We return the image with the detector rectangles.



#----------    Capturing from WEBCAM    ----------

video_capture = cv2.VideoCapture(0) # turn the webcam on. 
# 0: internal webcam, 
# 1: external webcam

while True: # We repeat infinitely (until break):
    _, frame = video_capture.read() # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, frame) # We get the output of our detect function.
    cv2.imshow('Video', canvas) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.





# -----------------    no comment version   -----------------

# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


