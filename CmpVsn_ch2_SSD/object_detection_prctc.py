
# --------------    Object Detection    --------------

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio


# Defining a function that will do the detections
    # We define a detect function that will take as inputs, 
        # a frame, 
        # a ssd neural network, and 
        # a transformation to be applied on the images, 
    # The function will return the frame with the detector rectangle.
def detect(frame, net, transform): 
    height, width = frame.shape[:2]     # We get the height and the width of the frame.
    frame_t = transform(frame)[0]       # We apply the transformation to our frame.
    x = torch.from_numpy(frame_t).permute(2, 0, 1)  # We convert the frame into a torch tensor.
    x = Variable(x.unsqueeze(0))    # We add a fake dimension corresponding to the batch.
    y = net(x)      # We feed the ssd-NN with the image and we get the output y.

    detections = y.data     # We create the detections tensor contained in the output y.
    # We create a tensor object of dimensions [width, height, width, height].
    scale = torch.Tensor([width, height, width, height])    

    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)): # For every class:
        j = 0       # We initialize the loop variable j that will correspond to the occurrences of the class.
        # all the 'occurrences' j of the class i that have a matching score larger than 0.6.
        while detections[0, i, j, 0] >= 0.6: 
            # coordinates of upper left and the lower right  corner of the detector rectangle.
            pt = (detections[0, i, j, 1:] * scale).numpy() 

            # We draw a rectangle around the detected object.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) 

            # We put the label of the class right above the rectangle.    
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            j += 1 # We increment j to get to the next occurrence.

    return frame # We return the original frame with the detector rectangle and the label around the detected object.


# Creating the SSD neural network
net = build_ssd('test')     # We create an object that is our neural network ssd.
# We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 


# Creating the transformation
    # We create an object of the BaseTransform class, a class that will do the required transformations 
        # so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) 


# Doing some OBJECT DETECTION on a video
reader = imageio.get_reader('funny_dog.mp4')    # We open the video.
fps = reader.get_meta_data()['fps']     # We get the 'fps' frequence (frames per second).
writer = imageio.get_writer('output.mp4', fps = fps)    # We create an output video with this 'same fps' frequence.

for i, frame in enumerate(reader):      # We 'ITERATE' on the frames of the output video:
    # We call our detect() function (defined above) to detect the object on the frame.
    frame = detect(frame, net.eval(), transform) 
    writer.append_data(frame)   # We add the next frame in the output video.
    print(i)    # We print the number of the processed frame.
    
writer.close()  # We close the process that handles the creation of the output video.





''' 

# ::::::::::::::::    no comment    ::::::::::::::::::

# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()

'''
