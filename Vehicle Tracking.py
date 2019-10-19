import torchvision
import torchvision.transforms as T
import imutils
import cv2
from PIL import Image

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (1280, 720))
vs = cv2.VideoCapture('highway.mp4')

# Scaling the frame for faster computation

WIDTH = 500
HEIGHT = 720 * WIDTH / 1280

# 

MINIMAL_WIDTH = WIDTH / 40
MINIMAL_HEIGHT = HEIGHT / 40
MINIMAL_AREA = WIDTH * HEIGHT / 1600

# Position of the highway sign board that blocks vehicles sometimes

w_left = int(WIDTH * 0.36)
w_right = int(WIDTH * 0.42)
h_top = int(HEIGHT * 0.43)
h_bottom = int(HEIGHT * 0.33)

# Counters for vehicles

vehicle_burnaby = 0
vehicle_coquitlam = 0

# Counting which frame we are processing

count = 0

# Checking if the vehicle has been counted

counted = []

# Classification using ResNet50
# Codes taken from https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/
# Author: Dr. Satya Mallick

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img, threshold):
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class

# Function for printing vehicle counts

def DisplayCount(frame, vehicle_burnaby, vehicle_coquitlam):
    text1 = "To Burnaby: {}".format(vehicle_burnaby)
    text2 = "To Coquitlam: {}".format(vehicle_coquitlam)
    cv2.putText(frame, text1, org=(20, 40) , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(frame, text2, org=(20, 70) , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

# Checking if a new vehicle has overlapped significantly with the other new vehicles

def OverlapWithNew(i, Boxes, X, Y, W, H):
    
    overlap = False
    
    for j in range(i):
        x, y, w, h = Boxes[j][0][0], Boxes[j][0][1], Boxes[j][1][0]-Boxes[j][0][0], Boxes[j][1][1]-Boxes[j][0][1]
        x_overlap = min(x+w, X+W) - max(x, X)
        y_overlap = min(y+h, Y+H) - max(y, Y)
        area = x_overlap * y_overlap
        
        if area > min(w * h, W * H, WIDTH * HEIGHT / 20) / 3 and x_overlap > 0 and y_overlap > 0:
            overlap = True
            
    return overlap

# Begin processing the frames, stop until one of two stopping conditions is met

while True:
    _, frame = vs.read()
    
    # First stopping condition: no more frames
    
    if frame is None:
        break

    # Resize to improve computation speed

    frame = imutils.resize(frame, width=WIDTH)

    # Frame number plus one

    count += 1

    # If this is the first frame

    if count == 1:

        # Create multitracker
        
        trackers = cv2.MultiTracker_create()

        # Objection detection
        
        Boxes, classes = get_prediction(frame, threshold=0.7)
        
        for i in range(len(Boxes)):
            if classes[i] == 'car' or classes[i] == 'truck':
                X, Y, W, H = Boxes[i][0][0], Boxes[i][0][1], Boxes[i][1][0]-Boxes[i][0][0], Boxes[i][1][1]-Boxes[i][0][1]

                overlap = OverlapWithNew(i, Boxes, X, Y, W, H)
                leaving = False

                if X > WIDTH * 0.7:
                    leaving = True

                if overlap == False and leaving == False:
                    tracker = cv2.TrackerCSRT_create()
                    trackers.add(tracker, frame, (X, Y, W, H))
                    counted.append(0)

    # Every 10 frames we do a objection detection again

    if count % 10 == 1 and count > 1:
        Boxes, classes = get_prediction(frame, threshold=0.7)
        for i in range(len(Boxes)):
            if classes[i] == 'car' or classes[i] == 'truck':
                X, Y, W, H = Boxes[i][0][0], Boxes[i][0][1], Boxes[i][1][0]-Boxes[i][0][0], Boxes[i][1][1]-Boxes[i][0][1]
                
                small = False
                leaving = False
                
                if W < MINIMAL_WIDTH or H < MINIMAL_HEIGHT or W * H < MINIMAL_AREA:
                    small = True

                overlap = OverlapWithNew(i, Boxes, X, Y, W, H)
                    
                for box in boxes_new: # check old boxes
                    (x, y, w, h) = [int(v) for v in box]
                    x_overlap = min(x+w, X+W) - max(x, X)
                    y_overlap = min(y+h, Y+H) - max(y, Y)
                    area = x_overlap * y_overlap
                    if area > min(w * h, W * H, WIDTH * HEIGHT / 20) / 3 and x_overlap > 0 and y_overlap > 0:
                        overlap = True

                if X > WIDTH * 0.7:
                    leaving = True
                
                if overlap == False and small == False and leaving == False:
                    tracker = cv2.TrackerCSRT_create()
                    trackers.add(tracker, frame, (X, Y, W, H))
                    counted.append(0)

    # Update new positions
    
    (success, boxes_new) = trackers.update(frame)

    for index, box in enumerate(boxes_new):
        (x, y, w, h) = [int(v) for v in box]

        DisplayCount(frame, vehicle_burnaby, vehicle_coquitlam)

        # Check if the vehicle has blocked by the traffic sign
        
        visable = True
        if x > w_left and x < w_right:
            if y > h_bottom and y < h_top:
                visable = False

        # Draw a green rectangle if the vehicle has not left, is not too small, and is not blocked by the traffic sign
                
        if x > WIDTH / 20 and y > HEIGHT / 4 and w * h > MINIMAL_AREA and visable:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Coquitlam + 1 if the vehicle left from the bottom left corner

        if x < WIDTH / 20 and w * h > MINIMAL_AREA:
            if counted[index] == 0:
                counted[index] = 1
                vehicle_coquitlam += 1

        # Burnaby + 1 if the vehicle left from the middle
        
        if y < HEIGHT / 4 and w * h > MINIMAL_AREA:
            if counted[index] == 0:
                counted[index] = 1
                vehicle_burnaby += 1
                
        DisplayCount(frame, vehicle_burnaby, vehicle_coquitlam)
        
    # Save to output video
    
    image = cv2.resize(frame, (1280, 720))
    out.write(image)

    # Show result on screen
    
    cv2.imshow("Frame", frame)

    # Second stopping condition: press ESC

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
vs.release()
out.release()
