import cv2
from ultralytics import solutions

record = cv2.VideoCapture("parking1.mp4")
if record.isOpened() == False:
    print("Video did not open")
    exit()

width = record.get(cv2.CAP_PROP_FRAME_WIDTH)
height = record.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = record.get(cv2.CAP_PROP_FPS)

recorder = cv2.VideoWriter("parking_mgt.mp4", cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(width), int(height)))

parking_mngt = solutions.ParkingManagement(model = "yolo11n.pt", json_file = "bounding_boxes.json")

while record.isOpened():

    state, scene = record.read()

    if state:

        output = parking_mngt.process(scene)

        recorder.write(output)
    
    else:
        break
    


record.release()
recorder.release()
cv2.destroyAllWindows()