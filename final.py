import cv2
import numpy as np
from bt import BT  # Bluetooth 클래스 불러오기

import time
import sys

# === 블루투스 연결 ===
bt = BT()
devices = bt.find_devices()
if len(devices) == 0:
    print("No Bluetooth devices found.")
    sys.exit()

print("Available Bluetooth devices:")
for i, dev in enumerate(devices):
    print(f"{i}: {dev['name']} ({dev['addr']})")

try:
    selected = int(input("Select device number to connect: "))
    target = devices[selected]
except (ValueError, IndexError):
    print("Invalid selection.")
    sys.exit()

sock = bt.connect(target['addr'])
if sock is None:
    print("Bluetooth connection failed.")
    sys.exit()
print(f"Connected to {target['name']}")

# === YOLO 설정 ===
cfg_file = '/home/team6/yolo4-tiny/yolov4-tiny.cfg'
weights_file = '/home/team6/yolo4-tiny/yolov4-tiny.weights'
names_file = '/home/team6/yolo4-tiny/coco.names'

with open(names_file, 'r') as f:
    class_names = f.read().strip().split('\n')

net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

input_width, input_height = 160, 160
frame_skip = 5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 오류!")
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        cv2.imshow("YOLOv4-tiny - Human Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes:
            i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            x, y, w_, h_ = boxes[i]
            label_class = class_names[class_ids[i]]
            if label_class == "person":
                person_center_x = x + w_ // 2
                frame_center_x = w // 2
                offset = person_center_x - frame_center_x

                if abs(offset) > 10:
                    direction = "LEFT" if offset < 0 else "RIGHT"
                    message = f"{direction} {abs(offset)}"
                    print("Send:", message)
                    try:
                        sock.send(message.encode())
                    except:
                        print("Bluetooth 전송 실패")

            label = f"{label_class}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv4-tiny - Human Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()