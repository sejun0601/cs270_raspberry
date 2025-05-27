import cv2
import numpy as np

# 경로 설정 (당신의 파일 위치에 맞게 바꾸세요)
cfg_file = '/home/team6/yolo4-tiny/yolov4-tiny.cfg'
weights_file = '/home/team6/yolo4-tiny/yolov4-tiny.weights'
names_file = '/home/team6/yolo4-tiny/coco.names'

# 클래스 이름 불러오기
with open(names_file, 'r') as f:
    class_names = f.read().strip().split('\n')

# 네트워크 초기화
net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 카메라 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# YOLO 입력 사이즈
input_width, input_height = 320, 320

# 프레임 스킵 설정
frame_skip = 3
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 오류!")
        break

    frame_count += 1

    # 스킵 프레임: 추론하지 않고 그대로 보여주기
    if frame_count % frame_skip != 0:
        cv2.imshow("YOLOv4-tiny - Human Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    h, w = frame.shape[:2]

    # 블롭 만들기
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)

    # 출력 레이어 추출
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    # 결과 처리
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.0:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대 억제 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes:
            i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            x, y, w_, h_ = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv4-tiny - Human Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()