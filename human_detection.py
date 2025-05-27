import cv2

cap = cv2.VideoCapture(0)

cascade_path = '/home/team6/opencv_cascades/haarcascade_fullbody.xml'
body_cascade = cv2.CascadeClassifier(cascade_path)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    
    cv2.imshow("Human Detection", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()