import cv2
from ultralytics import YOLO


model = YOLO('best.pt')

# Initialise the Webcam

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, conf = 0.1, verbose=False)

    # Get the first result object, which contains the predictions
    result = results[0]

    # Loop through each detected object in the frame
    for box in result.boxes:
        # Get coordinates of the bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        # Get the confidence score of the prediction
        confidence = box.conf[0]
        # Get the class id of the prediction
        class_id = int(box.cls[0])
        # Get the class name from the model's name mapping
        class_name = model.names[class_id]

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw the label and confidence score above the box
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame in a window
    cv2.imshow('Real-Time YOLOv8 Detection', frame)

    # Break the loop and close the window if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
