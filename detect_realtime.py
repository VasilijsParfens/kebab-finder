import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/kebab_finder/weights/best.pt')  # Path to your trained weights

# Path to the pre-recorded video
video_path = 'videos/test.mp4'  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for saving the output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete!")
        break

    # Run YOLO inference on the frame
    results = model.predict(frame, conf=0.5, show=False)

    # Draw bounding boxes and labels
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score
        label = int(box.cls[0])  # Class ID
        class_name = model.names[label]  # Class name

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Kebab Shop Detector', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
