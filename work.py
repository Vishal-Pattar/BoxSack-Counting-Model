import cv2
from ultralytics import YOLO, solutions
from datetime import datetime

model = YOLO("./models/best-v2-s20-2.pt")
cap = cv2.VideoCapture("./sample_videos/sample1.mp4")
assert cap.isOpened(), "Error reading video file"

# Define line points
line_points = [(0, 250), (640, 250)]  # Use for horizontal Line
# line_points = [(100, 0), (100, 640)]  # Use for Vertical Line

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

# Skip frames configuration
frame_skip = 4  # Process every 4th frame
frame_count = 0

# File to store logs
log_file = open('object_logs.txt', 'w')  # Open file in write mode

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Increment frame count
    frame_count += 1

    # Skip frames
    if frame_count % frame_skip != 0:
        continue

    tracks = model.track(frame, persist=True, show=False)

    # Check if data is present
    if counter.class_wise_count:
        for obj_class, counts in counter.class_wise_count.items():
            if 'IN' in counts and 'OUT' in counts:
                obj_in = counts['IN']
                obj_out = counts['OUT']
                if obj_in > 0 or obj_out > 0:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    obj_id = counter.count_ids
                    log_entry = f"{timestamp} - ID: {obj_id}, Class: {obj_class}, InCount: {obj_in}, OutCount: {obj_out}"
                    print(log_entry)

                    # Write log entry to file
                    log_file.write(log_entry + "\n")

    # Display the frame
    frame = counter.start_counting(frame, tracks)
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping detection.")
        break

# Close log file
log_file.close()
print(f"Total In: {counter.in_counts}, Total Out: {counter.out_counts}")

cap.release()
cv2.destroyAllWindows()