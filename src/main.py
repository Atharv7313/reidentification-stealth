import cv2
import os
from detector import PlayerDetector
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv11 detector
detector = PlayerDetector('best.pt')

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Input/output setup
cap = cv2.VideoCapture('videos/15sec_input_720p.mp4')
os.makedirs("detections", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('detections/output.mp4', fourcc, 30.0, (1280, 720))

# Color palette for consistent ID coloring
import random
id_colors = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detector.detect(frame)  # [(x1, y1, x2, y2), ...]

    # Convert boxes to Deep SORT format: [x, y, w, h]
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], 0.99, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if track_id not in id_colors:
            id_colors[track_id] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        color = id_colors[track_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
