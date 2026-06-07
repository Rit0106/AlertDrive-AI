from ultralytics import YOLO
import cv2
import logging
import time
import winsound
import os

# Suppress Ultralytics debug logs
logging.getLogger('ultralytics').setLevel(logging.ERROR)


# Define model paths using os.path.join
base_dir = os.path.dirname(os.path.abspath(__file__))

eye_model_path = os.path.join(base_dir, "model", "eye", "runs", "detect", "train3", "weights", "best.pt")
yawn_model_path = os.path.join(base_dir, "model", "yawn", "runs", "detect", "train", "weights", "best.pt")

# Load YOLOv8 models:
# model1 is for eye state detection (e.g., open/closed)
# model2 is for yawn detection

model1 = YOLO(eye_model_path)
model2 = YOLO(yawn_model_path)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Initialize counters
count = 0       # Eye closure count
ycount = 0      # Yawn detection count
frame = 0       # Frame counter
f = 2000        # Frequency for buzzer
t = 1000         # Duration of buzzer beep (ms)
tired = 0       # Tiredness level counter

drowsy_start_time = None
emergency_triggered = False

# Function to trigger a buzzer and print warning
def buzzer():
    winsound.Beep(f, t)
    print("Take a break")
def calculate_fatigue_score(count, ycount, tired):

    score = 0

    score += min(count // 2, 40)

    score += min(ycount // 5, 30)

    score += min(tired * 10, 30)

    return min(score, 100)


def get_status(score):

    if score < 30:
        return "ALERT"

    elif score < 60:
        return "SLIGHTLY TIRED"

    elif score < 80:
        return "DROWSY"

    else:
        return "CRITICAL"

while True:
    frame += 1  # Count each frame

    # Read a frame from the webcam
    success, img = cap.read()
    if not success or img is None:
        print("Failed to read frame or video ended.")
        break

    # Apply eye state detection (model1)
    results1 = model1.track(img, stream=True, conf=0.5, persist=True)
    for r in results1:
        img = r.plot()  # Draw results on frame
        boxes = r.boxes
        for box in boxes:

            cls = int(box.cls)
            print("Detected class:", cls)

    # Eyes closed
            if cls == 0:

                count += 1

                print(f"Eye Closure Count: {count}")

        # Drowsiness detected
                if count > 100:

                    buzzer()

            # Start timer only once
                    if drowsy_start_time is None:
                        drowsy_start_time = time.time()

            # Calculate elapsed time
                    elapsed = time.time() - drowsy_start_time

                    print(f"Drowsy for {int(elapsed)} seconds")

            # Emergency trigger after 20 seconds
                    if elapsed > 20 and not emergency_triggered:

                        print("🚨 EMERGENCY ALERT TRIGGERED 🚨")

                        emergency_triggered = True

    # Eyes open
            else:

                count -= 5

                if count < 0:
                    count = 0

                print(f"Eye Closure Count: {count}")

        # Reset if driver wakes up
                drowsy_start_time = None
                emergency_triggered = False
    print(f"Eye Closure Count: {count}")

    # Apply yawn detection (model2)
    results2 = model2.track(img, stream=True, persist=True, conf=0.8)
    for r in results2:
        img = r.plot()  # Draw results on frame
        boxes = r.boxes
        if len(boxes) > 0:
            ycount += 1  # Increment yawn counter when yawning is detected

        # Trigger alerts based on yawn count and tired level
        if ycount > 120 or tired > 3:
            buzzer()
        elif ycount > 90:
            print("You are tired")
            tired += 1

        # Reset counters after a set number of frames
        if frame > 1800:
            frame = 0
            ycount = 0
        if frame > 9000:
            tired = 0

        # Optional debug print
        # print(f"Frame: {frame}, Yawn Count: {ycount}")

    # Show the processed video feed
        fatigue_score = calculate_fatigue_score(
        count,
        ycount,
        tired
    )

    status = get_status(fatigue_score)

    cv2.putText(
        img,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Fatigue Score: {fatigue_score}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
