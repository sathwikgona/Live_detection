import cv2
import winsound
from flask import Flask, Response

# -------------------------
# Alarm function (Windows beep)
# -------------------------
def play_alarm():
    frequency = 2500  # Hz
    duration = 1000   # milliseconds
    winsound.Beep(frequency, duration)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Haar cascade for face detection
# -------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# -------------------------
# Video streaming generator
# -------------------------
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Alarm if more than 1 face
        if len(faces) > 1:
            play_alarm()

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display face count on the frame
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------------------------
# Flask route for video feed
# -------------------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)
