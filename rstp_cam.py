import cv2
from urllib.parse import quote
import time

username = quote("@clarence123")
password = quote("@clarence123")   # encode even if same
ip = "192.168.0.93"

RTSP_URL = f"rtsp://{username}:{password}@{ip}:554/stream1"

print("Opening RTSP stream...")
print(f"URL: rtsp://{username}:{password}@{ip}:554/stream1")

# Set buffer size and other properties for better RTSP handling
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Set buffer size to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Give the connection time to establish
time.sleep(2)

if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    print("Trying alternative stream paths...")
    
    # Try alternative stream paths
    alternative_paths = ["/stream2", "/h264", "/live", ""]
    for path in alternative_paths:
        alt_url = f"rtsp://{username}:{password}@{ip}:554{path}"
        print(f"Trying: {alt_url}")
        cap = cv2.VideoCapture(alt_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(2)
        if cap.isOpened():
            print(f"Successfully connected to: {alt_url}")
            RTSP_URL = alt_url
            break
    
    if not cap.isOpened():
        raise RuntimeError("Cannot open RTSP stream with any path")

print("Stream opened successfully")
print("Press 'q' to quit")

frame_count = 0
consecutive_failures = 0
max_failures = 30  # Allow some frame drops before giving up

while True:
    ret, frame = cap.read()
    
    if not ret:
        consecutive_failures += 1
        print(f"Frame drop ({consecutive_failures}/{max_failures})")
        
        if consecutive_failures >= max_failures:
            print("Too many consecutive failures. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(2)
            consecutive_failures = 0
            if not cap.isOpened():
                print("Reconnection failed. Exiting.")
                break
        continue
    
    consecutive_failures = 0
    frame_count += 1
    
    if frame_count % 30 == 0:  # Print status every 30 frames
        print(f"Frames received: {frame_count}")
    
    cv2.imshow("Tapo C500", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stream closed")
