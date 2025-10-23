import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load webcam
cap = cv2.VideoCapture(0)

# Accuracy tracking variables
total_predictions = 0
correct_predictions = 0
captured_image = None
captured_sign = ""
expected_sign = None  # First captured sign will be used as expected

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ab = b - a
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Function to recognize hand signs
def recognize_sign(landmarks):
    thumb_angle = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
    index_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[8])
    middle_angle = calculate_angle(landmarks[9], landmarks[10], landmarks[12])
    ring_angle = calculate_angle(landmarks[13], landmarks[14], landmarks[16])
    pinky_angle = calculate_angle(landmarks[17], landmarks[18], landmarks[20])
    
    if pinky_angle > 120 and index_angle < 40 and middle_angle < 40 and ring_angle < 40:
        return "Hello"
    elif index_angle > 120 and middle_angle > 120 and ring_angle > 120 and pinky_angle < 50:
        return "What is your name?"
    elif thumb_angle < 30 and index_angle < 30 and middle_angle < 30 and ring_angle < 30 and pinky_angle < 30:
        return "Stop"
    elif thumb_angle < 30 and index_angle > 120 and middle_angle > 120:
        return "Thank You"
    elif index_angle < 30 and middle_angle < 30 and ring_angle < 30 and pinky_angle < 30:
        return "Please"
    elif index_angle > 120 and middle_angle > 120 and ring_angle > 120 and pinky_angle > 120:
        return "Bye"
    elif thumb_angle < 40 and index_angle > 120 and pinky_angle > 120 and middle_angle < 40 and ring_angle < 40:
        return "I Love You"
    elif index_angle < 30 and middle_angle > 120 and ring_angle > 120:
        return "Yes"
    elif index_angle > 120 and middle_angle > 120 and thumb_angle < 40:
        return "No"
    elif thumb_angle > 120 and index_angle < 40 and middle_angle < 40:
        return "Good Job"
    
    return "Unknown"

# Start Video Capture Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    detected_sign = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_sign = recognize_sign(hand_landmarks.landmark)

    # Capture image when 'c' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and detected_sign != "Unknown":
        captured_image = frame.copy()
        captured_sign = detected_sign.lower()  # Convert detected sign to lowercase

        if expected_sign is None:
            expected_sign = captured_sign  # First captured sign is set as expected
            print(f"✅ First captured sign: '{expected_sign}' is set as the expected sign.")
        else:
            total_predictions += 1  # Increase total predictions
            print(f"Expected Sign: {expected_sign}, Captured Sign: {captured_sign}")

            # Check if detected sign matches expected sign
            if captured_sign == expected_sign:
                correct_predictions += 1
                print("✅ Match! Increasing correct predictions.")
            else:
                print("❌ No match!")

    # Display recognized sign in real-time
    cv2.putText(frame, f"Sign: {detected_sign}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Hand Sign Detection", frame)

    # Show captured image with accuracy
    if captured_image is not None and expected_sign is not None:
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"Total: {total_predictions}, Correct: {correct_predictions}, Accuracy: {accuracy:.2f}%")

        # Draw the detected sign and accuracy on the captured image
        cv2.putText(captured_image, f"Captured Sign: {captured_sign}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(captured_image, f"Accuracy: {accuracy:.2f}%", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Captured Image", captured_image)

    if key == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
