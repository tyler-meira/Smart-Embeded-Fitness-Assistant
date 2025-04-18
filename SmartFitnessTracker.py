import cv2
import time
import math
import threading
import speech_recognition as sr
import numpy as np
import google.generativeai as genai
import os
from sense_hat import SenseHat
from PIL import Image
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

#Gemnai Setup
genai.configure(api_key='AIzaSyCXXNHMYrYP3XGwsRoMHdxvb_GPiOHJR9A')
model = genai.GenerativeModel('models/gemini-1.5-pro')

# MoveNet model and Edge TPU setup
MODEL_PATH = 'movenet_single_pose_lightning_ptq_edgetpu.tflite'
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

NUM_KEYPOINTS = 17

REPS_PER_SET = 3

# Sense HAT setup
sense = SenseHat()

# Color definitions (corrected)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)    # Fixed from (255,0,0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)

# Digit patterns (3x5 grid)
digits = {
    '0': [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1],
    '1': [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1],
    '2': [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1],
    '3': [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1],
    '4': [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1],
    '5': [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1],
    '6': [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1],
    '7': [1,1,1, 0,0,1, 0,1,0, 1,0,0, 1,0,0],
    '8': [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1],
    '9': [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1]
}

# Exercise tracking
totalReps = 0
leftReps = 0 
rightReps = 0
squatReps = 0

leftSets = 0
rightSets = 0
squatSets = 0

rightExtended = False
leftExtended = False
squatExtended = False

stateStarted = True

# Exercise states
stateLeftReps = True
stateRightReps = False
stateSquatReps = False

# Angle thresholds
right_min_angle = float('inf')
right_max_angle = 0
left_min_angle = float('inf')
left_max_angle = 0
squat_min_angle = float('inf')
squat_max_angle = 0

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

response = ""

# Speech recognizer
recognizer = sr.Recognizer()


def update_display(count, color):
    """Update Sense HAT with count (0-99)"""
    sense.clear()
    num_str = f"{count:02d}"
    
    global stateStarted, response, leftSets, rightSets, squatSets
    
    #Sets Recomended Number of Sets by gemnai
    if stateLeftReps and stateStarted:
        stateStarted = False
        leftReps = 0
        response = send_gemini('Can you return me a single digit number for the reccomnded amount of sets to complete for bicept curls (ONLY THE NUMEBR NO OTHER TEXT)')
        print(response)
        for col in range(int(response)):
            sense.set_pixel(col,7,ORANGE)
    elif stateRightReps and stateStarted:
        stateStarted = False
        rightReps = 0
        response = send_gemini('Can you return me a single digit number for the reccomnded amount of sets to complete for bicept curls (ONLY THE NUMEBR NO OTHER TEXT)')
        print(response)
        for col in range(int(response)):
            sense.set_pixel(col,7,ORANGE)
    elif stateSquatReps and stateStarted:
        stateStarted = False
        squatSets = 0
        response = send_gemini('Can you return me a high amount of sets for squats for a low amount of reps (ONLY ONE NUMBER NO OTHER TEXT) in the range of 4 to 6')
        print(response)
        for col in range(int(response)):
            sense.set_pixel(col,7,ORANGE)
    else:
        for col in range(int(response)):
            
            if stateLeftReps:
                if col < leftSets:
                    sense.set_pixel(col, 7, GREEN)
                else:
                    sense.set_pixel(col, 7, ORANGE)
            elif stateRightReps:
                if col < rightSets:
                    sense.set_pixel(col, 7, GREEN)
                else:
                    sense.set_pixel(col, 7, ORANGE)
            elif stateSquatReps:
                if col < squatSets:
                    sense.set_pixel(col, 7, GREEN)
                else:
                    sense.set_pixel(col, 7, ORANGE)
    
    # First digit (tens place)
    for y in range(5):
        for x in range(3):
            if digits[num_str[0]][y*3 + x]:
                sense.set_pixel(x+1, y+1, color)
    
    # Second digit (ones place)
    for y in range(5):
        for x in range(3):
            if digits[num_str[1]][y*3 + x]:
                sense.set_pixel(x+5, y+1, color)
                
def send_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

def listen_for_reset():
    global totalReps, leftReps, rightReps, squatReps
    global stateLeftReps, stateRightReps, stateSquatReps, stateStarted
    
    while True:
        global stateStarted
        with sr.Microphone() as source:
            print("Listening for commands...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                
                if "reset" in text:
                    sense.clear()
                    leftSets = rightSets = squatSets = 0
                    totalReps = leftReps = rightReps = squatReps = 0
                    if stateLeftReps:
                        update_display(0, BLUE)
                    elif stateRightReps:
                        update_display(0, GREEN)
                    else:
                        update_display(0, WHITE)
                    print("Counts reset!")
                
                elif "next" in text:
                    if stateLeftReps:
                        stateLeftReps, stateRightReps = False, True
                        stateStarted = True
                        update_display(rightReps, GREEN)
                    elif stateRightReps:
                        stateRightReps, stateSquatReps = False, True
                        stateStarted = True
                        update_display(leftReps, WHITE)
                    else:
                        stateSquatReps, stateLeftReps = False, True
                        stateStarted = True
                        update_display(leftReps, BLUE)
                    print(f"Switched to {'right arm' if stateRightReps else 'squats' if stateSquatReps else 'left arm'} mode")
                        
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Recognition error: {e}")
            except Exception as e:
                print(f"Error: {e}")

def calculate_angle(a, b, c):
    """Calculate angle between three points (a, b, c) where b is vertex"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return np.degrees(angle)

# Start listening thread
reset_thread = threading.Thread(target=listen_for_reset, daemon=True)
reset_thread.start()

# Initialize display
update_display(0, BLUE)
prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break
     
    # Pose estimation
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img_pil.resize(common.input_size(interpreter), Image.LANCZOS)
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    pose = common.output_tensor(interpreter, 0).copy().reshape(NUM_KEYPOINTS, 3)

    # Get keypoints
    right_wrist = (pose[10][0], pose[10][1])
    right_elbow = (pose[8][0], pose[8][1])
    right_shoulder = (pose[6][0], pose[6][1])

    left_wrist = (pose[9][0], pose[9][1])
    left_elbow = (pose[7][0], pose[7][1])
    left_shoulder = (pose[5][0], pose[5][1])

    right_hip = (pose[12][0], pose[12][1])
    right_knee = (pose[14][0], pose[14][1])
    right_ankle = (pose[16][0], pose[16][1])

    left_hip = (pose[11][0], pose[11][1])
    left_knee = (pose[13][0], pose[13][1])
    left_ankle = (pose[15][0], pose[15][1])


    # Visualize keypoints
    for i in range(NUM_KEYPOINTS):
        x, y, confidence = pose[i]
        if confidence > 0.5:
            cv2.circle(frame, (int(y * frame.shape[1]), int(x * frame.shape[0])), 5, RED, -1)

    # Right arm rep counting
    if pose[10][2] > 0.5 and pose[8][2] > 0.5 and pose[6][2] > 0.5:
        right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)

        if right_angle < 90 and not rightExtended:
            rightExtended = True
        if right_angle > 120 and rightExtended:
            rightReps += 1
            rightExtended = False
                
               
    # Left arm rep counting
    if pose[9][2] > 0.5 and pose[7][2] > 0.5 and pose[5][2] > 0.5:
        left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)

        if left_angle < 90 and not leftExtended:
            leftExtended = True
        if left_angle > 120 and leftExtended:
            leftReps += 1
            leftExtended = False
                

     # Squat rep counting
    if pose[16][2] > 0.5 and pose[14][2] > 0.5 and pose[12][2] > 0.5:  # Checking confidence of right leg keypoints
        right_squat_angle = calculate_angle(right_hip, right_knee, right_ankle)

        if right_squat_angle < 100 and not squatExtended:  # Angle below 90 degrees (indicating a squat)
            squatExtended = True
        if right_squat_angle > 140 and squatExtended:  # Angle greater than 120 degrees (indicating standing up)
            squatReps += 1
            squatExtended = False

    if pose[15][2] > 0.5 and pose[13][2] > 0.5 and pose[11][2] > 0.5:  # Checking confidence of left leg keypoints
        left_squat_angle = calculate_angle(left_hip, left_knee, left_ankle)
        squat_min_angle = min(squat_min_angle, left_squat_angle)

        if left_squat_angle < 100 and not squatExtended:  # Angle below 90 degrees (indicating a squat)
            squatExtended = True
        if left_squat_angle > 140 and squatExtended:  # Angle greater than 120 degrees (indicating standing up)
            squatReps += 1
            squatExtended = False
            
           
    if stateLeftReps:
        if(leftReps >= REPS_PER_SET):
            leftReps = 0
            leftSets += 1
            update_display(leftReps, BLUE)
        else:
            update_display(leftReps, BLUE)
    elif stateRightReps:
        if(rightReps >= REPS_PER_SET):
            rightReps = 0
            rightSets += 1
            update_display(rightReps, RED)
        else:
            update_display(rightReps, RED)
    elif stateSquatReps:
        if(squatReps >= REPS_PER_SET):
            squatReps = 0
            squatSets += 1
            update_display(squatReps, WHITE)
        else:
            update_display(squatReps, WHITE)
                
         
#     if(leftReps >= REPS_PER_SET):
#         leftReps = 0
#         leftSets += 1
#         update_display(leftReps, BLUE)
# 
#     if(rightReps >= REPS_PER_SET):
#         rightReps = 0
#         rightSets += 1
#         update_display(rightReps, GREEN)
# 
#     if(squatReps >= REPS_PER_SET):
#         squatReps = 0
#         squatSets += 1
#         update_display(squatReps, WHITE)
# 
# 
#     # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display info on video
    cv2.putText(frame, f'Left Reps: {leftReps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(frame, f'Right Reps: {rightReps}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    cv2.putText(frame, f'Total Reps: {leftReps + rightReps}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f'Squat Reps: {squatReps}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)  # Added line for squat reps
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Show current mode
    mode = "LEFT ARM" if stateLeftReps else "RIGHT ARM" if stateRightReps else "SQUATS"
    cv2.putText(frame, f'Mode: {mode}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sense.clear()