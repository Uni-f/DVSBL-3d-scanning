import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
from measurement_utils import calculate_distance, pixels_to_cm
from pose_detector import PoseDetector
from data_manager import DataManager

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize pose detector
    pose_detector = PoseDetector()
    
    # Initialize data manager
    data_manager = DataManager('measurements.csv')
    
    # Reference object size in cm (e.g., A4 paper width)
    REFERENCE_OBJECT_SIZE_CM = 29.7
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
            
        # Process the frame and get landmarks
        frame, landmarks = pose_detector.process_frame(frame)
        
        if landmarks is not None and landmarks.size > 0:

            # Calculate reference pixels (distance between shoulders)
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_distance_pixels = calculate_distance(left_shoulder, right_shoulder)
            
            # Calculate pixel to cm ratio
            if REFERENCE_OBJECT_SIZE_CM is not None:
                pixels_per_cm = shoulder_distance_pixels / REFERENCE_OBJECT_SIZE_CM

            
            # Calculate and draw measurements
            measurements = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'shoulder_width': pixels_to_cm(shoulder_distance_pixels, pixels_per_cm),
                'hip_width': pixels_to_cm(
                    calculate_distance(
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
                    ),
                    pixels_per_cm
                ),
                'right_leg_length': pixels_to_cm(
                    calculate_distance(
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
                    ),
                    pixels_per_cm
                ),
                'left_leg_length': pixels_to_cm(
                    calculate_distance(
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
                    ),
                    pixels_per_cm
                )
            }
            
            # Draw measurements on frame
            pose_detector.draw_measurements(frame, landmarks, measurements)
            
            # Save measurements
            data_manager.save_measurements(measurements)
        
        # Display the frame
        cv2.imshow('Body Measurements', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
