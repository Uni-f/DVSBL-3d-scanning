import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def process_frame(self, frame):
        """Process frame and return landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get landmarks
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is not None:
            # Draw pose landmarks
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                                for lm in results.pose_landmarks.landmark])
            return frame, landmarks
            
        return frame, None
    
    def draw_measurements(self, frame, landmarks, measurements):
    # """ Draw measurements on the frame"""
    # Font settings
    
       font = cv2.FONT_HERSHEY_SIMPLEX
       font_scale = 0.5
       color = (255, 255, 255)  # White color
       thickness = 2

    # Display measurements as text
       y_position = 30  # Starting vertical position for text
       for measure, value in measurements.items():
            if measure != 'timestamp':
                 text = f"{measure}: {value:.2f} cm"
                 cv2.putText(frame, text, (10, y_position), font, font_scale, color, thickness)
                 y_position += 30  # Increment position for the next text line

    # Example: Add logic for drawing lines here if needed
    # if you need to draw specific landmarks or connections
       if landmarks is not None and len(landmarks) > 1:
        # Example: Drawing a line between two landmarks
         start_point = (int(landmarks[0][0]), int(landmarks[0][1]))
         end_point = (int(landmarks[1][0]), int(landmarks[1][1]))
         cv2.line(frame, start_point, end_point, (0, 255, 0), thickness)
