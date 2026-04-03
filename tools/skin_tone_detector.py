import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

class SkinToneDetector:
    def __init__(self):
        # Initialize MediaPipe face detection and mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam!")
            exit(1)
        
        # Set webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Buffer for smoothing skin tone detection
        self.skin_tone_buffer = deque(maxlen=10)
        
        # Skin tone classification thresholds (HSV-based)
        self.skin_tone_categories = {
            'Fair': {'h_min': 0, 'h_max': 30, 's_min': 0, 's_max': 50, 'v_min': 200, 'v_max': 255},
            'Light': {'h_min': 0, 'h_max': 30, 's_min': 0, 's_max': 80, 'v_min': 150, 'v_max': 220},
            'Medium': {'h_min': 0, 'h_max': 30, 's_min': 20, 's_max': 120, 'v_min': 100, 'v_max': 180},
            'Olive': {'h_min': 15, 'h_max': 45, 's_min': 30, 's_max': 100, 'v_min': 80, 'v_max': 160},
            'Brown': {'h_min': 0, 'h_max': 30, 's_min': 40, 's_max': 150, 'v_min': 60, 'v_max': 140},
            'Dark': {'h_min': 0, 'h_max': 30, 's_min': 60, 's_max': 200, 'v_min': 20, 'v_max': 100}
        }
    
    def get_skin_regions(self, landmarks, frame_shape):
        """Extract skin regions (cheeks and forehead) from MediaPipe facial landmarks"""
        height, width = frame_shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        points = np.array([(int(landmark.x * width), int(landmark.y * height)) 
                          for landmark in landmarks.landmark])
        
        # MediaPipe face mesh landmark indices for skin regions
        # Left cheek region (around landmark 116, 117, 118, 119, 120, 121, 126, 142, 36, 37, 38, 39, 40, 41)
        left_cheek_indices = [116, 117, 118, 119, 120, 121, 126, 142, 36, 37, 38, 39, 40, 41]
        left_cheek_points = points[left_cheek_indices]
        
        # Right cheek region (around landmark 345, 346, 347, 348, 349, 350, 355, 371, 266, 267, 268, 269, 270, 271)
        right_cheek_indices = [345, 346, 347, 348, 349, 350, 355, 371, 266, 267, 268, 269, 270, 271]
        right_cheek_points = points[right_cheek_indices]
        
        # Forehead region (around landmark 10, 151, 9, 8, 107, 55, 65, 52, 53, 46)
        forehead_indices = [10, 151, 9, 8, 107, 55, 65, 52, 53, 46]
        forehead_points = points[forehead_indices]
        
        return {
            'left_cheek': left_cheek_points,
            'right_cheek': right_cheek_points,
            'forehead': forehead_points
        }
    
    def extract_skin_color(self, frame, skin_regions):
        """Extract average skin color from defined regions"""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        all_skin_pixels = []
        
        for region_name, points in skin_regions.items():
            # Create mask for the region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Create a bounding rectangle for the region
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            
            # Expand the region slightly to capture more skin
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Extract pixels from the region
            region_bgr = frame[y:y+h, x:x+w]
            region_hsv = hsv_frame[y:y+h, x:x+w]
            
            # Filter for skin-like colors using HSV ranges
            # Skin typically has: H: 0-30, S: 20-150, V: 50-255
            lower_skin = np.array([0, 20, 50], dtype=np.uint8)
            upper_skin = np.array([30, 150, 255], dtype=np.uint8)
            
            skin_mask = cv2.inRange(region_hsv, lower_skin, upper_skin)
            
            # Get skin pixels
            skin_pixels_bgr = region_bgr[skin_mask > 0]
            skin_pixels_hsv = region_hsv[skin_mask > 0]
            
            if len(skin_pixels_bgr) > 0:
                all_skin_pixels.extend(skin_pixels_bgr)
        
        if len(all_skin_pixels) > 0:
            # Calculate average color
            avg_bgr = np.mean(all_skin_pixels, axis=0)
            avg_rgb = avg_bgr[::-1]  # Convert BGR to RGB
            
            # Convert to HSV
            avg_bgr_uint8 = np.uint8([[avg_bgr]])
            avg_hsv = cv2.cvtColor(avg_bgr_uint8, cv2.COLOR_BGR2HSV)[0][0]
            
            return avg_rgb, avg_hsv
        else:
            return None, None
    
    def classify_skin_tone(self, hsv_values):
        """Classify skin tone based on HSV values"""
        if hsv_values is None:
            return "Unknown"
        
        h, s, v = hsv_values
        
        # Check each category
        for category, thresholds in self.skin_tone_categories.items():
            if (thresholds['h_min'] <= h <= thresholds['h_max'] and
                thresholds['s_min'] <= s <= thresholds['s_max'] and
                thresholds['v_min'] <= v <= thresholds['v_max']):
                return category
        
        # If no exact match, find the closest category based on V (brightness)
        if v > 180:
            return "Fair"
        elif v > 140:
            return "Light"
        elif v > 100:
            return "Medium"
        elif v > 80:
            return "Olive"
        elif v > 50:
            return "Brown"
        else:
            return "Dark"
    
    def draw_landmarks_and_regions(self, frame, landmarks, skin_regions):
        """Draw facial landmarks and skin regions on the frame"""
        # Draw MediaPipe face mesh landmarks
        self.mp_drawing.draw_landmarks(
            frame, landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
            None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        
        # Draw skin regions
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors
        
        for i, (region_name, points) in enumerate(skin_regions.items()):
            # Create bounding rectangle
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i % len(colors)], 2)
            
            # Add region label
            cv2.putText(frame, region_name, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)
    
    def run(self):
        """Main loop for real-time skin tone detection"""
        print("Starting skin tone detection...")
        print("Press 'q' to quit")
        print("-" * 50)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and get landmarks
            face_results = self.face_mesh.process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                # Use the first detected face
                landmarks = face_results.multi_face_landmarks[0]
                
                # Extract skin regions
                skin_regions = self.get_skin_regions(landmarks, frame.shape)
                
                # Extract skin color
                rgb_values, hsv_values = self.extract_skin_color(frame, skin_regions)
                
                if rgb_values is not None and hsv_values is not None:
                    # Classify skin tone
                    skin_tone = self.classify_skin_tone(hsv_values)
                    
                    # Add to buffer for smoothing
                    self.skin_tone_buffer.append(skin_tone)
                    
                    # Get most common skin tone from buffer
                    if len(self.skin_tone_buffer) > 0:
                        most_common_tone = max(set(self.skin_tone_buffer), key=self.skin_tone_buffer.count)
                        
                        # Print results to console
                        print(f"\rSkin Tone: {most_common_tone} | RGB: {rgb_values.astype(int)} | HSV: {hsv_values}", end="", flush=True)
                        
                        # Draw landmarks and regions
                        self.draw_landmarks_and_regions(frame, landmarks, skin_regions)
                        
                        # Add skin tone text overlay
                        cv2.putText(frame, f"Skin Tone: {most_common_tone}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"RGB: {rgb_values.astype(int)}", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"HSV: {hsv_values}", 
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Skin Tone Detection', frame)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_detection.close()
        self.face_mesh.close()
        print("\nSkin tone detection stopped.")

if __name__ == "__main__":
    detector = SkinToneDetector()
    detector.run()
