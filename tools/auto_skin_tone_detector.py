import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class AutoSkinToneDetector:
    def __init__(self):
        # Initialize MediaPipe face detection and mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection and mesh with stricter parameters
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.8)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam!")
            exit(1)
        
        # Set webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Detection stability tracking
        self.detection_start_time = None
        self.stable_detection_duration = 3.0  # 3 seconds
        self.face_detected = False
        self.captured_image = None
        
        # Improved skin tone classification thresholds (HSV-based with lighting normalization)
        # These thresholds are more focused on hue and saturation rather than brightness
        self.skin_tone_categories = {
            'Fair': {'h_min': 0, 'h_max': 20, 's_min': 20, 's_max': 60, 'v_min': 80, 'v_max': 200},
            'Light': {'h_min': 0, 'h_max': 25, 's_min': 30, 's_max': 80, 'v_min': 70, 'v_max': 180},
            'Medium': {'h_min': 0, 'h_max': 25, 's_min': 40, 's_max': 100, 'v_min': 60, 'v_max': 160},
            'Olive': {'h_min': 15, 'h_max': 35, 's_min': 50, 's_max': 120, 'v_min': 50, 'v_max': 140},
            'Brown': {'h_min': 0, 'h_max': 25, 's_min': 60, 's_max': 140, 'v_min': 40, 'v_max': 120},
            'Dark': {'h_min': 0, 'h_max': 25, 's_min': 80, 's_max': 180, 'v_min': 20, 'v_max': 80}
        }
    
    def get_skin_regions(self, landmarks, frame_shape):
        """Extract skin regions (cheeks and forehead) from MediaPipe facial landmarks"""
        height, width = frame_shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        points = np.array([(int(landmark.x * width), int(landmark.y * height)) 
                          for landmark in landmarks.landmark])
        
        # MediaPipe face mesh landmark indices for skin regions
        # Left cheek region
        left_cheek_indices = [116, 117, 118, 119, 120, 121, 126, 142, 36, 37, 38, 39, 40, 41]
        left_cheek_points = points[left_cheek_indices]
        
        # Right cheek region
        right_cheek_indices = [345, 346, 347, 348, 349, 350, 355, 371, 266, 267, 268, 269, 270, 271]
        right_cheek_points = points[right_cheek_indices]
        
        # Forehead region
        forehead_indices = [10, 151, 9, 8, 107, 55, 65, 52, 53, 46]
        forehead_points = points[forehead_indices]
        
        return {
            'left_cheek': left_cheek_points,
            'right_cheek': right_cheek_points,
            'forehead': forehead_points
        }
    
    def normalize_lighting(self, frame):
        """Normalize lighting conditions to reduce brightness effects"""
        # Convert to LAB color space for better lighting normalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels back
        lab_normalized = cv2.merge([l, a, b])
        
        # Convert back to BGR
        normalized_frame = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
        
        return normalized_frame
    
    def validate_face_detection(self, landmarks, frame_shape):
        """Validate that we have detected a proper face with all required landmarks"""
        if not landmarks or not landmarks.landmark:
            return False
        
        height, width = frame_shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        points = np.array([(int(landmark.x * width), int(landmark.y * height)) 
                          for landmark in landmarks.landmark])
        
        # Check for essential facial landmarks
        # Eyes (landmarks 33, 133, 362, 263)
        left_eye = points[33]
        right_eye = points[362]
        
        # Nose (landmark 1)
        nose = points[1]
        
        # Mouth corners (landmarks 61, 291)
        left_mouth = points[61]
        right_mouth = points[291]
        
        # Check if landmarks are within reasonable bounds
        if (left_eye[0] < 0 or left_eye[0] >= width or left_eye[1] < 0 or left_eye[1] >= height or
            right_eye[0] < 0 or right_eye[0] >= width or right_eye[1] < 0 or right_eye[1] >= height or
            nose[0] < 0 or nose[0] >= width or nose[1] < 0 or nose[1] >= height or
            left_mouth[0] < 0 or left_mouth[0] >= width or left_mouth[1] < 0 or left_mouth[1] >= height or
            right_mouth[0] < 0 or right_mouth[0] >= width or right_mouth[1] < 0 or right_mouth[1] >= height):
            return False
        
        # Check face proportions (basic validation)
        # Distance between eyes should be reasonable
        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance < 30 or eye_distance > 200:  # Too close or too far apart
            return False
        
        # Nose should be between eyes
        nose_to_left_eye = np.linalg.norm(nose - left_eye)
        nose_to_right_eye = np.linalg.norm(nose - right_eye)
        if abs(nose_to_left_eye - nose_to_right_eye) > eye_distance * 0.3:
            return False
        
        # Mouth should be below nose
        if left_mouth[1] <= nose[1] or right_mouth[1] <= nose[1]:
            return False
        
        # Check if face is reasonably centered and sized
        face_center_x = (left_eye[0] + right_eye[0] + nose[0]) / 3
        face_center_y = (left_eye[1] + right_eye[1] + nose[1]) / 3
        
        # Face should be reasonably centered (not too far to edges)
        if (face_center_x < width * 0.2 or face_center_x > width * 0.8 or
            face_center_y < height * 0.2 or face_center_y > height * 0.8):
            return False
        
        # Face should be reasonably sized (not too small or too large)
        face_size = max(eye_distance, nose_to_left_eye, nose_to_right_eye)
        if face_size < 40 or face_size > 300:
            return False
        
        return True
    
    def extract_skin_color(self, frame, skin_regions):
        """Extract average skin color from defined regions with lighting normalization"""
        # Normalize lighting first
        normalized_frame = self.normalize_lighting(frame)
        hsv_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2HSV)
        
        all_skin_pixels = []
        all_skin_pixels_hsv = []
        
        for region_name, points in skin_regions.items():
            # Create mask for the region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Create a bounding rectangle for the region
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            
            # Expand the region slightly to capture more skin
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Extract pixels from the region
            region_bgr = normalized_frame[y:y+h, x:x+w]
            region_hsv = hsv_frame[y:y+h, x:x+w]
            
            # More restrictive skin detection to avoid non-skin areas
            # Skin typically has: H: 0-25, S: 30-150, V: 40-220
            lower_skin = np.array([0, 30, 40], dtype=np.uint8)
            upper_skin = np.array([25, 150, 220], dtype=np.uint8)
            
            skin_mask = cv2.inRange(region_hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3,3), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Get skin pixels
            skin_pixels_bgr = region_bgr[skin_mask > 0]
            skin_pixels_hsv = region_hsv[skin_mask > 0]
            
            if len(skin_pixels_bgr) > 0:
                all_skin_pixels.extend(skin_pixels_bgr)
                all_skin_pixels_hsv.extend(skin_pixels_hsv)
        
        if len(all_skin_pixels) > 0:
            # Calculate average color
            avg_bgr = np.mean(all_skin_pixels, axis=0)
            avg_rgb = avg_bgr[::-1]  # Convert BGR to RGB
            
            # Calculate average HSV from normalized image
            avg_hsv = np.mean(all_skin_pixels_hsv, axis=0)
            
            return avg_rgb, avg_hsv
        else:
            return None, None
    
    def classify_skin_tone(self, hsv_values):
        """Classify skin tone based on HSV values with improved algorithm"""
        if hsv_values is None:
            return "Unknown"
        
        h, s, v = hsv_values
        
        # Calculate distance scores for each category
        category_scores = {}
        
        for category, thresholds in self.skin_tone_categories.items():
            # Calculate how well the HSV values fit each category
            h_score = 1.0 if thresholds['h_min'] <= h <= thresholds['h_max'] else 0.0
            s_score = 1.0 if thresholds['s_min'] <= s <= thresholds['s_max'] else 0.0
            v_score = 1.0 if thresholds['v_min'] <= v <= thresholds['v_max'] else 0.0
            
            # Weighted score (hue and saturation are more important than brightness)
            total_score = (h_score * 0.4) + (s_score * 0.4) + (v_score * 0.2)
            category_scores[category] = total_score
        
        # Find the category with the highest score
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        # If the best score is too low, use fallback classification
        if best_score < 0.6:
            # Fallback classification based on hue and saturation (less dependent on brightness)
            if h <= 15 and s <= 50:
                return "Fair"
            elif h <= 20 and s <= 70:
                return "Light"
            elif h <= 25 and s <= 100:
                return "Medium"
            elif 15 <= h <= 35 and s >= 50:
                return "Olive"
            elif h <= 25 and s >= 80:
                return "Brown"
            else:
                return "Dark"
        
        return best_category
    
    def check_lighting_quality(self, frame):
        """Check if lighting is sufficient for skin tone detection with normal brightness"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Calculate standard deviation (contrast)
        std_brightness = np.std(gray)
        
        # Normal brightness conditions for everyday use
        # Accept normal indoor lighting (not too dark, not too bright)
        # Very lenient since we have lighting normalization
        return 30 <= mean_brightness <= 240 and std_brightness > 10
    
    def draw_detection_info(self, frame, time_remaining, lighting_ok):
        """Draw detection status information on frame"""
        # Draw countdown timer
        if time_remaining > 0:
            cv2.putText(frame, f"Hold still for {time_remaining:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Capturing image...", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw lighting status
        lighting_color = (0, 255, 0) if lighting_ok else (0, 0, 255)
        lighting_text = "Good lighting" if lighting_ok else "Poor lighting - adjust lighting"
        cv2.putText(frame, lighting_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, lighting_color, 2)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Main loop for automatic skin tone detection"""
        print("=" * 60)
        print("AUTOMATIC SKIN TONE DETECTOR")
        print("=" * 60)
        print("Instructions:")
        print("1. Position your FACE in the center of the camera")
        print("2. Make sure your entire face is visible")
        print("3. Avoid showing hands, objects, or partial faces")
        print("4. Normal indoor lighting is sufficient")
        print("5. Hold still for 3 seconds")
        print("6. The image will be captured automatically")
        print("7. Press 'q' to quit anytime")
        print("=" * 60)
        print("IMPORTANT: Only faces will be detected - hands or objects will be rejected!")
        print("=" * 60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Check lighting quality
            lighting_ok = self.check_lighting_quality(frame)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and get landmarks
            face_results = self.face_mesh.process(rgb_frame)
            
            current_time = time.time()
            
            # Validate face detection
            valid_face = False
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0]
                valid_face = self.validate_face_detection(landmarks, frame.shape)
            
            if valid_face and lighting_ok:
                # Valid face detected with good lighting
                if not self.face_detected:
                    # Start timing stable detection
                    self.detection_start_time = current_time
                    self.face_detected = True
                    print("Valid face detected! Hold still...")
                
                # Calculate time remaining
                elapsed_time = current_time - self.detection_start_time
                time_remaining = max(0, self.stable_detection_duration - elapsed_time)
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                
                # Check if 3 seconds have passed
                if elapsed_time >= self.stable_detection_duration:
                    # Capture the image
                    self.captured_image = frame.copy()
                    print("Image captured! Analyzing skin tone...")
                    break
                
                # Draw detection info
                self.draw_detection_info(frame, time_remaining, lighting_ok)
                
            else:
                # No valid face detected or poor lighting
                self.face_detected = False
                self.detection_start_time = None
                
                if not lighting_ok:
                    cv2.putText(frame, "Lighting too dim or bright", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Move to normal lighting", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif face_results.multi_face_landmarks:
                    # Face detected but not valid (might be hand or other object)
                    cv2.putText(frame, "Invalid face detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Show your face clearly", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, "Avoid hands or objects", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # No face detected at all
                    cv2.putText(frame, "No face detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Position your face in the center", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Auto Skin Tone Detector', frame)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Detection cancelled by user")
                break
        
        # Close webcam and windows
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Process captured image if available
        if self.captured_image is not None:
            self.analyze_captured_image()
        else:
            print("No image was captured. Please try again.")
    
    def analyze_captured_image(self):
        """Analyze the captured image for skin tone detection"""
        print("\n" + "=" * 60)
        print("ANALYZING CAPTURED IMAGE")
        print("=" * 60)
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
        
        # Detect faces and get landmarks
        face_results = self.face_mesh.process(rgb_image)
        
        if face_results.multi_face_landmarks:
            # Use the first detected face
            landmarks = face_results.multi_face_landmarks[0]
            
            # Extract skin regions
            skin_regions = self.get_skin_regions(landmarks, self.captured_image.shape)
            
            # Extract skin color
            rgb_values, hsv_values = self.extract_skin_color(self.captured_image, skin_regions)
            
            if rgb_values is not None and hsv_values is not None:
                # Classify skin tone
                skin_tone = self.classify_skin_tone(hsv_values)
                
                # Print results
                print(f"SKIN TONE DETECTION RESULTS:")
                print(f"Category: {skin_tone}")
                print(f"RGB Values: {rgb_values.astype(int)}")
                print(f"HSV Values: {hsv_values}")
                print("=" * 60)
                
                # Draw results on image
                result_image = self.captured_image.copy()
                
                # Draw landmarks and regions
                self.mp_drawing.draw_landmarks(
                    result_image, landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                
                # Draw skin regions
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors
                for i, (region_name, points) in enumerate(skin_regions.items()):
                    # Create bounding rectangle
                    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), colors[i % len(colors)], 2)
                    
                    # Add region label
                    cv2.putText(result_image, region_name, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2)
                
                # Add skin tone text overlay
                cv2.putText(result_image, f"Skin Tone: {skin_tone}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(result_image, f"RGB: {rgb_values.astype(int)}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result_image, f"HSV: {hsv_values}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display the result
                cv2.imshow('Skin Tone Detection Result', result_image)
                print("Press any key to close the result window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else:
                print("Error: Could not extract skin color from the image")
        else:
            print("Error: No face detected in the captured image")
        
        # Cleanup
        self.face_detection.close()
        self.face_mesh.close()
        print("Analysis complete. Program terminated.")

if __name__ == "__main__":
    detector = AutoSkinToneDetector()
    detector.run()