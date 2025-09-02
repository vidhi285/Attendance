import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from datetime import datetime
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """Handles face detection and recognition operations with improved performance."""
    
    def __init__(self):
        """
        Initialize face recognition system.
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_types = []  # 'student' or 'teacher'
        
        # Face recognition parameters
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # Video capture settings
        self.video_capture = None
        self.frame_resizing = 0.25  # Process smaller frame to improve speed
        
        # Recognition settings
        self.tolerance = 0.6  # Default tolerance
        self.model = 'hog'  # Use HOG model for better performance
        
        # Timer for performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Create photos directory if it doesn't exist
        self.photos_dir = "captured_photos"
        os.makedirs(self.photos_dir, exist_ok=True)
        
        logger.info("Face Recognition System initialized with optimized settings.")
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """
        Initialize video capture with specified camera index.
        Returns True on success, False otherwise.
        """
        try:
            # Release any existing capture
            if self.video_capture:
                self.video_capture.release()
            
            self.video_capture = cv2.VideoCapture(camera_index)
            
            if not self.video_capture.isOpened():
                logger.error("Failed to open camera.")
                return False
            
            # Set camera properties for better quality and performance
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def release_camera(self):
        """Release video capture resource."""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        cv2.destroyAllWindows()
        logger.info("Camera released.")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        Returns the frame as numpy array or None if failed.
        """
        if not self.video_capture or not self.video_capture.isOpened():
            logger.error("Camera not initialized or not available")
            return None
        
        ret, frame = self.video_capture.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None
        
        return frame
    
    def save_frame(self, frame: np.ndarray, filename: str) -> Optional[str]:
        """
        Save frame to disk with given filename.
        Returns the saved path or None if failed.
        """
        try:
            filepath = os.path.join(self.photos_dir, filename)
            success = cv2.imwrite(filepath, frame)
            if success:
                logger.info(f"Frame saved to {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save frame to {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return None
    
    def get_face_encoding_from_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extract face encodings from a frame.
        Returns list of face encodings found in the frame.
        """
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # FIXED: Use correct face_recognition import
            # Find faces and encodings
            self.face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            
            return [encoding.astype(np.float64) for encoding in face_encodings]
            
        except Exception as e:
            logger.error(f"Error extracting face encodings: {e}")
            return []
    
    def validate_face_encoding(self, encoding: np.ndarray) -> bool:
        """
        Validate if a face encoding is valid.
        Returns True if valid, False otherwise.
        """
        if encoding is None:
            return False
        
        if not isinstance(encoding, np.ndarray):
            return False
        
        # Check if encoding has correct dimensions (128 features)
        if encoding.size != 128:
            return False
        
        # Check if encoding is not all zeros
        if np.allclose(encoding, 0):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(encoding)) or np.any(np.isinf(encoding)):
            return False
        
        return True
    
    def load_known_faces(self, students: List[Dict], teachers: List[Dict]):
        """
        Load known faces from provided student and teacher data.
        Validates encodings before adding to the system.
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_types = []

        data_sets = [('student', students, 'student_id'), ('teacher', teachers, 'teacher_id')]
        
        for face_type, people, id_key in data_sets:
            for person in people:
                try:
                    encoding = person.get('face_encoding')
                    if self.validate_face_encoding(encoding):
                        self.known_face_encodings.append(encoding.astype(np.float64))
                        self.known_face_names.append(person['name'])
                        self.known_face_ids.append(person[id_key])
                        self.known_face_types.append(face_type)
                    else:
                        logger.warning(f"Invalid or zero encoding for {face_type}: {person.get('name', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Error loading {face_type} face: {person.get('name', 'Unknown')} - {e}")
        
        logger.info(f"Loaded {len(self.known_face_encodings)} known faces.")
    
    def recognize_faces_in_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Recognize faces in a frame and return recognized faces with annotated frame.
        This method is called by app.py for live recognition.
        """
        if len(self.known_face_encodings) == 0:
            logger.warning("No known faces loaded. Recognition is not possible.")
            return [], frame
            
        # Get face locations and encodings from frame
        face_locations, face_encodings = self.get_face_locations_and_encodings(frame)
        recognized_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare face encoding with known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            best_match_index = np.argmin(face_distances)
            
            name = "Unknown"
            person_id = None
            person_type = "unknown"
            is_known = False

            if face_distances[best_match_index] <= self.tolerance:
                name = self.known_face_names[best_match_index]
                person_id = self.known_face_ids[best_match_index]
                person_type = self.known_face_types[best_match_index]
                is_known = True

            recognized_faces.append({
                'name': name,
                'id': person_id,
                'type': person_type,
                'is_known': is_known,
                'distance': float(face_distances[best_match_index]),
                'face_location': face_location
            })
            
        annotated_frame = self._annotate_frame(frame.copy(), recognized_faces)
        
        return recognized_faces, annotated_frame

    def get_face_locations_and_encodings(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Finds all faces and their encodings in a single frame using an optimized pipeline.
        """
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # FIXED: Use correct face_recognition import
            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Store face locations for other methods to use
            self.face_locations = face_locations
            
            return face_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Error in face detection/encoding: {e}")
            return [], []

    def _annotate_frame(self, frame: np.ndarray, recognized_faces: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        for face in recognized_faces:
            location = face['face_location']
            name = face['name']
            
            # Scale coordinates back up to original frame size
            top, right, bottom, left = [int(v / self.frame_resizing) for v in location]
            
            color = (0, 255, 0) if face['is_known'] else (0, 0, 255) # Green for known, Red for unknown
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
        return frame
    
    def capture_enrollment_photo(self, person_name: str) -> Optional[np.ndarray]:
        """
        Captures a single high-quality photo for enrollment and returns its encoding.
        This function is interactive, showing a live camera feed.
        """
        if not self.video_capture or not self.video_capture.isOpened():
            logger.error("Camera not available for photo capture.")
            return None
            
        logger.info(f"Starting photo capture for {person_name}. Press SPACE to capture.")
        
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                return None
            
            # Find face locations on a smaller frame for performance
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # FIXED: Use correct face_recognition import
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            annotated_frame = frame.copy()
            if face_locations:
                for top, right, bottom, left in face_locations:
                    # Scale back up for drawing
                    top, right, bottom, left = [int(v * 2) for v in (top, right, bottom, left)]
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Face Detected", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "No face detected. Please position yourself clearly.", (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(annotated_frame, f"Capturing for {person_name}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Press SPACE to capture, ESC to cancel.", (10, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Enrollment Capture', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # On capture, use the full-size frame for higher-quality encoding
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # FIXED: Use correct face_recognition import
                encodings = face_recognition.face_encodings(rgb_frame, num_jitters=1)
                cv2.destroyAllWindows()
                
                if len(encodings) == 1:
                    logger.info(f"Photo captured and encoding generated for {person_name}.")
                    return encodings[0].astype(np.float64)
                elif len(encodings) > 1:
                    logger.warning("Multiple faces detected. Please ensure only one face is in the frame.")
                    return None
                else:
                    logger.warning("No face detected in the captured frame.")
                    return None
            
            elif key == 27: # ESC key
                logger.info("Photo capture cancelled.")
                cv2.destroyAllWindows()
                return None

    def get_system_info(self) -> Dict:
        """Get current system information."""
        return {
            'known_faces_count': len(self.known_face_encodings),
            'camera_active': self.video_capture is not None and self.video_capture.isOpened(),
            'tolerance': self.tolerance,
            'frame_resizing': self.frame_resizing
        }