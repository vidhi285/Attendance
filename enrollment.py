import streamlit as st
import cv2
import numpy as np
from typing import Optional, Dict, List
import re
import logging
from datetime import datetime
import time

from database import DatabaseManager
from face_recognition import FaceRecognitionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrollmentSystem:
    """Handles student and teacher enrollment with improved face capture"""
    
    def __init__(self, db_manager: DatabaseManager, face_system: FaceRecognitionSystem):
        """Initialize enrollment system"""
        self.db = db_manager
        self.face_system = face_system
        
        # Validation patterns
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.enrollment_pattern = re.compile(r'^[A-Za-z0-9]{6,20}$')
        self.subject_code_pattern = re.compile(r'^[A-Za-z0-9]{3,10}$')
        
        logger.info("Enrollment system initialized")
    
    def _ensure_capture_state(self):
        """Initialize session state for photo capture"""
        defaults = {
            'capture_count': 0,
            'captured_encodings': [],
            'captured_images': [],
            'last_face_encoding': None,
            'enrollment_in_progress': False,
            'current_enrollment_data': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    # Validation Methods
    def validate_email(self, email: str) -> bool:
        return bool(self.email_pattern.match(email))
    
    def validate_enrollment_id(self, enrollment_id: str) -> bool:
        return bool(self.enrollment_pattern.match(enrollment_id))
    
    def validate_subject_code(self, subject_code: str) -> bool:
        return bool(self.subject_code_pattern.match(subject_code))
    
    def validate_name(self, name: str) -> bool:
        return len(name.strip()) >= 2 and len(name.strip()) <= 100
    
    # Duplicate Checking
    def check_duplicate_student(self, enrollment_id: str, nuv_mail: str) -> Dict[str, bool]:
        try:
            result_enroll = self.db.execute_query(
                "SELECT student_id FROM students WHERE enrollment_id = %s",
                (enrollment_id,), fetch=True
            )
            result_email = self.db.execute_query(
                "SELECT student_id FROM students WHERE nuv_mail = %s",
                (nuv_mail,), fetch=True
            )
            return {
                'enrollment_exists': len(result_enroll) > 0,
                'email_exists': len(result_email) > 0
            }
        except Exception as e:
            logger.error(f"Error checking duplicate student: {e}")
            return {'enrollment_exists': False, 'email_exists': False}
    
    def check_duplicate_teacher(self, nuv_mail: str) -> bool:
        try:
            result = self.db.execute_query(
                "SELECT teacher_id FROM teachers WHERE nuv_mail = %s",
                (nuv_mail,), fetch=True
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking duplicate teacher: {e}")
            return False
    
    def capture_enrollment_photos(self, person_name: str, person_type: str = "student") -> Optional[np.ndarray]:
        """
        Streamlit-based photo capture interface with real-time feedback
        """
        self._ensure_capture_state()

        st.info(f"📸 Photo capture for {person_type}: {person_name}")
        st.warning("⚠️ Position yourself clearly in the camera frame. Ensure good lighting.")

        # Initialize camera
        if not self.face_system.video_capture or not self.face_system.video_capture.isOpened():
            if not self.face_system.initialize_camera():
                st.error("❌ Could not initialize camera. Please check camera permissions.")
                return None

        # Create UI columns
        col1, col2, col3 = st.columns([1, 1, 1])

        # Camera feed container
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        target_photos = 3
        min_photos = 1

        # Capture button
        with col1:
            capture_clicked = st.button("📷 Capture Photo", key=f"capture_{person_name}")

        # Reset button  
        with col2:
            reset_clicked = st.button("🔄 Reset", key=f"reset_{person_name}")

        # Complete button
        with col3:
            complete_clicked = st.button("✅ Complete", key=f"complete_{person_name}")

        # Handle button clicks
        if reset_clicked:
            st.session_state.capture_count = 0
            st.session_state.captured_encodings = []
            st.session_state.captured_images = []
            st.session_state.last_face_encoding = None
            st.success("🔄 Capture reset successfully")
            st.rerun()

        if capture_clicked:
            self._handle_photo_capture(person_name, video_placeholder)

        if complete_clicked:
            if len(st.session_state.captured_encodings) >= min_photos:
                final_encoding = self._finalize_encoding()
                if final_encoding is not None:
                    st.session_state.last_face_encoding = final_encoding
                    st.success(f"✅ Photo capture completed! Captured {len(st.session_state.captured_encodings)} photos.")
                    return final_encoding
            else:
                st.error(f"❌ Please capture at least {min_photos} photo(s) before completing.")

        # Show live camera feed
        self._show_camera_feed(video_placeholder)

        # Show progress
        current_count = len(st.session_state.captured_encodings)
        progress_placeholder.info(f"📊 Photos captured: {current_count}/{target_photos} (minimum {min_photos} required)")

        # Return encoding if available
        return st.session_state.get('last_face_encoding')

    def _handle_photo_capture(self, person_name: str, video_placeholder):
        """Handle single photo capture with validation"""
        try:
            frame = self.face_system.capture_frame()
            if frame is None:
                st.error("❌ Failed to capture frame from camera")
                return

            # Extract face encodings
            face_encodings = self.face_system.get_face_encoding_from_frame(frame)
            
            if len(face_encodings) == 0:
                st.error("❌ No face detected. Please position yourself clearly in the camera.")
            elif len(face_encodings) > 1:
                st.warning("⚠️ Multiple faces detected. Please ensure only one person is in frame.")
            else:
                # Valid single face detected
                encoding = face_encodings[0]
                
                # Validate encoding
                if self.face_system.validate_face_encoding(encoding):
                    # Store the encoding
                    st.session_state.captured_encodings.append(encoding)
                    st.session_state.capture_count = len(st.session_state.captured_encodings)
                    
                    # Save the frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{person_name.replace(' ', '_')}_{st.session_state.capture_count}_{timestamp}.jpg"
                    saved_path = self.face_system.save_frame(frame, filename)
                    
                    if saved_path:
                        st.session_state.captured_images.append(saved_path)
                    
                    st.success(f"✅ Photo {st.session_state.capture_count} captured successfully!")
                    time.sleep(1)  # Brief pause for user feedback
                    st.rerun()
                else:
                    st.error("❌ Invalid face encoding captured. Please try again with better lighting.")
                    
        except Exception as e:
            st.error(f"❌ Error during photo capture: {str(e)}")
            logger.error(f"Photo capture error: {e}")

    def _show_camera_feed(self, video_placeholder):
        """Display live camera feed with face detection overlay"""
        try:
            frame = self.face_system.capture_frame()
            if frame is None:
                video_placeholder.error("❌ Cannot display camera feed")
                return

            # Add face detection rectangle
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb_frame.shape[:2]

            # Draw center guide rectangle
            center_x, center_y = width // 2, height // 2
            guide_width, guide_height = 300, 350
            
            top_left = (center_x - guide_width // 2, center_y - guide_height // 2)
            bottom_right = (center_x + guide_width // 2, center_y + guide_height // 2)
            
            # Draw guide rectangle
            cv2.rectangle(rgb_frame, top_left, bottom_right, (255, 255, 0), 3)
            
            # Add instruction text
            cv2.putText(rgb_frame, "Position face in yellow rectangle", 
                       (center_x - 150, top_left[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Detect and highlight faces
            face_encodings = self.face_system.get_face_encoding_from_frame(frame)
            face_locations = getattr(self.face_system, 'face_locations', [])
            
            # Draw face detection boxes
            for (top, right, bottom, left) in face_locations:
                # Scale coordinates back to original size
                scale_factor = 1 / self.face_system.frame_resizing
                top = int(top * scale_factor)
                right = int(right * scale_factor)
                bottom = int(bottom * scale_factor)
                left = int(left * scale_factor)
                
                # Draw face box
                cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(rgb_frame, "Face Detected", (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
        except Exception as e:
            logger.error(f"Error displaying camera feed: {e}")
            video_placeholder.error(f"❌ Camera feed error: {str(e)}")

    def _finalize_encoding(self) -> Optional[np.ndarray]:
        """Create final face encoding from captured photos"""
        if not st.session_state.captured_encodings:
            return None
        
        try:
            if len(st.session_state.captured_encodings) == 1:
                final_encoding = st.session_state.captured_encodings[0]
            else:
                # Average multiple encodings for better accuracy
                final_encoding = np.mean(st.session_state.captured_encodings, axis=0)
            
            # Validate final encoding
            if self.face_system.validate_face_encoding(final_encoding):
                return final_encoding.astype(np.float64)
            else:
                logger.error("Final encoding validation failed")
                return None
                
        except Exception as e:
            logger.error(f"Error finalizing encoding: {e}")
            return None

    def enroll_student_interactive(self):
        """Complete student enrollment interface - called by app.py"""
        self._ensure_capture_state()
        
        st.subheader("👨‍🎓 Student Enrollment")
        
        # Show enrollment form
        with st.form("student_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                enrollment_id = st.text_input("🆔 Enrollment ID *", 
                                            placeholder="e.g., 2021CSE001",
                                            help="6-20 characters, alphanumeric").strip()
                name = st.text_input("👤 Full Name *", 
                                   placeholder="e.g., John Doe").strip()
            
            with col2:
                nuv_mail = st.text_input("📧 NUV Email *", 
                                       placeholder="e.g., john.doe@nuv.ac.in").strip()
                
                # Section selection
                existing_sections = self.db.get_section_list()
                section_options = existing_sections + ["➕ Create New Section"] if existing_sections else ["➕ Create New Section"]
                section_choice = st.selectbox("🏫 Section", options=section_options)
            
            # Handle new section
            if section_choice == "➕ Create New Section":
                section = st.text_input("📚 New Section Name *", 
                                      placeholder="e.g., CSE TY A").strip()
            else:
                section = section_choice
            
            form_submitted = st.form_submit_button("➡️ Validate & Proceed to Photo Capture", 
                                                 type="primary")
        
        # Process form submission
        if form_submitted:
            # Validate all inputs
            validation_errors = []
            
            if not self.validate_enrollment_id(enrollment_id):
                validation_errors.append("❌ Enrollment ID must be 6-20 alphanumeric characters")
            if not self.validate_name(name):
                validation_errors.append("❌ Name must be 2-100 characters")
            if not self.validate_email(nuv_mail):
                validation_errors.append("❌ Invalid email format")
            if not section or len(section.strip()) < 2:
                validation_errors.append("❌ Section name is required")
            
            # Show validation errors
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return False
            
            # Check duplicates
            duplicates = self.check_duplicate_student(enrollment_id, nuv_mail)
            if duplicates['enrollment_exists']:
                st.error("❌ Enrollment ID already exists")
                return False
            if duplicates['email_exists']:
                st.error("❌ Email already registered")
                return False
            
            # Store validated data
            st.session_state.current_enrollment_data = {
                'enrollment_id': enrollment_id,
                'name': name,
                'nuv_mail': nuv_mail,
                'section': section,
                'type': 'student'
            }
            st.session_state.enrollment_in_progress = True
            
            st.success("✅ Information validated! Please capture photos below.")
        
        # Show photo capture if form is validated
        if st.session_state.get('enrollment_in_progress') and st.session_state.get('current_enrollment_data'):
            st.markdown("---")
            st.subheader("📸 Photo Capture")
            
            data = st.session_state.current_enrollment_data
            
            # Capture photos
            final_encoding = self.capture_enrollment_photos(data['name'], data['type'])
            
            # Save to database if encoding is ready
            if final_encoding is not None:
                try:
                    student_id = self.db.enroll_student(
                        enrollment_id=data['enrollment_id'],
                        name=data['name'],
                        nuv_mail=data['nuv_mail'],
                        section=data['section'],
                        face_encoding=final_encoding
                    )
                    
                    st.success(f"🎉 Student enrolled successfully! Student ID: {student_id}")
                    st.balloons()
                    
                    # Clear session state
                    self._clear_enrollment_state()
                    
                    # Add refresh button
                    if st.button("🔄 Enroll Another Student"):
                        st.rerun()
                    
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Database error: {str(e)}")
                    logger.error(f"Student enrollment error: {e}")
                    return False
        
        return False

    def enroll_teacher_interactive(self):
        """Complete teacher enrollment interface - called by app.py"""
        self._ensure_capture_state()
        
        st.subheader("👨‍🏫 Teacher Enrollment")
        
        # Show enrollment form
        with st.form("teacher_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("👤 Full Name *", 
                                   placeholder="e.g., Dr. Jane Smith").strip()
                nuv_mail = st.text_input("📧 NUV Email *", 
                                       placeholder="e.g., jane.smith@nuv.ac.in").strip()
            
            with col2:
                subject_name = st.text_input("📚 Subject Name *", 
                                           placeholder="e.g., Data Structures").strip()
                subject_code = st.text_input("🔢 Subject Code *", 
                                           placeholder="e.g., CSE301").strip()
            
            form_submitted = st.form_submit_button("➡️ Validate & Proceed to Photo Capture", 
                                                 type="primary")
        
        # Process form submission
        if form_submitted:
            # Validate inputs
            validation_errors = []
            
            if not self.validate_name(name):
                validation_errors.append("❌ Name must be 2-100 characters")
            if not self.validate_email(nuv_mail):
                validation_errors.append("❌ Invalid email format")
            if not self.validate_name(subject_name):
                validation_errors.append("❌ Subject name is required")
            if not self.validate_subject_code(subject_code):
                validation_errors.append("❌ Subject code must be 3-10 alphanumeric characters")
            
            # Show validation errors
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return False
            
            # Check duplicates
            if self.check_duplicate_teacher(nuv_mail):
                st.error("❌ Teacher email already registered")
                return False
            
            # Store validated data
            st.session_state.current_enrollment_data = {
                'name': name,
                'nuv_mail': nuv_mail,
                'subject_name': subject_name,
                'subject_code': subject_code,
                'type': 'teacher'
            }
            st.session_state.enrollment_in_progress = True
            
            st.success("✅ Information validated! Please capture photos below.")
        
        # Show photo capture if form is validated
        if st.session_state.get('enrollment_in_progress') and st.session_state.get('current_enrollment_data'):
            st.markdown("---")
            st.subheader("📸 Photo Capture")
            
            data = st.session_state.current_enrollment_data
            
            # Capture photos
            final_encoding = self.capture_enrollment_photos(data['name'], data['type'])
            
            # Save to database if encoding is ready
            if final_encoding is not None:
                try:
                    teacher_id = self.db.enroll_teacher(
                        name=data['name'],
                        nuv_mail=data['nuv_mail'],
                        subject_name=data['subject_name'],
                        subject_code=data['subject_code'],
                        face_encoding=final_encoding
                    )
                    
                    st.success(f"🎉 Teacher enrolled successfully! Teacher ID: {teacher_id}")
                    st.balloons()
                    
                    # Clear session state
                    self._clear_enrollment_state()
                    
                    # Add refresh button
                    if st.button("🔄 Enroll Another Teacher"):
                        st.rerun()
                    
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Database error: {str(e)}")
                    logger.error(f"Teacher enrollment error: {e}")
                    return False
        
        return False

    def _clear_enrollment_state(self):
        """Clear enrollment session state"""
        keys_to_clear = [
            'capture_count', 'captured_encodings', 'captured_images', 
            'last_face_encoding', 'enrollment_in_progress', 'current_enrollment_data'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], dict):
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = None if key in ['last_face_encoding'] else False

    def get_enrollment_statistics(self) -> Dict[str, int]:
        """Get enrollment statistics from database"""
        try:
            student_count = self.db.execute_query("SELECT COUNT(*) as count FROM students", fetch=True)[0]['count']
            teacher_count = self.db.execute_query("SELECT COUNT(*) as count FROM teachers", fetch=True)[0]['count']
            section_count = len(self.db.get_section_list())
            subject_count = len(self.db.get_subject_codes())
            
            return {
                'total_students': student_count,
                'total_teachers': teacher_count,
                'total_sections': section_count,
                'total_subjects': subject_count
            }
        except Exception as e:
            logger.error(f"Error getting enrollment statistics: {e}")
            return {'total_students': 0, 'total_teachers': 0, 'total_sections': 0, 'total_subjects': 0}

    def display_enrollment_stats(self):
        """Display enrollment statistics in Streamlit"""
        st.subheader("📊 Enrollment Statistics")
        
        stats = self.get_enrollment_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👨‍🎓 Students", stats['total_students'])
        with col2:
            st.metric("👨‍🏫 Teachers", stats['total_teachers'])
        with col3:
            st.metric("🏫 Sections", stats['total_sections'])
        with col4:
            st.metric("📚 Subjects", stats['total_subjects'])