import streamlit as st
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import time

from database import DatabaseManager
from face_recognition import FaceRecognitionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttendanceManager:
    """Handles real-time attendance tracking during lecture sessions"""
    
    def __init__(self, db_manager: DatabaseManager, face_system: FaceRecognitionSystem):
        """Initialize attendance system"""
        self.db = db_manager
        self.face_system = face_system
        
        # Attendance tracking state
        self.current_lecture = None
        self.section_students = []
        self.recognized_students = set()  # Track already marked students
        
        # Recognition parameters
        self.recognition_confidence_threshold = 0.6
        self.recognition_cooldown = 3  # seconds between recognitions for same person
        self.last_recognition_times = {}
        
        logger.info("Attendance system initialized")
    
    def load_lecture_context(self, lecture_info: Dict) -> bool:
        """
        Load lecture context and students for attendance tracking
        
        Args:
            lecture_info: Active lecture information
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            self.current_lecture = lecture_info
            
            # Get students for the lecture section
            self.section_students = self.db.get_students_by_section(lecture_info['section'])
            
            # Load known faces (students + teachers for complete recognition)
            teachers = self.db.get_all_teachers()
            self.face_system.load_known_faces(self.section_students, teachers)
            
            logger.info(f"Loaded {len(self.section_students)} students for section {lecture_info['section']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading lecture context: {e}")
            return False
    
    def determine_attendance_status(self, recognition_time: datetime) -> str:
        """
        Determine attendance status based on recognition time vs cutoff time
        
        Args:
            recognition_time: When the student was recognized
        
        Returns:
            'Present' if before cutoff, 'Absent' if after cutoff
        """
        if not self.current_lecture:
            return 'Unknown'
        
        cutoff_time = self.current_lecture['cutoff_time']
        # Convert cutoff_time to datetime if it's a string
        if isinstance(cutoff_time, str):
            try:
                cutoff_time = datetime.fromisoformat(cutoff_time)
            except Exception:
                logger.error("Invalid cutoff_time format in lecture context")
                return 'Unknown'
        
        if recognition_time <= cutoff_time:
            return 'Present'
        else:
            return 'Absent'  # Late arrival
    
    def should_process_recognition(self, person_id: str) -> bool:
        """
        Check if enough time has passed since last recognition for this person
        
        Args:
            person_id: Unique identifier for the person
        
        Returns:
            True if should process, False if in cooldown period
        """
        current_time = time.time()
        
        if person_id in self.last_recognition_times:
            time_since_last = current_time - self.last_recognition_times[person_id]
            if time_since_last < self.recognition_cooldown:
                return False
        
        self.last_recognition_times[person_id] = current_time
        return True
    
    def process_face_recognition(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Process face recognition and handle attendance marking
        Compatible with app.py's process_recognition_frame method
        
        Args:
            frame: Current camera frame
        
        Returns:
            List of attendance events, annotated frame
        """
        if not self.current_lecture:
            return [], frame
        
        try:
            # Use the face recognition system's method (compatible with app.py)
            recognized_faces, annotated_frame = self.face_system.recognize_faces_in_frame(frame)
            
            attendance_events = []
            recognition_time = datetime.now()
            
            for face in recognized_faces:
                if face['is_known']:
                    # Known person (student or teacher)
                    person_key = f"{face['type']}_{face['id']}"
                    
                    # Check cooldown period
                    if not self.should_process_recognition(person_key):
                        continue
                    
                    if face['type'] == 'student':
                        # Process student attendance
                        event = self.process_student_attendance(face, recognition_time)
                        if event:
                            attendance_events.append(event)
                    
                    elif face['type'] == 'teacher':
                        # Log teacher presence (optional)
                        logger.info(f"Teacher {face['name']} detected in frame")
                        attendance_events.append({
                            'type': 'teacher_detected',
                            'name': face['name'],
                            'status': 'Teacher Present',
                            'timestamp': recognition_time
                        })
                
                else:
                    # Unknown person
                    person_key = f"unknown_{int(time.time() * 1000)}"  # Use timestamp for uniqueness
                    
                    if not self.should_process_recognition(person_key):
                        continue
                    
                    event = self.process_unknown_person(recognition_time)
                    if event:
                        attendance_events.append(event)
            
            return attendance_events, annotated_frame
            
        except Exception as e:
            logger.error(f"Error processing face recognition: {e}")
            return [], frame
    
    def process_student_attendance(self, face_data: Dict, recognition_time: datetime) -> Optional[Dict]:
        """
        Process student attendance marking
        
        Args:
            face_data: Recognized face information
            recognition_time: When the recognition occurred
        
        Returns:
            Attendance event dict or None if not processed
        """
        try:
            student_id = face_data['id']
            student_name = face_data['name']
            
            # Check if attendance already marked for this lecture
            lecture_id = self.current_lecture['lecture_id']
            
            # Determine attendance status
            status = self.determine_attendance_status(recognition_time)
            
            # Mark attendance in database (using same method as app.py)
            success = self.db.mark_attendance(
                lecture_id=lecture_id,
                student_id=student_id,
                enrollment_id=None,
                status=status,
                person_name=student_name
            )
            
            if success:
                # Add to recognized students set
                self.recognized_students.add(student_id)
                
                # Create attendance event (format compatible with app.py)
                event = {
                    'type': 'student_attendance',
                    'name': student_name,
                    'status': status,
                    'student_id': student_id,
                    'timestamp': recognition_time,
                    'time': recognition_time.strftime("%H:%M:%S"),
                    'confidence': face_data.get('distance', None)
                }
                
                logger.info(f"Student attendance marked: {student_name} - {status}")
                return event
            
            else:
                # Attendance might already exist, but don't treat as error
                logger.info(f"Attendance already marked or database error for {student_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing student attendance: {e}")
            return None
    
    def process_unknown_person(self, recognition_time: datetime) -> Optional[Dict]:
        """
        Process unknown person detection
        
        Args:
            recognition_time: When the detection occurred
        
        Returns:
            Attendance event dict or None if not processed
        """
        try:
            lecture_id = self.current_lecture['lecture_id']
            
            # Create unique person name with timestamp
            person_name = f"Unknown Person {recognition_time.strftime('%H%M%S')}"
            
            # Mark as unknown person (using same method as app.py)
            success = self.db.mark_attendance(
                lecture_id=lecture_id,
                student_id=None,
                enrollment_id='UNKNOWN',
                status='Unknown',
                person_name=person_name
            )
            
            if success:
                event = {
                    'type': 'unknown_person',
                    'name': 'Unknown Person',
                    'status': 'Unknown',
                    'timestamp': recognition_time,
                    'time': recognition_time.strftime("%H:%M:%S")
                }
                
                logger.info("Unknown person detected and logged")
                return event
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing unknown person: {e}")
            return None
    
    def run_continuous_attendance(self):
        """
        Run continuous attendance tracking with live camera feed
        This should be called in a Streamlit app with proper session management
        """
        if not self.current_lecture:
            st.error("No active lecture loaded")
            return
        
        st.subheader("📹 Live Attendance Tracking")
        
        # Display lecture info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Section", self.current_lecture['section'])
        with col2:
            st.metric("Subject", self.current_lecture['subject_code'])
        with col3:
            current_time = datetime.now()
            cutoff_time = self.current_lecture['cutoff_time']
            # Convert cutoff_time to datetime if it's a string
            if isinstance(cutoff_time, str):
                try:
                    cutoff_time = datetime.fromisoformat(cutoff_time)
                except Exception:
                    st.metric("Status", "⚠️ Invalid Cutoff Time")
                    cutoff_time = current_time
            if current_time <= cutoff_time:
                remaining = cutoff_time - current_time
                remaining_str = str(remaining).split('.')[0]
                st.metric("Time Remaining", remaining_str)
            else:
                st.metric("Status", "⚠️ Late Period", delta="After Cutoff")
        
        # Camera controls
        col1, col2 = st.columns(2)
        with col1:
            camera_active = st.checkbox("📷 Camera Active", value=True)
        with col2:
            show_names = st.checkbox("🏷️ Show Names", value=True)
        
        if camera_active:
            # Initialize camera if needed
            if not self.face_system.video_capture or not self.face_system.video_capture.isOpened():
                if not self.face_system.initialize_camera():
                    st.error("Could not initialize camera")
                    return
            
            # Placeholders for live content
            video_placeholder = st.empty()
            events_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            # Main attendance loop
            if 'attendance_events' not in st.session_state:
                st.session_state.attendance_events = []
            
            # Capture and process frame
            frame = self.face_system.capture_frame()
            if frame is not None:
                # Process attendance
                new_events, annotated_frame = self.process_face_recognition(frame)
                
                # Update events list
                st.session_state.attendance_events.extend(new_events)
                
                # Display video feed
                if not show_names:
                    # Remove names from display for privacy
                    display_frame = frame.copy()
                else:
                    display_frame = annotated_frame
                
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Display recent events
                if new_events:
                    with events_placeholder.container():
                        st.write("🔔 Recent Attendance Events:")
                        for event in new_events[-5:]:  # Show last 5 events
                            if event['type'] == 'student_attendance':
                                if event['status'] == 'Present':
                                    st.success(f"✅ {event['name']} - Present")
                                else:
                                    st.warning(f"⚠️ {event['name']} - Late")
                            elif event['type'] == 'teacher_detected':
                                st.info(f"👨‍🏫 {event['name']} - Teacher Present")
                            else:
                                st.error("❓ Unknown person detected")
                
                # Display current stats
                self.display_live_statistics(stats_placeholder)
        
        else:
            st.info("Camera is inactive. Enable camera to start attendance tracking.")
    
    def display_live_statistics(self, placeholder):
        """Display live attendance statistics"""
        if not self.current_lecture:
            return
        
        try:
            # Get current attendance summary
            summary = self.db.get_attendance_summary(self.current_lecture['lecture_id'])
            
            with placeholder.container():
                st.write("📊 Live Attendance Stats:")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Present", summary.get('Present', 0), delta=None)
                
                with col2:
                    st.metric("Late", summary.get('Absent', 0), delta=None)
                
                with col3:
                    st.metric("Unknown", summary.get('Unknown', 0), delta=None)
                
                with col4:
                    total = summary.get('Total', 0)
                    expected = len(self.section_students)
                    if expected > 0:
                        percentage = (summary.get('Present', 0) / expected) * 100
                        st.metric("Attendance %", f"{percentage:.1f}%")
                    else:
                        st.metric("Expected", expected)
        
        except Exception as e:
            logger.error(f"Error displaying statistics: {e}")
    
    def get_attendance_report(self, lecture_id: int) -> List[Dict]:
        """Get detailed attendance report for a lecture"""
        try:
            return self.db.get_lecture_attendance(lecture_id)
        except Exception as e:
            logger.error(f"Error getting attendance report: {e}")
            return []
    
    def get_absentees_list(self, lecture_id: int) -> List[Dict]:
        """Get list of students who were absent from the lecture"""
        try:
            # Get all students in the section
            lecture_info = self.db.get_lecture_by_id(lecture_id)
            if not lecture_info:
                return []
            
            section_students = self.db.get_students_by_section(lecture_info['section'])
            
            # Get students who attended
            attendance_records = self.db.get_lecture_attendance(lecture_id)
            attended_student_ids = {record.get('student_id') for record in attendance_records if record.get('student_id')}
            
            # Find absentees
            absentees = []
            for student in section_students:
                student_id = student.get('student_id')
                if student_id and student_id not in attended_student_ids:
                    absentees.append({
                        'student_id': student_id,
                        'enrollment_id': student.get('enrollment_id'),
                        'name': student.get('name'),
                        'nuv_mail': student.get('nuv_mail'),
                        'status': 'Absent (Not Detected)'
                    })
            
            return absentees
            
        except Exception as e:
            logger.error(f"Error getting absentees list: {e}")
            return []
    
    def reset_recognition_cache(self):
        """Reset recognition cache and cooldown timers"""
        self.recognized_students.clear()
        self.last_recognition_times.clear()
        logger.info("Recognition cache reset")
    
    def export_attendance_data(self, lecture_id: int) -> Dict[str, any]:
        """Export attendance data for reports"""
        try:
            lecture_info = self.db.get_lecture_by_id(lecture_id)
            attendance_records = self.db.get_lecture_attendance(lecture_id)
            absentees = self.get_absentees_list(lecture_id)
            summary = self.db.get_attendance_summary(lecture_id)
            
            return {
                'lecture_info': lecture_info,
                'attendance_records': attendance_records,
                'absentees': absentees,
                'summary': summary,
                'export_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error exporting attendance data: {e}")
            return {}