import streamlit as st
import cv2
import numpy as np
from typing import Optional, Dict, List
import logging
from datetime import datetime, timedelta

from database import DatabaseManager
from face_recognition import FaceRecognitionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeacherLectureSystem:
    """Handles teacher recognition and lecture session management"""
    
    def __init__(self, db_manager: DatabaseManager, face_system: FaceRecognitionSystem):
        """Initialize teacher lecture system"""
        self.db = db_manager
        self.face_system = face_system
        
        # Session state keys
        self.RECOGNIZED_TEACHER_KEY = 'recognized_teacher'
        self.ACTIVE_LECTURE_KEY = 'active_lecture'
        self.LECTURE_STUDENTS_KEY = 'lecture_students'
        
        logger.info("Teacher lecture system initialized")
    
    def detect_teacher_in_frame(self, frame: np.ndarray, tolerance: float = 0.6) -> Optional[Dict]:
        """
        Detect and recognize teacher in the current frame
        
        Returns:
            Teacher info dict or None if no teacher recognized
        """
        try:
            # Get all teachers for recognition
            teachers = self.db.get_all_teachers()
            
            if not teachers:
                return None
            
            # Load known teacher faces
            self.face_system.load_known_faces([], teachers)
            
            # Recognize faces in frame
            recognized_faces, _ = self.face_system.recognize_faces_in_frame(frame, tolerance)
            
            # Look for recognized teachers
            for face in recognized_faces:
                if face['is_known'] and face['type'] == 'teacher':
                    # Get full teacher details
                    teacher = self.db.get_teacher_by_id(face['id'])
                    if teacher:
                        teacher['confidence'] = face['confidence']
                        teacher['timestamp'] = face['timestamp']
                        return teacher
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting teacher: {e}")
            return None
    
    def start_lecture_session(self) -> bool:
        """Interactive lecture session start with teacher recognition"""
        st.header("🎓 Start Lecture Session")
        
        # Step 1: Teacher Recognition
        if self.RECOGNIZED_TEACHER_KEY not in st.session_state:
            st.subheader("👨‍🏫 Teacher Recognition")
            st.info("Please look at the camera for teacher verification")
            
            # Initialize camera if needed
            if not self.face_system.video_capture or not self.face_system.video_capture.isOpened():
                if not self.face_system.initialize_camera():
                    st.error("❌ Could not initialize camera")
                    return False
            
            # Live teacher recognition
            video_placeholder = st.empty()
            recognition_placeholder = st.empty()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Verify Teacher"):
                    frame = self.face_system.capture_frame()
                    if frame is not None:
                        teacher = self.detect_teacher_in_frame(frame)
                        
                        if teacher:
                            st.session_state[self.RECOGNIZED_TEACHER_KEY] = teacher
                            st.success(f"✅ Teacher recognized: {teacher['name']}")
                            st.rerun()
                        else:
                            st.error("❌ Teacher not recognized. Please ensure you are enrolled in the system.")
            
            with col2:
                if st.button("🔄 Refresh Camera"):
                    self.face_system.release_camera()
                    self.face_system.initialize_camera()
            
            # Show live feed
            frame = self.face_system.capture_frame()
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            return False
        
        # Step 2: Lecture Details Input
        teacher = st.session_state[self.RECOGNIZED_TEACHER_KEY]
        
        st.success(f"✅ Verified Teacher: {teacher['name']}")
        st.info(f"Subject: {teacher['subject_name']} ({teacher['subject_code']})")
        
        with st.form("lecture_setup_form"):
            st.subheader("📚 Lecture Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Section selection
                available_sections = self.db.get_section_list()
                section = st.selectbox(
                    "🏫 Section",
                    options=available_sections,
                    help="Select the section for this lecture"
                )
                
                cutoff_minutes = st.number_input(
                    "⏰ Cutoff Time (minutes)",
                    min_value=1,
                    max_value=60,
                    value=15,
                    help="Students arriving after this time will be marked as late"
                )
            
            with col2:
                # Use teacher's subject code by default, but allow override
                subject_code = st.text_input(
                    "📝 Subject Code",
                    value=teacher['subject_code'],
                    help="Subject code for this lecture"
                )
                
                # Optional lecture notes
                lecture_notes = st.text_area(
                    "📋 Lecture Notes (Optional)",
                    placeholder="Brief description of today's lecture topic...",
                    help="Optional notes about the lecture content"
                )
            
            start_button = st.form_submit_button("🚀 Start Lecture Session")
        
        if start_button:
            try:
                # Start lecture session
                lecture_id = self.db.start_lecture(
                    teacher_id=teacher['teacher_id'],
                    section=section,
                    subject_code=subject_code,
                    cutoff_minutes=cutoff_minutes
                )
                
                # Store lecture info in session state
                lecture_info = {
                    'lecture_id': lecture_id,
                    'teacher_id': teacher['teacher_id'],
                    'teacher_name': teacher['name'],
                    'section': section,
                    'subject_code': subject_code,
                    'subject_name': teacher['subject_name'],
                    'cutoff_minutes': cutoff_minutes,
                    'start_time': datetime.now(),
                    'cutoff_time': datetime.now() + timedelta(minutes=cutoff_minutes),
                    'lecture_notes': lecture_notes
                }
                
                st.session_state[self.ACTIVE_LECTURE_KEY] = lecture_info
                st.session_state[self.LECTURE_STUDENTS_KEY] = []
                
                st.success("🎉 Lecture session started successfully!")
                st.info(f"Lecture ID: {lecture_id}")
                st.info(f"Cutoff time: {lecture_info['cutoff_time'].strftime('%H:%M:%S')}")
                
                return True
                
            except Exception as e:
                st.error(f"❌ Error starting lecture: {str(e)}")
                logger.error(f"Error starting lecture: {e}")
                return False
        
        return False
    
    def get_active_lecture_info(self) -> Optional[Dict]:
        """Get current active lecture information"""
        if self.ACTIVE_LECTURE_KEY in st.session_state:
            return st.session_state[self.ACTIVE_LECTURE_KEY]
        
        # Check database for active lecture
        active_lecture = self.db.get_active_lecture()
        if active_lecture:
            # Convert to session state format
            lecture_info = {
                'lecture_id': active_lecture['lecture_id'],
                'teacher_id': active_lecture['teacher_id'],
                'teacher_name': active_lecture['teacher_name'],
                'section': active_lecture['section'],
                'subject_code': active_lecture['subject_code'],
                'subject_name': active_lecture['subject_name'],
                'cutoff_minutes': active_lecture['cutoff_minutes'],
                'start_time': active_lecture['start_time'],
                'cutoff_time': active_lecture['cutoff_time'],
                'lecture_notes': ''
            }
            st.session_state[self.ACTIVE_LECTURE_KEY] = lecture_info
            return lecture_info
        
        return None
    
    def display_lecture_status(self):
        """Display current lecture status"""
        lecture_info = self.get_active_lecture_info()
        
        if not lecture_info:
            st.info("ℹ️ No active lecture session")
            return
        
        st.success("📚 Active Lecture Session")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👨‍🏫 Teacher", lecture_info['teacher_name'])
            st.metric("🏫 Section", lecture_info['section'])
        
        with col2:
            st.metric("📝 Subject", f"{lecture_info['subject_name']}")
            st.metric("🔢 Subject Code", lecture_info['subject_code'])
        
        with col3:
            current_time = datetime.now()
            elapsed = current_time - lecture_info['start_time']
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
            
            st.metric("⏰ Elapsed Time", elapsed_str)
            
            # Check if cutoff time has passed
            if current_time > lecture_info['cutoff_time']:
                st.error("⚠️ Cutoff time passed - Late arrivals will be marked as Absent")
            else:
                remaining = lecture_info['cutoff_time'] - current_time
                remaining_str = str(remaining).split('.')[0]
                st.success(f"✅ On-time window: {remaining_str}")
        
        # Lecture details
        st.info(f"🕐 Started: {lecture_info['start_time'].strftime('%H:%M:%S')}")
        st.info(f"⏰ Cutoff: {lecture_info['cutoff_time'].strftime('%H:%M:%S')}")
        
        if lecture_info.get('lecture_notes'):
            st.text_area("📋 Lecture Notes", value=lecture_info['lecture_notes'], disabled=True)
    
    def end_lecture_session(self) -> bool:
        """End the current active lecture session"""
        lecture_info = self.get_active_lecture_info()
        
        if not lecture_info:
            st.error("❌ No active lecture to end")
            return False
        
        st.header("🛑 End Lecture Session")
        
        # Display lecture summary before ending
        self.display_lecture_summary(lecture_info['lecture_id'])
        
        st.warning("⚠️ Are you sure you want to end this lecture session?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes, End Lecture", type="primary"):
                try:
                    # End lecture in database
                    self.db.end_lecture(lecture_info['lecture_id'])
                    
                    # Clear session state
                    if self.ACTIVE_LECTURE_KEY in st.session_state:
                        del st.session_state[self.ACTIVE_LECTURE_KEY]
                    if self.RECOGNIZED_TEACHER_KEY in st.session_state:
                        del st.session_state[self.RECOGNIZED_TEACHER_KEY]
                    if self.LECTURE_STUDENTS_KEY in st.session_state:
                        del st.session_state[self.LECTURE_STUDENTS_KEY]
                    
                    st.success("🎉 Lecture session ended successfully!")
                    st.info("You can now view the final attendance report.")
                    
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Error ending lecture: {str(e)}")
                    logger.error(f"Error ending lecture: {e}")
                    return False
        
        with col2:
            if st.button("❌ Cancel"):
                st.info("Lecture session continues...")
        
        return False
    
    def display_lecture_summary(self, lecture_id: int):
        """Display lecture attendance summary"""
        try:
            # Get attendance summary
            summary = self.db.get_attendance_summary(lecture_id)
            
            st.subheader("📊 Attendance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("✅ Present", summary.get('Present', 0))
            
            with col2:
                st.metric("❌ Absent", summary.get('Absent', 0))
            
            with col3:
                st.metric("❓ Unknown", summary.get('Unknown', 0))
            
            with col4:
                st.metric("👥 Total", summary.get('Total', 0))
            
            # Calculate attendance percentage
            total = summary.get('Total', 0)
            present = summary.get('Present', 0)
            
            if total > 0:
                attendance_rate = (present / total) * 100
                st.metric("📈 Attendance Rate", f"{attendance_rate:.1f}%")
            
        except Exception as e:
            st.error(f"Error loading lecture summary: {str(e)}")
            logger.error(f"Error in display_lecture_summary: {e}")
    
    def reset_teacher_recognition(self):
        """Reset teacher recognition state"""
        if self.RECOGNIZED_TEACHER_KEY in st.session_state:
            del st.session_state[self.RECOGNIZED_TEACHER_KEY]
        st.info("Teacher recognition reset. Please verify again.")
    
    def get_section_students(self, section: str) -> List[Dict]:
        """Get all students for a specific section"""
        try:
            return self.db.get_students_by_section(section)
        except Exception as e:
            logger.error(f"Error getting section students: {e}")
            return []


