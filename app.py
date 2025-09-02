import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import threading
import queue
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules with error handling
try:
    from database import DatabaseManager
except ImportError:
    st.error("❌ DatabaseManager module not found. Please create database.py")
    st.stop()

try:
    from face_recognition import FaceRecognitionSystem
except ImportError:
    st.error("❌ FaceRecognitionSystem module not found. Please create face_recognition.py")
    st.stop()

try:
    from enrollment import EnrollmentSystem
except ImportError:
    st.error("❌ EnrollmentSystem module not found. Please create enrollment.py")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-connected {
        background-color: #28a745;
    }
    .status-disconnected {
        background-color: #dc3545;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class AttendanceApp:
    def __init__(self):
        """Initialize the attendance application"""
        self.initialize_session_state()
        self.initialize_database()
        self.initialize_face_recognition()
        self.initialize_enrollment_system()
    
    def __del__(self):
        """Cleanup resources when app is destroyed"""
        try:
            if hasattr(self, 'face_system') and self.face_system:
                self.face_system.release_camera()
            if hasattr(self, 'db') and self.db:
                self.db.close_connection()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'db_initialized' not in st.session_state:
            st.session_state.db_initialized = False
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "🏠 Dashboard"
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = None
        if 'face_system' not in st.session_state:
            st.session_state.face_system = None
    
    def initialize_database(self):
        """Initialize database connection"""
        try:
            if not st.session_state.db_initialized:
                # Get database configuration from environment variables or use defaults
                db_config = {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'user': os.getenv('DB_USER', 'root'),
                    'password': os.getenv('DB_PASSWORD', 'vidhi'),  # Consider using env variable
                    'database': os.getenv('DB_NAME', 'attendance_system')
                }
                
                self.db = DatabaseManager(**db_config)
                
                # Test connection
                if self.db.health_check():
                    st.session_state.db_initialized = True
                    st.session_state.db_manager = self.db
                    logger.info("Database initialized successfully")
                else:
                    st.error("❌ Database connection failed")
                    st.stop()
            else:
                # Reuse existing connection
                self.db = st.session_state.get('db_manager')
                if not self.db:
                    st.session_state.db_initialized = False
                    self.initialize_database()
                    
        except Exception as e:
            st.error(f"❌ Database initialization error: {str(e)}")
            st.info("Please ensure MySQL server is running and database is created")
            st.stop()
    
    def initialize_face_recognition(self):
        """Initialize face recognition system"""
        try:
            if not st.session_state.get('face_system'):
                self.face_system = FaceRecognitionSystem()
                st.session_state.face_system = self.face_system
                logger.info("Face recognition system initialized")
            else:
                self.face_system = st.session_state.face_system
        except Exception as e:
            st.error(f"❌ Face recognition initialization error: {str(e)}")
            st.stop()
    
    def initialize_enrollment_system(self):
        """Initialize enrollment system"""
        try:
            self.enrollment = EnrollmentSystem(self.db, self.face_system)
            logger.info("Enrollment system initialized")
        except Exception as e:
            st.error(f"❌ Enrollment system initialization error: {str(e)}")
            st.stop()
    
    def load_known_faces(self):
        """Load all known faces from database"""
        try:
            students = self.db.get_all_students()
            teachers = self.db.get_all_teachers()
            
            self.face_system.load_known_faces(students, teachers)
            
            return len(students), len(teachers)
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
            return 0, 0
    
    def start_camera_feed(self):
        """Start continuous camera feed for recognition"""
        if not st.session_state.camera_active:
            if self.face_system.initialize_camera():
                st.session_state.camera_active = True
                logger.info("Camera feed started")
                return True
            else:
                st.error("❌ Failed to start camera")
                return False
        return True
    
    def stop_camera_feed(self):
        """Stop camera feed"""
        if st.session_state.camera_active:
            self.face_system.release_camera()
            st.session_state.camera_active = False
            logger.info("Camera feed stopped")
    
    def process_recognition_frame(self, frame):
        """Process a single frame for face recognition"""
        current_lecture = self.db.get_active_lecture()
        
        if not current_lecture:
            return frame, []
        
        # Recognize faces in frame
        recognized_faces, annotated_frame = self.face_system.recognize_faces_in_frame(frame)
        
        # Process each recognized face for attendance
        attendance_updates = []
        
        for face_data in recognized_faces:
            if face_data['is_known']:
                # Determine attendance status based on time
                current_time = datetime.now()
                cutoff_time = current_lecture['cutoff_time']
                
                if isinstance(cutoff_time, str):
                    cutoff_time = datetime.fromisoformat(cutoff_time)
                
                status = 'Present' if current_time <= cutoff_time else 'Absent'
                
                # Mark attendance for known person
                if face_data['type'] == 'student':
                    success = self.db.mark_attendance(
                        lecture_id=current_lecture['lecture_id'],
                        student_id=face_data['id'],
                        enrollment_id=None,
                        status=status,
                        person_name=face_data['name']
                    )
                    
                    if success:
                        attendance_updates.append({
                            'name': face_data['name'],
                            'status': status,
                            'type': 'student',
                            'time': current_time.strftime("%H:%M:%S")
                        })
                
                elif face_data['type'] == 'teacher':
                    # Teacher detected - just log, don't mark attendance
                    attendance_updates.append({
                        'name': face_data['name'],
                        'status': 'Teacher Present',
                        'type': 'teacher',
                        'time': current_time.strftime("%H:%M:%S")
                    })
            
            else:
                # Unknown person detected
                success = self.db.mark_attendance(
                    lecture_id=current_lecture['lecture_id'],
                    student_id=None,
                    enrollment_id='UNKNOWN',
                    status='Unknown',
                    person_name=f"Unknown Person {datetime.now().strftime('%H%M%S')}"
                )
                
                if success:
                    attendance_updates.append({
                        'name': 'Unknown Person',
                        'status': 'Unknown',
                        'type': 'unknown',
                        'time': current_time.strftime("%H:%M:%S")
                    })
        
        return annotated_frame, attendance_updates
    
    def sidebar_navigation(self):
        """Create sidebar navigation"""
        st.sidebar.title("🎓 AI Attendance System")
        
        # System status
        st.sidebar.subheader("📊 System Status")
        
        # Database status
        db_status = "🟢 Connected" if self.db.health_check() else "🔴 Disconnected"
        st.sidebar.markdown(f"Database: {db_status}")
        
        # Camera status
        camera_status = "🟢 Active" if st.session_state.camera_active else "🔴 Inactive"
        st.sidebar.markdown(f"Camera: {camera_status}")
        
        # Current lecture status
        current_lecture = self.db.get_active_lecture()
        lecture_status = "🟢 Active" if current_lecture else "🔴 No Active Lecture"
        st.sidebar.markdown(f"Lecture: {lecture_status}")
        
        st.sidebar.markdown("---")
        
        # Navigation menu
        page = st.sidebar.selectbox(
            "📋 Navigate to:",
            [
                "🏠 Dashboard",
                "📹 Live Recognition",
                "👨‍🎓 Student Enrollment",
                "👨‍🏫 Teacher Enrollment",
                "📚 Start Lecture",
                "📊 Attendance Reports",
                "⚙️ Settings"
            ]
        )
        
        return page
    
    def dashboard_page(self):
        """Main dashboard page"""
        st.markdown('<h1 class="main-header">🎓 AI-Powered Classroom Attendance System</h1>', unsafe_allow_html=True)
        
        # System overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📹 Camera Status", 
                     "Active" if st.session_state.camera_active else "Inactive",
                     delta="Ready" if st.session_state.camera_active else "Not Ready")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            current_lecture = self.db.get_active_lecture()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📚 Active Lecture", 
                     "Yes" if current_lecture else "No",
                     delta=current_lecture['subject_name'] if current_lecture else "None")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            student_count, teacher_count = self.load_known_faces()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("👥 Known Faces", 
                     f"{student_count + teacher_count}",
                     delta=f"{student_count} Students, {teacher_count} Teachers")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Current lecture details
        if current_lecture:
            st.subheader("📚 Current Lecture Details")
            
            lecture_col1, lecture_col2, lecture_col3 = st.columns(3)
            
            with lecture_col1:
                st.info(f"**Teacher:** {current_lecture['teacher_name']}")
                st.info(f"**Subject:** {current_lecture['subject_name']}")
            
            with lecture_col2:
                st.info(f"**Section:** {current_lecture['section']}")
                st.info(f"**Subject Code:** {current_lecture['subject_code']}")
            
            with lecture_col3:
                start_time = current_lecture['start_time']
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time)
                
                cutoff_time = current_lecture['cutoff_time']
                if isinstance(cutoff_time, str):
                    cutoff_time = datetime.fromisoformat(cutoff_time)
                
                st.info(f"**Start Time:** {start_time.strftime('%H:%M:%S')}")
                st.info(f"**Cutoff Time:** {cutoff_time.strftime('%H:%M:%S')}")
            
            # End lecture button
            if st.button("🔚 End Current Lecture", type="primary"):
                self.db.end_lecture(current_lecture['lecture_id'])
                st.success("✅ Lecture ended successfully!")
                st.rerun()
            
            # Current attendance summary
            attendance_summary = self.db.get_attendance_summary(current_lecture['lecture_id'])
            
            st.subheader("📊 Live Attendance Summary")
            
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("✅ Present", attendance_summary['Present'])
            
            with summary_col2:
                st.metric("❌ Absent/Late", attendance_summary['Absent'])
            
            with summary_col3:
                st.metric("❓ Unknown", attendance_summary['Unknown'])
            
            with summary_col4:
                st.metric("👥 Total", attendance_summary['Total'])
        
        else:
            st.info("📝 No active lecture. Go to 'Start Lecture' to begin a new session.")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("📹 Start Live Recognition", type="secondary"):
                st.session_state.selected_page = "📹 Live Recognition"
                st.rerun()
        
        with action_col2:
            if st.button("👨‍🎓 Enroll Student", type="secondary"):
                st.session_state.selected_page = "👨‍🎓 Student Enrollment"
                st.rerun()
        
        with action_col3:
            if st.button("📚 Start New Lecture", type="secondary"):
                st.session_state.selected_page = "📚 Start Lecture"
                st.rerun()
        
        # Display enrollment statistics
        try:
            self.enrollment.display_enrollment_stats()
        except Exception as e:
            logger.error(f"Error displaying enrollment stats: {e}")
    
    def live_recognition_page(self):
        """Live face recognition page"""
        st.header("📹 Live Face Recognition")
        
        # Check if there's an active lecture
        current_lecture = self.db.get_active_lecture()
        
        if not current_lecture:
            st.warning("⚠️ No active lecture found. Please start a lecture first.")
            if st.button("📚 Go to Start Lecture"):
                st.session_state.selected_page = "📚 Start Lecture"
                st.rerun()
            return
        
        # Load known faces
        student_count, teacher_count = self.load_known_faces()
        st.info(f"🔍 Loaded {student_count} students and {teacher_count} teachers for recognition")
        
        # Camera controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎥 Start Camera", type="primary"):
                if self.start_camera_feed():
                    st.success("✅ Camera started successfully!")
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Camera", type="secondary"):
                self.stop_camera_feed()
                st.info("🔄 Camera stopped")
                st.rerun()
        
        # Live video feed
        if st.session_state.camera_active:
            video_placeholder = st.empty()
            attendance_placeholder = st.empty()
            
            # Auto-refresh for live feed
            if st.button("🔄 Refresh Feed"):
                pass
            
            try:
                frame = self.face_system.capture_frame()
                
                if frame is not None:
                    # Process frame for recognition
                    annotated_frame, attendance_updates = self.process_recognition_frame(frame)
                    
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display video
                    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    # Display recent attendance updates
                    if attendance_updates:
                        attendance_placeholder.subheader("🎯 Recent Recognition Events")
                        
                        for update in attendance_updates:
                            status_color = {
                                'Present': '🟢',
                                'Absent': '🟡',
                                'Unknown': '🔴',
                                'Teacher Present': '🔵'
                            }.get(update['status'], '⚪')
                            
                            attendance_placeholder.success(
                                f"{status_color} {update['name']} - {update['status']} at {update['time']}"
                            )
                
                else:
                    st.error("❌ Failed to capture frame from camera")
            
            except Exception as e:
                st.error(f"❌ Error in live recognition: {str(e)}")
                logger.error(f"Live recognition error: {e}")
        
        else:
            st.info("📷 Camera is not active. Click 'Start Camera' to begin live recognition.")
    
    def student_enrollment_page(self):
        """Student enrollment page"""
        try:
            success = self.enrollment.enroll_student_interactive()
            if success:
                st.balloons()
        except Exception as e:
            st.error(f"❌ Error in student enrollment: {str(e)}")
            logger.error(f"Student enrollment error: {e}")
    
    def teacher_enrollment_page(self):
        """Teacher enrollment page"""
        try:
            success = self.enrollment.enroll_teacher_interactive()
            if success:
                st.balloons()
        except Exception as e:
            st.error(f"❌ Error in teacher enrollment: {str(e)}")
            logger.error(f"Teacher enrollment error: {e}")
    
    def start_lecture_page(self):
        """Start lecture page"""
        st.header("📚 Start New Lecture Session")
        
        # Check for existing active lecture
        current_lecture = self.db.get_active_lecture()
        
        if current_lecture:
            st.warning(f"⚠️ There is already an active lecture: {current_lecture['subject_name']} by {current_lecture['teacher_name']}")
            
            if st.button("🔚 End Current Lecture First", type="secondary"):
                self.db.end_lecture(current_lecture['lecture_id'])
                st.success("✅ Previous lecture ended successfully!")
                st.rerun()
            
            return
        
        # Load teachers for selection
        teachers = self.db.get_all_teachers()
        
        if not teachers:
            st.error("❌ No teachers enrolled. Please enroll teachers first.")
            return
        
        with st.form("start_lecture_form"):
            st.subheader("🎓 Lecture Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Teacher selection
                teacher_options = [f"{t['name']} ({t['subject_name']})" for t in teachers]
                selected_teacher_idx = st.selectbox(
                    "👨‍🏫 Select Teacher",
                    range(len(teacher_options)),
                    format_func=lambda i: teacher_options[i]
                )
                
                selected_teacher = teachers[selected_teacher_idx]
                
                # Section selection
                sections = self.db.get_section_list()
                if sections:
                    section = st.selectbox("🏫 Section", sections)
                else:
                    section = st.text_input("🏫 Section", placeholder="e.g., CSE TY A")
            
            with col2:
                # Subject code (pre-filled from teacher)
                subject_code = st.text_input(
                    "📚 Subject Code",
                    value=selected_teacher['subject_code'],
                    disabled=True
                )
                
                # Cutoff time in minutes
                cutoff_minutes = st.number_input(
                    "⏰ Attendance Cutoff (minutes)",
                    min_value=1,
                    max_value=60,
                    value=15,
                    help="Students arriving after this time will be marked as late"
                )
            
            # Additional settings
            st.subheader("⚙️ Lecture Settings")
            
            auto_end = st.checkbox(
                "🔄 Auto-end lecture after specified duration",
                help="Automatically end the lecture after a set time"
            )
            
            if auto_end:
                lecture_duration = st.number_input(
                    "📝 Lecture Duration (minutes)",
                    min_value=30,
                    max_value=180,
                    value=60
                )
            
            submit_button = st.form_submit_button("🚀 Start Lecture", type="primary")
        
        if submit_button:
            if not section:
                st.error("❌ Section is required")
                return
            
            try:
                # Start the lecture
                lecture_id = self.db.start_lecture(
                    teacher_id=selected_teacher['teacher_id'],
                    section=section,
                    subject_code=selected_teacher['subject_code'],
                    cutoff_minutes=cutoff_minutes
                )
                
                st.success(f"🎉 Lecture started successfully! Lecture ID: {lecture_id}")
                
                # Display lecture details
                st.info(f"""
                **Lecture Details:**
                - **Teacher:** {selected_teacher['name']}
                - **Subject:** {selected_teacher['subject_name']} ({selected_teacher['subject_code']})
                - **Section:** {section}
                - **Cutoff Time:** {cutoff_minutes} minutes from now
                - **Status:** Active ✅
                """)
                
                # Redirect to live recognition
                if st.button("📹 Start Live Recognition", type="primary"):
                    st.session_state.selected_page = "📹 Live Recognition"
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error starting lecture: {str(e)}")
                logger.error(f"Start lecture error: {e}")
    
    def attendance_reports_page(self):
        """Attendance reports and analytics page"""
        st.header("📊 Attendance Reports & Analytics")
        
        # Report type selection
        report_type = st.selectbox(
            "📋 Select Report Type",
            [
                "Current Lecture Summary",
                "Lecture History",
                "Student Attendance History",
                "Section-wise Analytics",
                "Export Data"
            ]
        )
        
        if report_type == "Current Lecture Summary":
            current_lecture = self.db.get_active_lecture()
            
            if not current_lecture:
                st.warning("⚠️ No active lecture found")
                return
            
            # Lecture details
            st.subheader(f"📚 {current_lecture['subject_name']} - {current_lecture['section']}")
            st.caption(f"Teacher: {current_lecture['teacher_name']}")
            
            # Attendance summary
            attendance_summary = self.db.get_attendance_summary(current_lecture['lecture_id'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("✅ Present", attendance_summary['Present'])
            with col2:
                st.metric("❌ Absent/Late", attendance_summary['Absent'])
            with col3:
                st.metric("❓ Unknown", attendance_summary['Unknown'])
            with col4:
                st.metric("👥 Total", attendance_summary['Total'])
            
            # Detailed attendance list
            st.subheader("📝 Detailed Attendance List")
            
            attendance_records = self.db.get_lecture_attendance(current_lecture['lecture_id'])
            
            if attendance_records:
                # Create DataFrame
                df_data = []
                for record in attendance_records:
                    df_data.append({
                        'Name': record['student_name'] or record['person_name'],
                        'Enrollment ID': record['student_enrollment_id'] or record['enrollment_id'],
                        'Status': record['status'],
                        'Timestamp': record['timestamp'].strftime('%H:%M:%S') if record['timestamp'] else 'N/A'
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Export current lecture
                if st.button("📥 Export Current Lecture Attendance"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"attendance_{current_lecture['subject_code']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.info("📝 No attendance records yet")
        
        elif report_type == "Export Data":
            st.subheader("📥 Export System Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("👨‍🎓 Export All Students"):
                    students = self.db.get_all_students()
                    if students:
                        df_data = [{
                            'Student ID': s['student_id'],
                            'Enrollment ID': s['enrollment_id'],
                            'Name': s['name'],
                            'Email': s['nuv_mail'],
                            'Section': s['section']
                        } for s in students]
                        
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="💾 Download Students CSV",
                            data=csv,
                            file_name=f"students_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No students to export")
            
            with col2:
                if st.button("👨‍🏫 Export All Teachers"):
                    teachers = self.db.get_all_teachers()
                    if teachers:
                        df_data = [{
                            'Teacher ID': t['teacher_id'],
                            'Name': t['name'],
                            'Email': t['nuv_mail'],
                            'Subject': t['subject_name'],
                            'Subject Code': t['subject_code']
                        } for t in teachers]
                        
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="💾 Download Teachers CSV",
                            data=csv,
                            file_name=f"teachers_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No teachers to export")
    
    def settings_page(self):
        """Application settings page"""
        st.header("⚙️ System Settings")
        
        # Database settings
        st.subheader("🗄️ Database Configuration")
        
        with st.expander("Database Connection", expanded=False):
            db_host = st.text_input("Host", value=os.getenv('DB_HOST', 'localhost'))
            db_user = st.text_input("Username", value=os.getenv('DB_USER', 'root'))
            db_password = st.text_input("Password", type="password")
            db_name = st.text_input("Database", value=os.getenv('DB_NAME', 'attendance_system'))
            
            if st.button("🔌 Test Connection"):
                st.info("Connection test feature - would test with new credentials")
        
        # Camera settings
        st.subheader("📹 Camera Configuration")
        
        with st.expander("Camera Settings", expanded=False):
            camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0)
            face_recognition_tolerance = st.slider("Recognition Tolerance", 0.1, 1.0, 0.6, 0.1)
            frame_resize_factor = st.slider("Frame Resize Factor", 0.1, 1.0, 0.25, 0.05)
            
            if st.button("🎥 Test Camera"):
                if self.face_system.initialize_camera(camera_index):
                    st.success("✅ Camera test successful")
                    frame = self.face_system.capture_frame()
                    if frame is not None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb_frame, caption="Test Frame", use_column_width=True)
                    self.face_system.release_camera()
                else:
                    st.error("❌ Camera test failed")
        
        # System maintenance
        st.subheader("🔧 System Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Session Data"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Session data cleared")
                st.rerun()
        
        with col2:
            if st.button("🔄 Restart Camera System"):
                self.stop_camera_feed()
                self.initialize_face_recognition()
                st.success("✅ Camera system restarted")
        
        # System information
        st.subheader("ℹ️ System Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.info(f"""
            **Database Status:** {"🟢 Connected" if self.db.health_check() else "🔴 Disconnected"}
            **Camera Status:** {"🟢 Active" if st.session_state.camera_active else "🔴 Inactive"}
            **OpenCV Version:** {cv2.__version__}
            """)
        
        with info_col2:
            try:
                stats = self.enrollment.get_enrollment_statistics()
                st.info(f"""
                **Total Students:** {stats['total_students']}
                **Total Teachers:** {stats['total_teachers']}
                **Total Sections:** {stats['total_sections']}
                **Total Subjects:** {stats['total_subjects']}
                """)
            except Exception as e:
                st.info(f"""
                **Statistics:** Error loading stats
                **Error:** {str(e)[:50]}...
                """)
    
    def run(self):
        """Main application runner"""
        try:
            # Sidebar navigation
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = "🏠 Dashboard"
            
            selected_page = self.sidebar_navigation()
            
            # Update selected page
            if selected_page != st.session_state.selected_page:
                st.session_state.selected_page = selected_page
            
            # Route to appropriate page
            if st.session_state.selected_page == "🏠 Dashboard":
                self.dashboard_page()
            
            elif st.session_state.selected_page == "📹 Live Recognition":
                self.live_recognition_page()
            
            elif st.session_state.selected_page == "👨‍🎓 Student Enrollment":
                self.student_enrollment_page()
            
            elif st.session_state.selected_page == "👨‍🏫 Teacher Enrollment":
                self.teacher_enrollment_page()
            
            elif st.session_state.selected_page == "📚 Start Lecture":
                self.start_lecture_page()
            
            elif st.session_state.selected_page == "📊 Attendance Reports":
                self.attendance_reports_page()
            
            elif st.session_state.selected_page == "⚙️ Settings":
                self.settings_page()
            
            # Footer
            st.markdown("---")
            st.markdown(
                "<div style='text-align: center; color: gray;'>"
                "🎓 AI-Powered Classroom Attendance System | "
                f"Built with Streamlit & OpenCV | {datetime.now().strftime('%Y')}"
                "</div>",
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")
            logger.error(f"Application error: {e}")
            
            # Show debug info in development
            if os.getenv('DEBUG', 'False').lower() == 'true':
                st.error(f"Debug info: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def main():
    """Main function to run the application"""
    try:
        # Check if running in development mode
        if os.getenv('DEBUG', 'False').lower() == 'true':
            st.sidebar.info("🔧 Debug mode enabled")
        
        # Initialize and run the application
        app = AttendanceApp()
        app.run()
        
    except Exception as e:
        st.error(f"❌ Failed to start application: {str(e)}")
        logger.exception("Application startup error")
        
        # Show helpful error messages
        st.info("""
        **Troubleshooting Tips:**
        - Ensure MySQL server is running
        - Check database credentials in .env file
        - Verify camera is connected and not used by another app
        - Make sure all required modules are installed
        - Check that database tables exist
        """)
        
        # Show requirements
        st.subheader("📋 Required Files")
        st.code("""
        Create these files in your project directory:
        
        1. database.py - DatabaseManager class
        2. face_recognition.py - FaceRecognitionSystem class  
        3. enrollment.py - EnrollmentSystem class
        4. .env - Environment variables
        5. requirements.txt - Python dependencies
        """)
        
        # Show sample .env file
        st.subheader("📄 Sample .env file")
        st.code("""
        DB_HOST=localhost
        DB_USER=root
        DB_PASSWORD=vidhi
        DB_NAME=attendance_system
        DEBUG=True
        """)


# Main execution
if __name__ == "__main__":
    main()