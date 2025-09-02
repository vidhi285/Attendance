# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Dict, List, Optional, Any
# import logging
# import os
# import io
# import base64

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class AttendanceReportGenerator:
#     """Generate various attendance reports and exports"""
    
#     def __init__(self, db_manager):
#         """Initialize report generator"""
#         self.db = db_manager
        
#     def generate_lecture_report(self, lecture_id: int) -> pd.DataFrame:
#         """Generate detailed lecture attendance report"""
#         try:
#             # Get lecture info
#             lecture_info = self.db.get_lecture_by_id(lecture_id)
#             if not lecture_info:
#                 raise ValueError(f"Lecture {lecture_id} not found")
            
#             # Get attendance records
#             attendance_records = self.db.get_lecture_attendance(lecture_id)
            
#             # Get all students in section for complete report
#             section_students = self.db.get_students_by_section(lecture_info['section'])
            
#             # Create comprehensive report
#             report_data = []
            
#             # Track attended students
#             attended_students = {record['student_id']: record for record in attendance_records if record['student_id']}
            
#             # Add all section students
#             for student in section_students:
#                 student_id = student['student_id']
                
#                 if student_id in attended_students:
#                     record = attended_students[student_id]
#                     report_data.append({
#                         'Enrollment ID': student['enrollment_id'],
#                         'Student Name': student['name'],
#                         'Email': student['nuv_mail'],
#                         'Section': student['section'],
#                         'Status': record['status'],
#                         'Timestamp': record['timestamp'].strftime('%H:%M:%S') if record['timestamp'] else '',
#                         'Attendance Date': lecture_info['start_time'].strftime('%Y-%m-%d'),
#                         'Subject': lecture_info['subject_name'],
#                         'Subject Code': lecture_info['subject_code'],
#                         'Teacher': lecture_info['teacher_name']
#                     })
#                 else:
#                     # Student was absent
#                     report_data.append({
#                         'Enrollment ID': student['enrollment_id'],
#                         'Student Name': student['name'],
#                         'Email': student['nuv_mail'],
#                         'Section': student['section'],
#                         'Status': 'Absent (Not Detected)',
#                         'Timestamp': '',
#                         'Attendance Date': lecture_info['start_time'].strftime('%Y-%m-%d'),
#                         'Subject': lecture_info['subject_name'],
#                         'Subject Code': lecture_info['subject_code'],
#                         'Teacher': lecture_info['teacher_name']
#                     })
            
#             # Add unknown persons
#             for record in attendance_records:
#                 if not record['student_id'] and record['status'] == 'Unknown':
#                     report_data.append({
#                         'Enrollment ID': 'Unknown',
#                         'Student Name': 'Unknown Person',
#                         'Email': '',
#                         'Section': lecture_info['section'],
#                         'Status': 'Unknown',
#                         'Timestamp': record['timestamp'].strftime('%H:%M:%S'),
#                         'Attendance Date': lecture_info['start_time'].strftime('%Y-%m-%d'),
#                         'Subject': lecture_info['subject_name'],
#                         'Subject Code': lecture_info['subject_code'],
#                         'Teacher': lecture_info['teacher_name']
#                     })
            
#             df = pd.DataFrame(report_data)
#             return df
            
#         except Exception as e:
#             logger.error(f"Error generating lecture report: {e}")
#             return pd.DataFrame()
    
#     def export_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
#         """Export DataFrame to CSV file"""
#         try:
#             if filename is None:
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"attendance_report_{timestamp}.csv"
            
#             # Create reports directory
#             os.makedirs("reports", exist_ok=True)
#             filepath = os.path.join("reports", filename)
            
#             df.to_csv(filepath, index=False)
#             logger.info(f"Report exported to CSV: {filepath}")
#             return filepath
            
#         except Exception as e:
#             logger.error(f"Error exporting to CSV: {e}")
#             return ""
    
#     def export_to_excel(self, df: pd.DataFrame, filename: str = None) -> str:
#         """Export DataFrame to Excel file with formatting"""
#         try:
#             if filename is None:
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"attendance_report_{timestamp}.xlsx"
            
#             # Create reports directory
#             os.makedirs("reports", exist_ok=True)
#             filepath = os.path.join("reports", filename)
            
#             # Create Excel writer with formatting
#             with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
#                 df.to_excel(writer, sheet_name='Attendance Report', index=False)
                
#                 # Get workbook and worksheet for formatting
#                 workbook = writer.book
#                 worksheet = writer.sheets['Attendance Report']
                
#                 # Auto-adjust column widths
#                 for column in worksheet.columns:
#                     max_length = 0
#                     column_letter = column[0].column_letter
                    
#                     for cell in column:
#                         try:
#                             if len(str(cell.value)) > max_length:
#                                 max_length = len(str(cell.value))
#                         except:
#                             pass
                    
#                     adjusted_width = min(max_length + 2, 50)
#                     worksheet.column_dimensions[column_letter].width = adjusted_width
            
#             logger.info(f"Report exported to Excel: {filepath}")
#             return filepath
            
#         except Exception as e:
#             logger.error(f"Error exporting to Excel: {e}")
#             return ""
    
#     def generate_summary_statistics(self, lecture_id: int) -> Dict[str, Any]:
#         """Generate comprehensive attendance statistics"""
#         try:
#             lecture_info = self.db.get_lecture_by_id(lecture_id)
#             if not lecture_info:
#                 return {}
            
#             # Get basic summary
#             summary = self.db.get_attendance_summary(lecture_id)
            
#             # Get section student count
#             section_students = self.db.get_students_by_section(lecture_info['section'])
#             total_expected = len(section_students)
            
#             # Calculate percentages
#             present_count = summary.get('Present', 0)
#             absent_count = total_expected - present_count
#             unknown_count = summary.get('Unknown', 0)
            
#             present_percentage = (present_count / total_expected * 100) if total_expected > 0 else 0
#             absent_percentage = (absent_count / total_expected * 100) if total_expected > 0 else 0
            
#             return {
#                 'lecture_info': {
#                     'lecture_id': lecture_id,
#                     'date': lecture_info['start_time'].strftime('%Y-%m-%d'),
#                     'time': lecture_info['start_time'].strftime('%H:%M:%S'),
#                     'subject': lecture_info['subject_name'],
#                     'subject_code': lecture_info['subject_code'],
#                     'section': lecture_info['section'],
#                     'teacher': lecture_info['teacher_name'],
#                     'cutoff_minutes': lecture_info['cutoff_minutes']
#                 },
#                 'attendance_stats': {
#                     'total_expected': total_expected,
#                     'present_count': present_count,
#                     'absent_count': absent_count,
#                     'unknown_count': unknown_count,
#                     'present_percentage': round(present_percentage, 2),
#                     'absent_percentage': round(absent_percentage, 2)
#                 },
#                 'timing_info': {
#                     'start_time': lecture_info['start_time'],
#                     'end_time': lecture_info['end_time'],
#                     'cutoff_time': lecture_info['cutoff_time'],
#                     'duration_minutes': self._calculate_duration(lecture_info['start_time'], lecture_info['end_time'])
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"Error generating summary statistics: {e}")
#             return {}
    
#     def _calculate_duration(self, start_time: datetime, end_time: Optional[datetime]) -> int:
#         """Calculate lecture duration in minutes"""
#         if not end_time:
#             end_time = datetime.now()
        
#         duration = end_time - start_time
#         return int(duration.total_seconds() / 60)

# class TimeUtils:
#     """Time-related utility functions"""
    
#     @staticmethod
#     def format_duration(seconds: int) -> str:
#         """Format duration in seconds to human-readable format"""
#         if seconds < 60:
#             return f"{seconds}s"
#         elif seconds < 3600:
#             minutes = seconds // 60
#             remaining_seconds = seconds % 60
#             return f"{minutes}m {remaining_seconds}s"
#         else:
#             hours = seconds // 3600
#             remaining_minutes = (seconds % 3600) // 60
#             return f"{hours}h {remaining_minutes}m"
    
#     @staticmethod
#     def is_within_timeframe(target_time: datetime, reference_time: datetime, 
#                            tolerance_minutes: int = 5) -> bool:
#         """Check if target time is within tolerance of reference time"""
#         time_diff = abs((target_time - reference_time).total_seconds() / 60)
#         return time_diff <= tolerance_minutes
    
#     @staticmethod
#     def get_academic_week(date: datetime) -> int:
#         """Get academic week number (assuming semester starts in August)"""
#         # Define semester start (adjust as needed)
#         current_year = date.year
#         if date.month >= 8:
#             semester_start = datetime(current_year, 8, 1)
#         else:
#             semester_start = datetime(current_year - 1, 8, 1)
        
#         week_diff = (date - semester_start).days // 7
#         return max(1, week_diff + 1)

# class ValidationUtils:
#     """Validation utility functions"""
    
#     @staticmethod
#     def validate_enrollment_id(enrollment_id: str) -> bool:
#         """Validate enrollment ID format"""
#         import re
#         pattern = re.compile(r'^[A-Za-z0-9]{6,20}$')
#         return bool(pattern.match(enrollment_id.strip()))
    
#     @staticmethod
#     def validate_email(email: str) -> bool:
#         """Validate email format"""
#         import re
#         pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
#         return bool(pattern.match(email.strip()))
    
#     @staticmethod
#     def validate_section_format(section: str) -> bool:
#         """Validate section format (e.g., CSE TY A, BCA FY B)"""
#         import re
#         # Pattern: Department + Year + Division
#         pattern = re.compile(r'^[A-Z]{2,5}\s+[A-Z]{2}\s+[A-Z]$')
#         return bool(pattern.match(section.strip().upper()))
    
#     @staticmethod
#     def sanitize_filename(filename: str) -> str:
#         """Sanitize filename for safe file system usage"""
#         import re
#         # Remove or replace invalid characters
#         sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
#         # Remove extra spaces and limit length
#         sanitized = '_'.join(sanitized.split())
#         return sanitized[:100]  # Limit to 100 characters

# class DataUtils:
#     """Data processing utility functions"""
    
#     @staticmethod
#     def calculate_attendance_trends(attendance_data: List[Dict]) -> Dict[str, float]:
#         """Calculate attendance trends from historical data"""
#         if not attendance_data:
#             return {'trend': 0.0, 'average': 0.0}
        
#         # Group by date and calculate daily percentages
#         daily_percentages = {}
        
#         for record in attendance_data:
#             date = record['date']
#             if date not in daily_percentages:
#                 daily_percentages[date] = {'present': 0, 'total': 0}
            
#             daily_percentages[date]['total'] += 1
#             if record['status'] == 'Present':
#                 daily_percentages[date]['present'] += 1
        
#         # Calculate percentages
#         percentages = []
#         for date, data in daily_percentages.items():
#             if data['total'] > 0:
#                 percentage = (data['present'] / data['total']) * 100
#                 percentages.append(percentage)
        
#         if not percentages:
#             return {'trend': 0.0, 'average': 0.0}
        
#         average = sum(percentages) / len(percentages)
        
#         # Simple trend calculation (last vs first)
#         if len(percentages) > 1:
#             trend = percentages[-1] - percentages[0]
#         else:
#             trend = 0.0
        
#         return {'trend': trend, 'average': average}
    
#     @staticmethod
#     def export_to_base64_csv(df: pd.DataFrame) -> str:
#         """Export DataFrame to base64 encoded CSV for download"""
#         try:
#             csv_buffer = io.StringIO()
#             df.to_csv(csv_buffer, index=False)
#             csv_string = csv_buffer.getvalue()
#             csv_bytes = csv_string.encode('utf-8')
#             csv_base64 = base64.b64encode(csv_bytes).decode('utf-8')
#             return csv_base64
#         except Exception as e:
#             logger.error(f"Error creating base64 CSV: {e}")
#             return ""

# class ConfigManager:
#     """Configuration management utilities"""
    
#     DEFAULT_CONFIG = {
#         'camera': {
#             'index': 0,
#             'width': 640,
#             'height': 480,
#             'fps': 30
#         },
#         'recognition': {
#             'tolerance': 0.6,
#             'cooldown_seconds': 3,
#             'confidence_threshold': 0.5
#         },
#         'attendance': {
#             'default_cutoff_minutes': 15,
#             'max_cutoff_minutes': 60
#         },
#         'database': {
#             'host': 'localhost',
#             'user': 'root',
#             'password': '',
#             'database': 'attendance_system'
#         }
#     }
    
#     @classmethod
#     def get_config(cls) -> Dict[str, Any]:
#         """Get configuration with defaults"""
#         try:
#             # Try to load from config file if exists
#             import json
#             if os.path.exists('config.json'):
#                 with open('config.json', 'r') as f:
#                     user_config = json.load(f)
#                 # Merge with defaults
#                 config = cls.DEFAULT_CONFIG.copy()
#                 config.update(user_config)
#                 return config
#         except:
#             pass
        
#         return cls.DEFAULT_CONFIG.copy()
    
#     @classmethod
#     def save_config(cls, config: Dict[str, Any]):
#         """Save configuration to file"""
#         try:
#             import json
#             with open('config.json', 'w') as f:
#                 json.dump(config, f, indent=2)
#             logger.info("Configuration saved")
#         except Exception as e:
#             logger.error(f"Error saving configuration: {e}")

# def create_sample_data():
#     """Create sample data for testing (optional)"""
#     sample_students = [
#         {
#             'enrollment_id': '2024CSE001',
#             'name': 'Alice Johnson',
#             'nuv_mail': 'alice.johnson@nuv.ac.in',
#             'section': 'CSE TY A'
#         },
#         {
#             'enrollment_id': '2024CSE002', 
#             'name': 'Bob Smith',
#             'nuv_mail': 'bob.smith@nuv.ac.in',
#             'section': 'CSE TY A'
#         },
#         {
#             'enrollment_id': '2024BCA001',
#             'name': 'Carol Brown',
#             'nuv_mail': 'carol.brown@nuv.ac.in', 
#             'section': 'BCA FY B'
#         }
#     ]
    
#     sample_teachers = [
#         {
#             'name': 'Dr. Emily Davis',
#             'nuv_mail': 'emily.davis@nuv.ac.in',
#             'subject_name': 'Data Structures and Algorithms',
#             'subject_code': 'CSE301'
#         },
#         {
#             'name': 'Prof. Michael Wilson',
#             'nuv_mail': 'michael.wilson@nuv.ac.in',
#             'subject_name': 'Database Management Systems',
#             'subject_code': 'CSE302'
#         }
#     ]
    
#     return sample_students, sample_teachers


import os
import csv
import json
import logging
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import zipfile
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Utils:
    """
    Utility class for the AI-Powered Classroom Attendance System
    Contains helper functions for reports, exports, notifications, and system maintenance
    """
    
    def _init_(self, database_manager=None):
        """
        Initialize Utils class
        
        Args:
            database_manager: Optional DatabaseManager instance
        """
        self.db = database_manager
        
        # Create output directories
        self.create_output_directories()
        
        # Email configuration (you should set these in environment variables)
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email': os.getenv('SYSTEM_EMAIL', ''),
            'password': os.getenv('SYSTEM_EMAIL_PASSWORD', ''),
            'use_tls': True
        }
        
        # Report templates
        self.report_templates = {
            'daily_summary': 'Daily Attendance Summary - {date}',
            'lecture_report': 'Lecture Report - {subject} ({section})',
            'student_report': 'Student Attendance Report - {student_name}',
            'section_analysis': 'Section Analysis - {section}'
        }
    
    def create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            'reports',
            'exports', 
            'logs',
            'backup',
            'temp',
            'attachments'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directory '{directory}' created/verified")
    
    # Time and Date Utilities
    @staticmethod
    def get_current_time() -> datetime:
        """Get current timestamp"""
        return datetime.now()
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """
        Format datetime object to string
        
        Args:
            dt: Datetime object
            format_str: Format string
            
        Returns:
            str: Formatted datetime string
        """
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_datetime(date_str: str, format_str: str = '%Y-%m-%d %H:%M:%S') -> datetime:
        """
        Parse datetime string to datetime object
        
        Args:
            date_str: Date string
            format_str: Format string
            
        Returns:
            datetime: Parsed datetime object
        """
        return datetime.strptime(date_str, format_str)
    
    @staticmethod
    def get_academic_year() -> str:
        """
        Get current academic year (April to March)
        
        Returns:
            str: Academic year in format "2023-24"
        """
        now = datetime.now()
        if now.month >= 4:  # April onwards
            start_year = now.year
            end_year = now.year + 1
        else:  # January to March
            start_year = now.year - 1
            end_year = now.year
        
        return f"{start_year}-{str(end_year)[2:]}"
    
    @staticmethod
    def get_semester() -> str:
        """
        Get current semester based on month
        
        Returns:
            str: Current semester (Odd/Even)
        """
        month = datetime.now().month
        # Odd semester: July-December, Even semester: January-June
        return "Odd" if 7 <= month <= 12 else "Even"
    
    @staticmethod
    def calculate_duration(start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Calculate duration between two timestamps
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            dict: Duration breakdown
        """
        duration = end_time - start_time
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60