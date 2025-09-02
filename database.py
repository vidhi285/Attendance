import mysql.connector
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles all database operations for the attendance system"""
    
    def __init__(self, host='localhost', user='root', password='vidhi', database='attendance_system'):
        """Initialize database connection"""
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self._connected = False
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        if self._connected and self.connection is not None:
            try:
                # Test if connection is still alive
                self.connection.ping(reconnect=True)
                return self.connection
            except:
                self._connected = False
                
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=True  # Enable autocommit for immediate persistence
            )
            self._connected = True
            logger.info("Database connected successfully!")
            return self.connection
        except mysql.connector.Error as err:
            logger.error(f"Database connection failed: {err}")
            self._connected = False
            return None

    def close(self):
        """Close database connection"""
        if self.connection is not None and self._connected:
            self.connection.close()
            self._connected = False
            logger.info("Database connection closed")

    def _get_cursor(self):
        """Helper: ensure connected and return a new cursor (dictionary=True)."""
        if not self.connect():
            raise RuntimeError("Database not connected")
        return self.connection.cursor(dictionary=True)

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute a database query safely with better error handling."""
        cursor = None
        try:
            cursor = self._get_cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if fetch:
                rows = cursor.fetchall()
                return rows
            else:
                # For insert/update operations
                affected_rows = cursor.rowcount
                last_id = cursor.lastrowid if hasattr(cursor, "lastrowid") else None
                
                # Force commit if autocommit is off
                if not self.connection.autocommit:
                    self.connection.commit()
                    
                return last_id if last_id else affected_rows
                
        except mysql.connector.Error as err:
            logger.error(f"Query execution error: {err} -- Query: {query} -- Params: {params}")
            if self.connection and not self.connection.autocommit:
                try:
                    self.connection.rollback()
                except:
                    pass
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
    
    # Student Operations
    def enroll_student(self, enrollment_id: str, name: str, nuv_mail: str,
                       section: str, face_encoding: np.ndarray) -> int:
        """Enroll a new student with proper encoding handling"""
        try:
            # Convert numpy array to JSON string
            if isinstance(face_encoding, np.ndarray):
                fe_json = json.dumps(face_encoding.tolist())
            else:
                raise ValueError("Face encoding must be a numpy array")
                
            query = """
            INSERT INTO students (enrollment_id, name, nuv_mail, section, face_encoding)
            VALUES (%s, %s, %s, %s, %s)
            """
            params = (enrollment_id, name, nuv_mail, section, fe_json)
            
            student_id = self.execute_query(query, params)
            logger.info(f"Student enrolled successfully: {name} (ID: {student_id})")
            
            # Verify the insertion
            verify_query = "SELECT student_id FROM students WHERE student_id = %s"
            result = self.execute_query(verify_query, (student_id,), fetch=True)
            if not result:
                raise Exception("Student insertion verification failed")
                
            return student_id
            
        except Exception as e:
            logger.error(f"Error enrolling student {name}: {e}")
            raise
    
    def get_all_students(self) -> List[Dict]:
        """Retrieve all students with proper face encoding conversion"""
        try:
            query = "SELECT * FROM students ORDER BY name"
            rows = self.execute_query(query, fetch=True)
            
            for r in rows:
                try:
                    encoding_str = r.get('face_encoding', '[]')
                    if encoding_str:
                        r['face_encoding'] = np.array(json.loads(encoding_str), dtype=np.float64)
                    else:
                        r['face_encoding'] = np.array([])
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Invalid face encoding for student {r.get('name', 'Unknown')}: {e}")
                    r['face_encoding'] = np.array([])
                    
            logger.info(f"Retrieved {len(rows)} students from database")
            return rows
        except Exception as e:
            logger.error(f"Error retrieving students: {e}")
            return []
    
    def get_student_by_id(self, student_id: int) -> Optional[Dict]:
        """Get student by student_id"""
        try:
            query = "SELECT * FROM students WHERE student_id = %s"
            result = self.execute_query(query, (student_id,), fetch=True)
            
            if result:
                student = result[0]
                try:
                    student['face_encoding'] = np.array(json.loads(student['face_encoding']), dtype=np.float64)
                except:
                    student['face_encoding'] = np.array([])
                return student
            return None
        except Exception as e:
            logger.error(f"Error retrieving student {student_id}: {e}")
            return None
    
    # Teacher Operations  
    def enroll_teacher(self, name: str, nuv_mail: str, subject_name: str,
                      subject_code: str, face_encoding: np.ndarray) -> int:
        """Enroll a new teacher with proper encoding handling"""
        try:
            # Convert numpy array to JSON string
            if isinstance(face_encoding, np.ndarray):
                face_encoding_json = json.dumps(face_encoding.tolist())
            else:
                raise ValueError("Face encoding must be a numpy array")
            
            query = """
            INSERT INTO teachers (name, nuv_mail, subject_name, subject_code, face_encoding)
            VALUES (%s, %s, %s, %s, %s)
            """
            params = (name, nuv_mail, subject_name, subject_code, face_encoding_json)
            
            teacher_id = self.execute_query(query, params)
            logger.info(f"Teacher enrolled successfully: {name} (ID: {teacher_id})")
            
            # Verify the insertion
            verify_query = "SELECT teacher_id FROM teachers WHERE teacher_id = %s"
            result = self.execute_query(verify_query, (teacher_id,), fetch=True)
            if not result:
                raise Exception("Teacher insertion verification failed")
                
            return teacher_id
            
        except Exception as e:
            logger.error(f"Error enrolling teacher {name}: {e}")
            raise
    
    def get_all_teachers(self) -> List[Dict]:
        """Retrieve all teachers with proper face encoding conversion"""
        try:
            query = "SELECT * FROM teachers ORDER BY name"
            teachers = self.execute_query(query, fetch=True)
            
            for teacher in teachers:
                try:
                    encoding_str = teacher.get('face_encoding', '[]')
                    if encoding_str:
                        teacher['face_encoding'] = np.array(json.loads(encoding_str), dtype=np.float64)
                    else:
                        teacher['face_encoding'] = np.array([])
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Invalid face encoding for teacher {teacher.get('name', 'Unknown')}: {e}")
                    teacher['face_encoding'] = np.array([])
                    
            logger.info(f"Retrieved {len(teachers)} teachers from database")
            return teachers
        except Exception as e:
            logger.error(f"Error retrieving teachers: {e}")
            return []
    
    # Lecture Operations - FIXED
    def start_lecture(self, teacher_id: int, subject_name: str, subject_code: str,
                     section: str, duration_minutes: int, cutoff_minutes: int) -> int:
        """Start a new lecture session with proper parameter handling"""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            cutoff_time = start_time + timedelta(minutes=cutoff_minutes)
            
            # First, end any existing active lectures
            end_active_query = "UPDATE lectures SET is_active = FALSE WHERE is_active = TRUE"
            self.execute_query(end_active_query)
            
            query = """
            INSERT INTO lectures (teacher_id, subject_name, subject_code, section, 
                                start_time, end_time, cutoff_minutes, cutoff_time, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (teacher_id, subject_name, subject_code, section, 
                     start_time, end_time, cutoff_minutes, cutoff_time, True)
            
            lecture_id = self.execute_query(query, params)
            logger.info(f"Lecture started successfully: ID {lecture_id}, Section: {section}")
            return lecture_id
            
        except Exception as e:
            logger.error(f"Error starting lecture: {e}")
            raise
    
    def end_lecture(self, lecture_id: int) -> bool:
        """End an active lecture session"""
        try:
            end_time = datetime.now()
            query = """
            UPDATE lectures SET end_time = %s, is_active = FALSE 
            WHERE lecture_id = %s AND is_active = TRUE
            """
            params = (end_time, lecture_id)
            
            affected_rows = self.execute_query(query, params)
            if affected_rows > 0:
                logger.info(f"Lecture ended successfully: ID {lecture_id}")
                return True
            else:
                logger.warning(f"No active lecture found with ID {lecture_id}")
                return False
        except Exception as e:
            logger.error(f"Error ending lecture {lecture_id}: {e}")
            return False
    
    def get_active_lecture(self) -> Optional[Dict]:
        """Get currently active lecture with teacher details"""
        try:
            query = """
            SELECT l.*, t.name as teacher_name
            FROM lectures l
            JOIN teachers t ON l.teacher_id = t.teacher_id
            WHERE l.is_active = TRUE
            ORDER BY l.start_time DESC
            LIMIT 1
            """
            result = self.execute_query(query, fetch=True)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting active lecture: {e}")
            return None
    
    # Attendance Operations - FIXED
    def mark_attendance(self, lecture_id: int, student_id: Optional[int], 
                       enrollment_id: Optional[str], status: str, 
                       person_name: str = None) -> bool:
        """Mark attendance with proper duplicate checking"""
        try:
            timestamp = datetime.now()
            
            # Check if attendance already exists for this lecture and person
            if student_id:
                check_query = """
                SELECT attendance_id FROM attendance 
                WHERE lecture_id = %s AND student_id = %s
                """
                existing = self.execute_query(check_query, (lecture_id, student_id), fetch=True)
                if existing:
                    logger.info(f"Attendance already marked for student ID {student_id} in lecture {lecture_id}")
                    return False
            else:
                # For unknown persons, check by name and enrollment_id
                check_query = """
                SELECT attendance_id FROM attendance 
                WHERE lecture_id = %s AND enrollment_id = %s AND person_name = %s
                """
                existing = self.execute_query(check_query, (lecture_id, enrollment_id, person_name), fetch=True)
                if existing:
                    logger.info(f"Attendance already marked for {person_name} in lecture {lecture_id}")
                    return False
            
            # Insert new attendance record
            query = """
            INSERT INTO attendance (lecture_id, student_id, enrollment_id, timestamp, status, person_name)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (lecture_id, student_id, enrollment_id, timestamp, status, person_name)
            
            attendance_id = self.execute_query(query, params)
            if attendance_id:
                logger.info(f"Attendance marked successfully: {person_name or f'Student ID {student_id}'} - {status}")
                return True
            else:
                logger.error("Failed to insert attendance record")
                return False
                
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            return False
    
    def get_attendance_summary(self, lecture_id: int) -> Dict:
        """Get attendance summary for a lecture"""
        try:
            query = """
            SELECT 
                status,
                COUNT(*) as count
            FROM attendance
            WHERE lecture_id = %s
            GROUP BY status
            """
            results = self.execute_query(query, (lecture_id,), fetch=True)
            
            summary = {'Present': 0, 'Absent': 0, 'Unknown': 0, 'Total': 0}
            
            for row in results:
                summary[row['status']] = row['count']
                summary['Total'] += row['count']
            
            return summary
        except Exception as e:
            logger.error(f"Error getting attendance summary: {e}")
            return {'Present': 0, 'Absent': 0, 'Unknown': 0, 'Total': 0}
    
    def get_recent_lectures(self, limit: int = 10) -> List[Dict]:
        """Get recent lectures"""
        try:
            query = """
            SELECT l.*, t.name as teacher_name
            FROM lectures l
            JOIN teachers t ON l.teacher_id = t.teacher_id
            ORDER BY l.start_time DESC
            LIMIT %s
            """
            return self.execute_query(query, (limit,), fetch=True)
        except Exception as e:
            logger.error(f"Error getting recent lectures: {e}")
            return []
    
    def get_lecture_attendance(self, lecture_id: int) -> List[Dict]:
        """Get all attendance records for a specific lecture"""
        try:
            query = """
            SELECT a.*, s.name as student_name, s.enrollment_id as student_enrollment_id
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            WHERE a.lecture_id = %s
            ORDER BY a.timestamp
            """
            return self.execute_query(query, (lecture_id,), fetch=True)
        except Exception as e:
            logger.error(f"Error getting lecture attendance: {e}")
            return []
    
    # Utility Functions
    def get_section_list(self) -> List[str]:
        """Get list of all unique sections"""
        try:
            query = "SELECT DISTINCT section FROM students ORDER BY section"
            results = self.execute_query(query, fetch=True)
            return [row['section'] for row in results]
        except Exception as e:
            logger.error(f"Error getting sections: {e}")
            return []
    
    def get_subject_codes(self) -> List[str]:
        """Get list of all unique subject codes"""
        try:
            query = "SELECT DISTINCT subject_code FROM teachers ORDER BY subject_code"
            results = self.execute_query(query, fetch=True)
            return [row['subject_code'] for row in results]
        except Exception as e:
            logger.error(f"Error getting subject codes: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            self.execute_query("SELECT 1", fetch=True)
            return True
        except:
            return False

    def get_student_attendance_history(self, student_id: int, start_date, end_date) -> List[Dict]:
        """Get student attendance history for date range"""
        try:
            query = """
            SELECT a.*, l.subject_name, l.subject_code, l.section
            FROM attendance a
            JOIN lectures l ON a.lecture_id = l.lecture_id
            WHERE a.student_id = %s AND DATE(a.timestamp) BETWEEN %s AND %s
            ORDER BY a.timestamp DESC
            """
            return self.execute_query(query, (student_id, start_date, end_date), fetch=True)
        except Exception as e:
            logger.error(f"Error getting student attendance history: {e}")
            return []

    def get_teacher_statistics(self) -> List[Dict]:
        """Get teacher statistics"""
        try:
            query = """
            SELECT t.name, t.subject_name, COUNT(l.lecture_id) as total_lectures
            FROM teachers t
            LEFT JOIN lectures l ON t.teacher_id = l.teacher_id
            GROUP BY t.teacher_id, t.name, t.subject_name
            ORDER BY total_lectures DESC
            """
            return self.execute_query(query, fetch=True)
        except Exception as e:
            logger.error(f"Error getting teacher statistics: {e}")
            return []

    def get_daily_attendance(self, date) -> List[Dict]:
        """Get daily attendance report"""
        try:
            query = """
            SELECT a.*, s.name as student_name, s.enrollment_id, 
                   l.subject_name, l.subject_code, l.section
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            JOIN lectures l ON a.lecture_id = l.lecture_id
            WHERE DATE(a.timestamp) = %s
            ORDER BY a.timestamp
            """
            return self.execute_query(query, (date,), fetch=True)
        except Exception as e:
            logger.error(f"Error getting daily attendance: {e}")
            return []

