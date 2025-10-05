# ==============================================================================
#                 Unified Facial Recognition and Web API System
# ==============================================================================
# This script combines the data enrollment/recognition logic with the Flask web API.
# It will first check and enroll employees from the 'employees' folder if the
# database is empty, and then it will launch the web server to provide API access
# to the data.
# ==============================================================================

# --- Core Imports ---
import os
import platform
import random
import string
import time
import tempfile
import shutil
from datetime import datetime, date, timedelta
from functools import wraps

# --- Authentication Decorators ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Decode JWT token to get user identity
            from flask_jwt_extended import decode_token
            decoded_token = decode_token(token)
            user_email = decoded_token['sub']
            
            # Get user from database
            user = User.query.filter_by(Email=user_email).first()
            if not user or user.Role != 'admin':
                return jsonify({'error': 'Admin access required'}), 403
                
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
    return decorated_function

def employee_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Decode JWT token to get user identity
            from flask_jwt_extended import decode_token
            decoded_token = decode_token(token)
            user_email = decoded_token['sub']
            
            # Get user from database
            user = User.query.filter_by(Email=user_email).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
                
            # Add user info to request context for use in the route
            request.current_user = user
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
    return decorated_function

# --- Force TensorFlow to use CPU ---
# This must be done BEFORE any tensorflow or keras imports.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- Machine Learning Imports ---
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

# --- Web Framework and Database Imports ---
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, make_response, g, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager, verify_jwt_in_request, get_jwt, set_access_cookies, unset_jwt_cookies
from flask_babel import Babel, gettext, ngettext

# --- PDF Generation ---
import asyncio
from playwright.async_api import async_playwright
import tempfile
import os

def generate_pdf_from_html(html_content):
    """
    Generate PDF from HTML content using Playwright.
    """
    try:
        # Use asyncio to run the async function
        return asyncio.run(_generate_pdf_async(html_content))
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise

async def _generate_pdf_async(html_content):
    """
    Async helper function for PDF generation.
    """
    try:
        # Launch browser
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            )

            # Create new page
            page = await browser.new_page()

            # Set viewport
            await page.set_viewport_size({'width': 1200, 'height': 800})

            # Set HTML content
            await page.set_content(html_content, wait_until='networkidle')

            # Wait for content to load
            await page.wait_for_selector('body', timeout=10000)

            # Generate PDF
            pdf_buffer = await page.pdf(
                format='A4',
                print_background=True,
                margin={
                    'top': '1cm',
                    'right': '1cm',
                    'bottom': '1cm',
                    'left': '1cm'
                }
            )

            # Close browser
            await browser.close()

            return pdf_buffer

    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise


# ==============================================================================
#                           SYSTEM CONFIGURATION
# ==============================================================================

# --- System Settings ---
BASE_PROJECT_DIR = os.getcwd()
DATABASE_FILE = 'facial_attendance_system.db'
DATABASE_PATH = os.path.join(BASE_PROJECT_DIR, DATABASE_FILE)
FLASK_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}' # For standalone engine

# --- Directories ---
EMPLOYEE_FOLDER_TRAINING = os.path.join(BASE_PROJECT_DIR, "employees")
NEW_EMPLOYEE_FOLDER_TESTING = os.path.join(BASE_PROJECT_DIR, "New employees")

# --- Recognition Parameters ---
TARGET_SIZE = (160, 160)
THRESHOLD = 0.60 # Cosine similarity threshold for recognition (adjusted for better accuracy)
MAX_LIVE_CAMERA_ATTEMPTS = 3
MAX_BATCH_TEST_ATTEMPTS_PER_PERSON = 2  # From new.ipynb

# ==============================================================================
#                           DATABASE SETUP (MODELS)
# ==============================================================================
# Using Flask-SQLAlchemy for integration with the Flask app context.

# --- Flask App Initialization ---
app = Flask(__name__)
# --- إعدادات الترجمة ---
# 1. مفتاح سري للتطبيق، ضروري لعمل الجلسات (sessions) التي تخزن لغة المستخدم
# الرجاء تغييره في بيئة الإنتاج إلى قيمة عشوائية وآمنة
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'

# 2. تحديد اللغات المتاحة في التطبيق
app.config['LANGUAGES'] = {
    'en': 'English',
    'ar': 'العربية'
}
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

# 3. تهيئة مكتبة الترجمة Babel
babel = Babel(app)

# 4. دالة اختيار اللغة: هذه الدالة تحدد اللغة التي سيتم استخدامها في كل طلب
# بناءً على القيمة المخزنة في جلسة المستخدم (session)
def get_locale():
    # الحصول على اللغة من الجلسة
    locale = session.get('locale', 'en')

    # التأكد من أن اللغة مدعومة
    if locale not in app.config['LANGUAGES']:
        locale = 'en'

    return locale

# Register the locale selector in a backwards-compatible way.
# Older Flask-Babel versions provided the @babel.localeselector decorator.
# Newer versions changed the API (for example: locale_selector). Try
# multiple registration methods so the code works regardless of the
# installed Flask-Babel version.
try:
    if hasattr(babel, 'locale_selector'):
        # Newer API: babel.locale_selector accepts the function
        babel.locale_selector(get_locale)
    elif hasattr(babel, 'localeselector'):
        # Older API: use localeselector (decorator style) by calling it
        babel.localeselector(get_locale)
    else:
        # Fallback: some versions expose a assignable attribute
        setattr(babel, 'locale_selector_func', get_locale)
except Exception:
    # If registration fails for any reason, continue -- the app can
    # still run and you may need to set locale elsewhere.
    pass

# Make get_locale available in Jinja templates so templates can call
# {{ get_locale() }}. Some templates in this project call get_locale()
# directly instead of using Babel's helper; expose it explicitly.
try:
    app.jinja_env.globals['get_locale'] = get_locale
except Exception:
    # If templates aren't initialized yet, ignore; Jinja globals can be
    # set later by code that imports this module.
    pass

# دالة ترجمة محسنة للتأكد من عمل الترجمة
def safe_gettext(text):
    """دالة ترجمة آمنة تستخدم force_locale إذا لزم الأمر"""
    try:
        # الحصول على اللغة الحالية
        current_locale = get_locale()

        # إذا كانت اللغة العربية، استخدم force_locale للتأكد من الترجمة
        if current_locale == 'ar':
            from flask_babel import force_locale
            with force_locale('ar'):
                return gettext(text)
        else:
            return gettext(text)
    except:
        # في حالة الخطأ، أعد النص الأصلي
        return text

# Expose Flask-Babel translation helpers to Jinja so templates using
# _('...') or gettext(...) work regardless of Babel version integration.
try:
    app.jinja_env.globals['_'] = safe_gettext
    app.jinja_env.globals['gettext'] = safe_gettext
    app.jinja_env.globals['ngettext'] = ngettext
except Exception:
    pass

# --- نهاية قسم إعداد الترجمة ---

app.config['SQLALCHEMY_DATABASE_URI'] = FLASK_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)
bcrypt = Bcrypt(app) 
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change_me_in_env")  # IMPORTANT: set JWT_SECRET_KEY in env for production
app.config["JWT_TOKEN_LOCATION"] = ["headers", "cookies"] # Allow JWT to be sent in headers or cookies
app.config["JWT_COOKIE_SECURE"] = False # Set to True in production with HTTPS
app.config["JWT_COOKIE_CSRF_PROTECT"] = True # Enable CSRF protection for cookies
jwt = JWTManager(app)

# --- Database Extension Initialization ---
db = SQLAlchemy(app)

# --- Standalone SQLAlchemy Engine for Enrollment Script ---
# This allows the enrollment functions to run outside of the Flask app context.
engine = create_engine(SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base() # We will use Flask-SQLAlchemy's db.Model

# --- Database Table Models ---
# These models are used by both Flask and the recognition script.

class User(db.Model):
    __tablename__ = 'Users'
    Id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Email = db.Column(db.String(100), unique=True, nullable=False)
    Password = db.Column(db.String(100), nullable=False)
    Role = db.Column(db.String(20), default='employee')
    employee_id = db.Column(db.Integer, db.ForeignKey('employees_info.id'), nullable=True)
    is_active = db.Column(db.Integer, default=1)
    security_code = db.Column(db.String(50))
    last_login = db.Column(db.String(50))
    created_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat())
    updated_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat(), onupdate=lambda: datetime.utcnow().isoformat())
    
    # Relationship to EmployeeInfo
    employee_info = relationship('EmployeeInfo', backref='user_account', foreign_keys=[employee_id])

class Department(db.Model):
    __tablename__ = 'departments'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

class EmployeeInfo(db.Model):
    __tablename__ = 'employees_info'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), unique=True)
    position = db.Column(db.String(50))
    department = db.Column(db.String(50))  # Keep for backward compatibility
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=True)
    hired_date = db.Column(db.String(50))
    photo = db.Column(db.String(255))
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat())
    access_code = db.Column(db.String(20), unique=True)
    # New fields for profile page
    phone = db.Column(db.String(50), nullable=True)
    address = db.Column(db.Text, nullable=True)
    bio = db.Column(db.Text, nullable=True)
    emergency_contact = db.Column(db.String(50), nullable=True)

class FaceEmbedding(db.Model):
    __tablename__ = 'face_embeddings'
    embedding_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    employee_info_id = db.Column(db.Integer, db.ForeignKey('employees_info.id'), nullable=False)
    embedding = db.Column(db.LargeBinary, nullable=False)
    image_source_filename = db.Column(db.String(255))

class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    employee_info_id = db.Column(db.Integer, db.ForeignKey('employees_info.id'), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    check_in = db.Column(db.String(50))
    check_out = db.Column(db.String(50))
    status = db.Column(db.String(20))
    is_late = db.Column(db.Integer, default=0)
    created_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat())

class SystemSetting(db.Model):
    __tablename__ = 'system_settings'
    id = db.Column(db.Integer, primary_key=True)
    work_start_time = db.Column(db.String(20), default='08:00:00')
    work_end_time = db.Column(db.String(20), default='17:00:00')
    max_late_minutes = db.Column(db.Integer, default=15)
    attendance_deadline = db.Column(db.String(20), default='09:30:00')  # Check-in deadline
    checkout_start_time = db.Column(db.String(20), default='22:00:00')  # Earliest checkout time
    language = db.Column(db.String(10), default='en')
    date_format = db.Column(db.String(20), default='MM/DD/YYYY')
    time_format = db.Column(db.String(10), default='12h')
    auto_checkout = db.Column(db.Integer, default=0)
    weekend_days = db.Column(db.String(20), default='5,6')  # Friday, Saturday as comma-separated
    email_notifications = db.Column(db.Integer, default=1)
    browser_notifications = db.Column(db.Integer, default=0)
    notification_types = db.Column(db.String(100), default='late,absent,system')  # Comma-separated
    session_timeout = db.Column(db.Integer, default=30)
    password_policy = db.Column(db.String(100), default='complexity,history')  # Comma-separated
    two_factor_auth = db.Column(db.Integer, default=0)
    api_access = db.Column(db.Integer, default=0)
    # حقل جديد لتخزين خيار سياسة الانصراف الصارمة
    # إذا كانت القيمة True، فلن يُسمح بالانصراف إلا بعد وقت انتهاء العمل الرسمي
    enforce_work_end_time = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat(), onupdate=lambda: datetime.utcnow().isoformat())

class Notification(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False)  # attendance, system, alert, reminder
    priority = db.Column(db.String(20), default='medium')  # high, medium, low
    read = db.Column(db.Integer, default=0)
    details = db.Column(db.Text)
    related_employee_id = db.Column(db.Integer, db.ForeignKey('employees_info.id'))
    created_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat())
    updated_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat(), onupdate=lambda: datetime.utcnow().isoformat())

class OnboardingSession(db.Model):
    __tablename__ = 'onboarding_sessions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    verification_code = db.Column(db.String(20))
    code_verified = db.Column(db.Integer, default=0)
    face_attempts = db.Column(db.Integer, default=0)
    face_captured = db.Column(db.Integer, default=0)
    photo_path = db.Column(db.String(255))
    employee_id = db.Column(db.Integer, db.ForeignKey('employees_info.id'))
    status = db.Column(db.String(20), default='started')  # started, face_failed, code_verified, completed
    expires_at = db.Column(db.String(50))
    created_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat())
    updated_at = db.Column(db.String(50), default=lambda: datetime.utcnow().isoformat(), onupdate=lambda: datetime.utcnow().isoformat())

# ==============================================================================
#                      FACE RECOGNITION & ENROLLMENT LOGIC
# ==============================================================================

# --- Recognition Tools Initialization (Matching new.ipynb exactly) ---
# Initialize MTCNN detector
try:
    detector = MTCNN()
    print("MTCNN detector initialized.")
except Exception as e:
    print(f"Error initializing MTCNN: {e}")

# Initialize FaceNet embedder
try:
    embedder = FaceNet()
    print("FaceNet embedder initialized.")
except Exception as e:
    print(f"Error initializing FaceNet: {e}")

# Platform Specific Camera Setup (from new.ipynb)
IS_PI = platform.system() == 'Linux' and 'raspberrypi' in platform.uname().node.lower()
USE_PICAMERA = False
picam2 = None
if IS_PI:
    try:
        from picamera2 import Picamera2
        USE_PICAMERA = True
        picam2 = Picamera2()
        picam2.start()
        print("✅ النظام هو Raspberry Pi - تم تفعيل Picamera2.")
    except ImportError:
        print("ℹ️ Picamera2 library not found on Pi, OpenCV will be used for camera if needed.")
        USE_PICAMERA = False
else:
    print(f"ℹ️ النظام الحالي هو: {platform.system()}, اسم الجهاز: {platform.uname().node}")
    if platform.system() == "Windows":
        print(f"ℹ️ النظام الحالي هو Windows, اسم الجهاز: {platform.uname().node}")
    print("✅ تم تفعيل OpenCV للكاميرا (وليس Picamera2).")

# --- Helper Functions ---
def initialize_database():
    """Creates all tables from the models if they don't exist."""
    print("\n--- Initializing Database ---")
    with app.app_context():
        db.create_all()
        print("✅ Database tables verified/created.")

        # Check if Attendance table has correct schema
        try:
            # Try to access the Attendance table to verify schema
            inspector = db.inspect(db.engine)
            attendance_columns = inspector.get_columns('attendance')
            column_names = [col['name'] for col in attendance_columns]

            if 'employee_info_id' in column_names:
                print("✅ Attendance table has correct employee_info_id column.")
            else:
                print("❌ Warning: Attendance table missing employee_info_id column.")

            if 'employee_id' in column_names:
                print("⚠️  Warning: Attendance table has old employee_id column (should be employee_info_id).")

        except Exception as e:
            print(f"⚠️  Could not verify Attendance table schema: {e}")

def initialize_database():
    """
    Creates all tables from the models if they don't exist and handles database migrations.
    This function is now more robust to ensure new columns are added and default settings exist.
    """
    print("\n--- Initializing Database ---")
    with app.app_context():
        db.create_all()
        print("✅ Database tables verified/created.")

        # Check if Attendance table has correct schema
        try:
            inspector = db.inspect(db.engine)
            attendance_columns = inspector.get_columns('attendance')
            column_names = [col['name'] for col in attendance_columns]

            if 'employee_info_id' in column_names:
                print("✅ Attendance table has correct employee_info_id column.")
            else:
                print("❌ Warning: Attendance table missing employee_info_id column.")

            if 'employee_id' in column_names:
                print("⚠️  Warning: Attendance table has old employee_id column (should be employee_info_id).")

        except Exception as e:
            print(f"⚠️  Could not verify Attendance table schema: {e}")

        # Handle database migration for new columns in SystemSetting
        try:
            inspector = db.inspect(db.engine)
            system_settings_columns = [c['name'] for c in inspector.get_columns('system_settings')]

            # إضافة عمود 'attendance_deadline' إذا لم يكن موجوداً
            if 'attendance_deadline' not in system_settings_columns:
                print("🔄 Migrating database: Adding 'attendance_deadline' column...")
                try:
                    db.session.execute(text("ALTER TABLE system_settings ADD COLUMN attendance_deadline VARCHAR(20) DEFAULT '09:30:00'"))
                    db.session.commit()
                    print("✅ Column 'attendance_deadline' added successfully.")
                except Exception as alter_e:
                    db.session.rollback()
                    if "duplicate column name" not in str(alter_e).lower():
                        print(f"❌ Failed to add column 'attendance_deadline': {alter_e}")

            # إضافة عمود 'checkout_start_time' إذا لم يكن موجوداً
            if 'checkout_start_time' not in system_settings_columns:
                print("🔄 Migrating database: Adding 'checkout_start_time' column...")
                try:
                    db.session.execute(text("ALTER TABLE system_settings ADD COLUMN checkout_start_time VARCHAR(20) DEFAULT '16:30:00'"))
                    db.session.commit()
                    print("✅ Column 'checkout_start_time' added successfully.")
                except Exception as alter_e:
                    db.session.rollback()
                    if "duplicate column name" not in str(alter_e).lower():
                        print(f"❌ Failed to add column 'checkout_start_time': {alter_e}")

            # [إصلاح نهائي] إضافة عمود 'enforce_work_end_time' إذا لم يكن موجوداً
            if 'enforce_work_end_time' not in system_settings_columns:
                print("🔄 Migrating database: Adding 'enforce_work_end_time' column...")
                try:
                    db.session.execute(text("ALTER TABLE system_settings ADD COLUMN enforce_work_end_time BOOLEAN DEFAULT 0"))
                    db.session.commit()
                    print("✅ Column 'enforce_work_end_time' added successfully.")
                except Exception as alter_e:
                    db.session.rollback()
                    if "duplicate column name" not in str(alter_e).lower():
                        print(f"❌ Failed to add column 'enforce_work_end_time': {alter_e}")

            print("✅ Database schema is up to date with all necessary columns.")

            # Handle database migration for new columns in EmployeeInfo
            try:
                employee_info_columns = [c['name'] for c in inspector.get_columns('employees_info')]
                new_columns = {
                    'phone': "ALTER TABLE employees_info ADD COLUMN phone VARCHAR(50)",
                    'address': "ALTER TABLE employees_info ADD COLUMN address TEXT",
                    'bio': "ALTER TABLE employees_info ADD COLUMN bio TEXT",
                    'emergency_contact': "ALTER TABLE employees_info ADD COLUMN emergency_contact VARCHAR(50)"
                }
                for col, statement in new_columns.items():
                    if col not in employee_info_columns:
                        print(f"🔄 Migrating database: Adding '{col}' column to employees_info...")
                        try:
                            db.session.execute(text(statement))
                            db.session.commit()
                            print(f"✅ Column '{col}' added successfully.")
                        except Exception as alter_e:
                            db.session.rollback()
                            if "duplicate column name" not in str(alter_e).lower():
                                print(f"❌ Failed to add column '{col}': {alter_e}")
            except Exception as emp_mig_e:
                print(f"⚠️ Could not perform migration for employees_info table: {emp_mig_e}")


        except Exception as e:
            db.session.rollback()
            import traceback
            print(f"❌ Critical Error during database migration: {e}")
            print(traceback.format_exc())

        # Add default settings if they don't exist or are incomplete
        try:
            settings = SystemSetting.query.get(1)
            if not settings:
                print("Creating full default system settings...")
                default_settings = SystemSetting(
                    id=1,
                    work_start_time='08:00:00',
                    work_end_time='17:00:00',
                    max_late_minutes=15,
                    attendance_deadline='09:30:00',
                    checkout_start_time='16:30:00',
                    language='en',
                    date_format='MM/DD/YYYY',
                    time_format='12h',
                    auto_checkout=0,
                    weekend_days='5,6',
                    email_notifications=1,
                    browser_notifications=0,
                    notification_types='late,absent,system',
                    session_timeout=30,
                    password_policy='complexity,history',
                    two_factor_auth=0,
                    api_access=0,
                    enforce_work_end_time=False
                )
                db.session.add(default_settings)
                db.session.commit()
                print("✅ Full default system settings added.")
            else:
                # Ensure existing settings have default values for new fields if they are None
                # This handles cases where the column was added but the existing row has NULL
                changed = False
                if settings.attendance_deadline is None:
                    settings.attendance_deadline = '09:30:00'
                    changed = True
                if settings.checkout_start_time is None:
                    settings.checkout_start_time = '16:30:00'
                    changed = True
                if settings.enforce_work_end_time is None:
                    settings.enforce_work_end_time = False
                    changed = True
                if changed:
                    db.session.commit()
                    print("✅ Updated existing system settings with default values for new fields.")
                else:
                    print("✅ System settings already exist and are up to date.")
        except Exception as e:
            db.session.rollback()
            import traceback
            print(f"❌ Critical Error during system settings initialization: {e}")
            print(traceback.format_exc())
            # Fallback: attempt to create minimal settings if all else fails
            try:
                db.session.execute(text("INSERT OR IGNORE INTO system_settings (id, attendance_deadline, checkout_start_time, enforce_work_end_time) VALUES (1, '09:30:00', '16:30:00', 0)"))
                db.session.commit()
                print("✅ Minimal system settings created via SQL as fallback.")
            except Exception as sql_e:
                db.session.rollback()
                print(f"❌ Could not create minimal settings even as fallback: {sql_e}")

def reset_database():
    """Reset database by dropping and recreating all tables."""
    print("\n--- Resetting Database ---")
    with app.app_context():
        try:
            print("🗑️  Dropping all tables...")
            db.drop_all()
            print("✅ All tables dropped.")

            print("🔨 Creating all tables...")
            db.create_all()
            print("✅ All tables created.")

            # Verify Attendance table schema
            inspector = db.inspect(db.engine)
            attendance_columns = inspector.get_columns('attendance')
            column_names = [col['name'] for col in attendance_columns]

            print(f"📋 Attendance table columns: {column_names}")

            if 'employee_info_id' in column_names:
                print("✅ Attendance table has correct employee_info_id column.")
            else:
                print("❌ ERROR: Attendance table missing employee_info_id column!")

        except Exception as e:
            print(f"❌ Database reset failed: {e}")
            raise

        # Create default departments if none exist
        if Department.query.count() == 0:
            print("Creating default departments...")
            default_departments = [
                'Human Resources',
                'Information Technology',
                'Finance',
                'Marketing',
                'Operations',
                'Sales',
                'Customer Service',
                'Research & Development'
            ]

            for dept_name in default_departments:
                department = Department(name=dept_name)
                db.session.add(department)

            db.session.commit()
            print("✅ Default departments created successfully.")
        else:
            print("✅ Departments already exist in database.")

        # Add default settings if they don't exist
        try:
            # This part is now handled by the main initialize_database, but kept for reset consistency
            if not SystemSetting.query.get(1):
                print("Creating full default system settings after reset...")
                default_settings = SystemSetting(
                    id=1,
                    work_start_time='08:00:00',
                    work_end_time='17:00:00',
                    max_late_minutes=15,
                    attendance_deadline='09:30:00',
                    checkout_start_time='16:30:00',
                    language='en',
                    date_format='MM/DD/YYYY',
                    time_format='12h',
                    auto_checkout=0,
                    weekend_days='5,6',
                    email_notifications=1,
                    browser_notifications=0,
                    notification_types='late,absent,system',
                    session_timeout=30,
                    password_policy='complexity,history',
                    two_factor_auth=0,
                    api_access=0,
                    enforce_work_end_time=False
                )
                db.session.add(default_settings)
                db.session.commit()
                print("✅ Full default system settings added after reset.")
            else:
                print("✅ System settings already exist after reset.")
        except Exception as e:
            db.session.rollback()
            import traceback
            print(f"❌ Critical Error during system settings initialization after reset: {e}")
            print(traceback.format_exc())

def generate_unique_code(session, length=6):
    """Generates a unique access code for an employee."""
    characters = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choices(characters, k=length))
        if not session.query(EmployeeInfo).filter_by(access_code=code).first():
            return code

def load_and_preprocess_image(image_input, target_size=TARGET_SIZE, is_path=True, is_precropped=False):
    """Load and preprocess image - exact copy from new.ipynb"""
    image = None
    img_identifier = ""
    try:
        if is_path:
            img_identifier = os.path.basename(image_input)
            image = cv2.imread(image_input)
            if image is None:
                print(f"⚠️ [خطأ] الصورة غير موجودة أو تالفة: {image_input}")
                return None
        else:
            img_identifier = "live_frame"
            image = image_input

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_to_resize = None
        if is_precropped:
            face_to_resize = image_rgb
        else:
            faces = detector.detect_faces(image_rgb)
            if not faces:
                print(f"⚠️ [خطأ] لا يوجد وجه في الصورة: {img_identifier}")
                return None
            x, y, w, h = faces[0]['box']
            x, y = abs(x), abs(y)
            face_to_resize = image_rgb[y:y+h, x:x+w]

        if face_to_resize.size == 0:
            print(f"⚠️ [خطأ] اقتصاص الوجه نتج عنه صورة فارغة: {img_identifier}")
            return None
        resized_face = cv2.resize(face_to_resize, target_size)
        return resized_face
    except Exception as e:
        print(f"⚠️ [خطأ] أثناء معالجة الصورة {img_identifier if is_path else 'live_frame'}: {e}")
        return None

def get_embedding(image_face_processed):
    """Get embedding - exact copy from new.ipynb"""
    if image_face_processed is None:
        return None
    try:
        raw_embeddings_output = embedder.embeddings([image_face_processed])
        embedding_array = None

        if isinstance(raw_embeddings_output, list):
            if len(raw_embeddings_output) > 0:
                embedding_array = raw_embeddings_output[0]
            else:
                print("    ⚠️ [خطأ] embedder.embeddings() returned an empty list.")
                return None
        elif isinstance(raw_embeddings_output, np.ndarray):
            embedding_array = raw_embeddings_output
        else:
            print(f"    ⚠️ [خطأ] Unexpected output type from embedder.embeddings(): {type(raw_embeddings_output)}")
            return None

        if not isinstance(embedding_array, np.ndarray):
            print(f"    ⚠️ [خطأ] Processed embedding is not a NumPy array after extraction. Type: {type(embedding_array)}")
            return None

        # Ensure it's a 1D vector or can be made one
        if embedding_array.ndim == 2 and embedding_array.shape[0] == 1: # Shape like (1, 512)
            embedding_array = embedding_array.flatten()
        elif embedding_array.ndim != 1: # Not 1D and not easily flattenable from (1,N)
            print(f"    ⚠️ [خطأ] Extracted embedding is not a 1D vector after potential flatten. Dimensions: {embedding_array.ndim}, Shape: {embedding_array.shape}")
            return None

        norm_val = np.linalg.norm(embedding_array)
        if norm_val == 0:
            print("    ⚠️ [تحذير] Embedding norm is zero. Returning unnormalized.")
            return embedding_array

        embedding_normalized = embedding_array / norm_val
        return embedding_normalized

    except Exception as e:
        print(f"    ⚠️ [General Error] during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def insert_employee_and_embedding(session, employee_name, embedding, image_filename):
    """Finds or creates an employee, then adds a new face embedding."""
    if embedding is None: return False
    
    employee = session.query(EmployeeInfo).filter_by(full_name=employee_name).first()
    if not employee:
        access_code = generate_unique_code(session)
        employee = EmployeeInfo(full_name=employee_name, access_code=access_code)
        session.add(employee)
        session.flush() # Get the new employee ID
        print(f"✅ Added new employee '{employee_name}' with ID {employee.id}.")

    embedding_blob = embedding.astype(np.float32).tobytes()
    new_embedding = FaceEmbedding(
        employee_info_id=employee.id,
        embedding=embedding_blob,
        image_source_filename=image_filename
    )
    session.add(new_embedding)
    print(f"✅ Stored new embedding for: {employee_name} (from {image_filename}).")
    return True

def enroll_employees_from_folder(training_folder_path):
    """Processes a directory of employee photos and enrolls them in the database."""
    print(f"\n--- Starting Employee Enrollment from: {training_folder_path} ---")
    if not os.path.isdir(training_folder_path):
        print(f"⚠️ [Error] Training folder '{training_folder_path}' not found.")
        return
    
    session = SessionLocal()
    try:
        for employee_name in os.listdir(training_folder_path):
            employee_path = os.path.join(training_folder_path, employee_name)
            if os.path.isdir(employee_path):
                print(f"\nProcessing employee: {employee_name}")
                for image_filename in os.listdir(employee_path):
                    if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(employee_path, image_filename)
                        face_image = load_and_preprocess_image(image_path)
                        if face_image is not None:
                            embedding = get_embedding(face_image)
                            if embedding is not None:
                                insert_employee_and_embedding(session, employee_name, embedding, image_filename)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"⚠️ [Error] Enrollment failed. Changes rolled back: {e}")
    finally:
        session.close()
    print("\n--- Employee Enrollment Finished ---")

def load_all_known_embeddings_from_db():
    """Load all known embeddings from database - exact copy from new.ipynb"""
    known_embeddings_map = {}
    try:
        # Get all face embeddings with employee info
        all_face_embeddings = db.session.query(FaceEmbedding, EmployeeInfo).join(
            EmployeeInfo, FaceEmbedding.employee_info_id == EmployeeInfo.id
        ).filter(EmployeeInfo.status == 'active').all()

        for face_embedding, employee_info in all_face_embeddings:
            name = employee_info.full_name
            emp_id = employee_info.id
            fixed_code = employee_info.access_code

            # Convert stored embedding back to numpy array
            embedding_vector = np.frombuffer(face_embedding.embedding, dtype=np.float32)
            norm_val = np.linalg.norm(embedding_vector)

            if name not in known_embeddings_map:
                known_embeddings_map[name] = {
                    'employee_id': emp_id,
                    'fixed_code': fixed_code,
                    'embeddings': []
                }

            if norm_val > 0:
                known_embeddings_map[name]['embeddings'].append(embedding_vector / norm_val)
            else:
                known_embeddings_map[name]['embeddings'].append(embedding_vector)

        print(f"✅ تم تحميل بيانات لـ {len(known_embeddings_map)} موظف، بإجمالي {sum(len(data['embeddings']) for data in known_embeddings_map.values())} تضمين.")
        return known_embeddings_map

    except Exception as e:
        print(f"⚠️ [خطأ] تحميل التضمينات: {e}")
        return {}

def log_failed_attempt(employee_name_attempted, image_path_or_data, attempt_number, reason):
    """Log failed attempt - simplified version from new.ipynb"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        img_log_path = image_path_or_data if isinstance(image_path_or_data, str) else "live_frame_data"
        print(f"    ⚠️ [فشل المحاولة {attempt_number}] {employee_name_attempted}: {reason}")
        # Could add database logging here if needed
    except Exception as e:
        print(f"⚠️ [خطأ] تسجيل محاولة فاشلة: {e}")

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity - exact copy from new.ipynb"""
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    if norm_emb1 == 0 or norm_emb2 == 0:
        return -1.0
    return np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)

def generate_face_embedding(face_image):
    """
    Generate face embedding from a face image using the exact same pipeline as
    load_and_preprocess_image + get_embedding functions.

    Args:
        face_image: numpy array of face image (RGB format from MTCNN detection)

    Returns:
        numpy array: Face embedding vector, or None if processing fails
    """
    try:
        # Ensure face_image is in the right format
        if face_image is None or face_image.size == 0:
            print("⚠️ [خطأ] Empty or invalid face image")
            return None

        # Step 1: Use the same preprocessing as load_and_preprocess_image
        # The face_image is already cropped from MTCNN, so we treat it as precropped
        face_to_resize = face_image

        if face_to_resize.size == 0:
            print(f"⚠️ [خطأ] Face image is empty after processing")
            return None

        # Resize using the exact same method as load_and_preprocess_image
        resized_face = cv2.resize(face_to_resize, TARGET_SIZE)

        # Step 2: Use the exact same embedding generation as get_embedding
        raw_embeddings_output = embedder.embeddings([resized_face])
        embedding_array = None

        if isinstance(raw_embeddings_output, list):
            if len(raw_embeddings_output) > 0:
                embedding_array = raw_embeddings_output[0]
            else:
                print("    ⚠️ [خطأ] embedder.embeddings() returned an empty list.")
                return None
        elif isinstance(raw_embeddings_output, np.ndarray):
            embedding_array = raw_embeddings_output
        else:
            print(f"    ⚠️ [خطأ] Unexpected output type from embedder.embeddings(): {type(raw_embeddings_output)}")
            return None

        if not isinstance(embedding_array, np.ndarray):
            print(f"    ⚠️ [خطأ] Processed embedding is not a NumPy array after extraction. Type: {type(embedding_array)}")
            return None

        # Ensure it's a 1D vector or can be made one - exact same logic as get_embedding
        if embedding_array.ndim == 2 and embedding_array.shape[0] == 1: # Shape like (1, 512)
            embedding_array = embedding_array.flatten()
        elif embedding_array.ndim != 1: # Not 1D and not easily flattenable from (1,N)
            print(f"    ⚠️ [خطأ] Extracted embedding is not a 1D vector after potential flatten. Dimensions: {embedding_array.ndim}, Shape: {embedding_array.shape}")
            return None

        # Normalize exactly as in get_embedding
        norm_val = np.linalg.norm(embedding_array)
        if norm_val == 0:
            print("    ⚠️ [تحذير] Embedding norm is zero. Returning unnormalized.")
            return embedding_array

        embedding_normalized = embedding_array / norm_val
        print(f"✅ Generated embedding with shape: {embedding_normalized.shape}, norm: {np.linalg.norm(embedding_normalized):.4f}")
        return embedding_normalized

    except Exception as e:
        print(f"⚠️ [خطأ] Error generating face embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

def recognize_face_from_database(input_embedding, session):
    """
    Recognize face from database - matching new.ipynb approach
    """
    if input_embedding is None:
        print("⚠️ [خطأ] Input embedding is None")
        return None, -1.0

    try:
        # Get all face embeddings from database
        all_face_embeddings = session.query(FaceEmbedding, EmployeeInfo).join(
            EmployeeInfo, FaceEmbedding.employee_info_id == EmployeeInfo.id
        ).filter(EmployeeInfo.status == 'active').all()

        if not all_face_embeddings:
            print("⚠️ [خطأ] No face embeddings found in database")
            return None, -1.0

        best_match = None
        highest_similarity = -1.0

        print(f"🔍 Comparing against {len(all_face_embeddings)} stored faces...")

        for face_embedding, employee_info in all_face_embeddings:
            try:
                # Convert stored embedding back to numpy array
                stored_embedding = np.frombuffer(face_embedding.embedding, dtype=np.float32)

                # Both embeddings should already be normalized from the same pipeline
                # Just calculate similarity directly using the cosine_similarity function
                similarity = cosine_similarity(input_embedding, stored_embedding)

                print(f"    👤 {employee_info.full_name}: similarity = {similarity:.4f}")

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = employee_info

            except Exception as e:
                print(f"⚠️ [خطأ] Error processing embedding for {employee_info.full_name}: {e}")
                continue

        print(f"🎯 Best match: {best_match.full_name if best_match else 'None'} with similarity {highest_similarity:.4f}")
        print(f"🎯 Threshold: {THRESHOLD}")

        if highest_similarity >= THRESHOLD:
            print(f"✅ Recognition successful: {best_match.full_name}")
            return best_match, highest_similarity
        else:
            print(f"❌ Recognition failed: similarity {highest_similarity:.4f} < threshold {THRESHOLD}")
            return None, highest_similarity

    except Exception as e:
        print(f"⚠️ [خطأ] Error in face recognition: {e}")
        import traceback
        traceback.print_exc()
        return None, -1.0

def test_recognition_from_folder(test_folder_path, known_db_embeddings):
    """Test recognition from folder - exact copy from new.ipynb"""
    print(f"\n--- جاري مقارنة الموظفين من مجلد الاختبار: {test_folder_path} ---")
    if not os.path.isdir(test_folder_path):
        print(f"⚠️ [خطأ] مجلد الاختبار '{test_folder_path}' غير موجود.")
        return {}

    summary_results = {}
    for person_folder_name in os.listdir(test_folder_path):
        current_person_folder_path = os.path.join(test_folder_path, person_folder_name)
        if not os.path.isdir(current_person_folder_path):
            continue

        print(f"\n➡️  جاري اختبار صور الموظف: {person_folder_name}")
        image_files = sorted([f for f in os.listdir(current_person_folder_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not image_files:
            print(f"  لا توجد صور في مجلد {person_folder_name}.")
            summary_results[person_folder_name] = ("No Test Images", -1.0, "N/A", "No Test Images Found")
            continue

        person_recognized_by_face = False
        best_score_for_this_person = -1.0
        name_recognized_for_this_person = "غير معروف"
        code_of_recognized_person = "N/A"
        status_for_this_person = "Failed (All Face Attempts)"

        for attempt_idx, img_filename in enumerate(image_files[:MAX_BATCH_TEST_ATTEMPTS_PER_PERSON], 1):
            img_full_path = os.path.join(current_person_folder_path, img_filename)
            print(f"    محاولة {attempt_idx}/{MAX_BATCH_TEST_ATTEMPTS_PER_PERSON}: معالجة '{img_filename}'")

            is_test_img_precropped = img_filename.startswith('face_')
            processed_img = load_and_preprocess_image(img_full_path, is_path=True, is_precropped=is_test_img_precropped)

            if processed_img is None:
                log_failed_attempt(person_folder_name, img_full_path, attempt_idx, "فشل المعالجة/لا يوجد وجه")
                continue

            current_emb = get_embedding(processed_img)
            if current_emb is None:
                log_failed_attempt(person_folder_name, img_full_path, attempt_idx, "فشل استخراج التضمين")
                continue

            temp_best_match = None
            temp_highest_sim = -1.0
            for db_name, db_data in known_db_embeddings.items():
                for db_emb in db_data['embeddings']:
                    sim = cosine_similarity(current_emb, db_emb)
                    if sim > temp_highest_sim:
                        temp_highest_sim = sim
                        temp_best_match = db_name

            if temp_best_match and temp_highest_sim > best_score_for_this_person:
                 best_score_for_this_person = temp_highest_sim
                 name_recognized_for_this_person = temp_best_match
                 if temp_best_match in known_db_embeddings:
                     code_of_recognized_person = known_db_embeddings[temp_best_match]['fixed_code']

            if temp_best_match and temp_highest_sim >= THRESHOLD:
                print(f"    ✅ تم التعرف على: {temp_best_match} (التشابه: {temp_highest_sim:.4f})")
                person_recognized_by_face = True
                status_for_this_person = "Face Recognized"
                break
            else:
                log_failed_attempt(person_folder_name, img_full_path, attempt_idx,
                                 f"تشابه منخفض ({temp_highest_sim:.4f} مع {temp_best_match if temp_best_match else 'لا أحد'})")

        summary_results[person_folder_name] = (name_recognized_for_this_person, best_score_for_this_person,
                                             code_of_recognized_person, status_for_this_person)
    return summary_results

def record_attendance(employee_id):
    """Record check-in for an employee."""
    try:
        today = date.today().isoformat()
        now = datetime.now()
        current_time_str = now.strftime('%H:%M:%S')
        current_time_obj = now.time()

        # Check if employee already checked in today
        existing_record = Attendance.query.filter_by(
            employee_info_id=employee_id,
            date=today
        ).first()

        if existing_record and existing_record.check_in:
            return {"error": "Employee has already checked in today"}, 400

        # --- [تعديل] منطق التحقق من التأخير ---
        # الخطوة 1: جلب إعدادات النظام المتعلقة بأوقات العمل
        settings = SystemSetting.query.get(1)
        is_late_check = 0
        status = 'Present'

        if settings:
            try:
                
                # الخطوة 2: تحويل وقت بدء العمل الرسمي (من نص) إلى كائن وقت للمقارنة
                # [إصلاح] التعامل مع تنسيقات الوقت المختلفة (مع أو بدون الثواني)
                work_start_time_str = settings.work_start_time
                if len(work_start_time_str.split(':')) == 2:  # إذا كان التنسيق HH:MM فقط
                    work_start_time_str += ':00'  # إضافة الثواني
                work_start_time_obj = datetime.strptime(work_start_time_str, '%H:%M:%S').time()

                # الخطوة 3: حساب آخر وقت مسموح به للحضور (وقت البدء + فترة السماح)
                # نستخدم timedelta لإضافة الدقائق المسموح بها إلى تاريخ اليوم مع وقت بدء العمل
                grace_period_end = (datetime.combine(date.today(), work_start_time_obj) + 
                                    timedelta(minutes=settings.max_late_minutes))
                
                grace_period_end_time_obj = grace_period_end.time()

                # الخطوة 4: مقارنة وقت الحضور الفعلي مع نهاية فترة السماح
                # إذا كان وقت الحضور الحالي أكبر من الوقت المسموح به، يتم تسجيل الموظف كمتأخر
                if current_time_obj > grace_period_end_time_obj:
                    is_late_check = 1  # 1 تعني "نعم، متأخر"
                    status = 'Late'    # تحديث الحالة إلى "متأخر"
                    
            except (ValueError, TypeError) as e:
                # في حال وجود خطأ في صيغة الوقت في الإعدادات، يتم تجاهل التحقق من التأخير
                print(f"⚠️  تحذير: لا يمكن التحقق من التأخير بسبب خطأ في الإعدادات: {e}")

        # Create new check-in record
        new_attendance = Attendance(
            employee_info_id=employee_id,
            date=today,
            check_in=current_time_str,
            status=status, # استخدام الحالة المحدثة
            is_late=is_late_check # إضافة قيمة التأخير
        )
        db.session.add(new_attendance)
        db.session.commit()
        return {"message": "Check-in recorded successfully", "action": "checkin"}, 200

    except Exception as e:
        db.session.rollback()
        return {"error": f"Failed to record attendance: {str(e)}"}, 500

def normalize_time_format(time_str):
    """
    تحويل صيغة الوقت إلى HH:MM:SS القياسية
    يتعامل مع صيغ HH:MM و HH:MM:SS ويضمن صحة القيم
    """
    if not time_str:
        return '16:30:00'  # القيمة الافتراضية لأبكر وقت انصراف
    
    # إزالة أي مسافات زائدة
    time_str = time_str.strip()
    
    # التحقق من وجود فاصلتين في النص
    if ':' not in time_str:
        print(f"⚠️ تحذير: صيغة وقت غير صالحة '{time_str}'، استخدام القيمة الافتراضية")
        return '16:30:00'
    
    parts = time_str.split(':')
    
    if len(parts) == 2:  # صيغة HH:MM (من input type="time" في HTML)
        try:
            hour = int(parts[0])
            minute = int(parts[1])
            
            # التحقق من صحة القيم (ساعات 0-23، دقائق 0-59)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                # تحويل إلى صيغة HH:MM:SS
                return f"{hour:02d}:{minute:02d}:00"
            else:
                print(f"⚠️ تحذير: قيم وقت خارج النطاق '{time_str}'، استخدام القيمة الافتراضية")
                return '16:30:00'
        except ValueError:
            print(f"⚠️ تحذير: لا يمكن تحليل القيم العددية في '{time_str}'، استخدام القيمة الافتراضية")
            return '16:30:00'
    
    elif len(parts) == 3:  # صيغة HH:MM:SS (صيغة قاعدة البيانات)
        try:
            hour, minute, second = map(int, parts)
            # التحقق من صحة جميع القيم
            if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                # التأكد من التنسيق الصحيح (02d للأرقام أقل من 10)
                return f"{hour:02d}:{minute:02d}:{second:02d}"
            else:
                print(f"⚠️ تحذير: قيم خارج النطاق في '{time_str}'، استخدام القيمة الافتراضية")
                return '16:30:00'
        except ValueError:
            print(f"⚠️ تحذير: خطأ في تحليل صيغة HH:MM:SS '{time_str}'، استخدام القيمة الافتراضية")
            return '16:30:00'
    
    else:
        # أي صيغة أخرى غير مدعومة
        print(f"⚠️ تحذير: صيغة وقت غير مدعومة '{time_str}'، استخدام القيمة الافتراضية")
        return '16:30:00'

def safe_parse_time(time_str, default_time='16:30:00'):
    """
    تحليل وقت بأمان مع قيمة افتراضية
    يحاول تحليل HH:MM:SS أولاً، ثم HH:MM، وأخيراً يستخدم القيمة الافتراضية
    """
    if not time_str:
        return default_time
    
    # المحاولة الأولى: HH:MM:SS
    try:
        parsed = datetime.strptime(time_str, '%H:%M:%S')
        return time_str  # القيمة صحيحة بالفعل
    except ValueError:
        # المحاولة الثانية: HH:MM
        try:
            parsed = datetime.strptime(time_str, '%H:%M')
            # تحويل إلى HH:MM:SS
            return parsed.strftime('%H:%M') + ':00'
        except ValueError:
            # فشل كلا المحاولتين، استخدام القيمة الافتراضية
            print(f"⚠️ تحذير: صيغة الوقت غير صالحة '{time_str}'، استخدام الافتراضي {default_time}")
            return default_time

def safe_get_time_setting(time_field, default_time_str='16:30:00'):
    """
    الحصول على إعداد وقت من قاعدة البيانات بأمان
    يستخدم safe_parse_time للتعامل مع الصيغ غير الصالحة
    """
    settings = SystemSetting.query.get(1)
    if not settings or not getattr(settings, time_field):
        print(f"ℹ️ إعداد {time_field} غير موجود، استخدام القيمة الافتراضية {default_time_str}")
        return default_time_str
    
    time_value = getattr(settings, time_field)
    normalized_time = safe_parse_time(time_value, default_time_str)
    
    if normalized_time != time_value:
        print(f"ℹ️ تم تحويل {time_field} من '{time_value}' إلى '{normalized_time}'")
    
    return normalized_time

def record_checkout(employee_id):
    """Record check-out for an employee."""
    try:
        today = date.today().isoformat()
        current_time = datetime.now().strftime('%H:%M:%S')

        # Find existing attendance record for today
        existing_record = Attendance.query.filter_by(
            employee_info_id=employee_id,
            date=today
        ).first()

        if not existing_record or not existing_record.check_in:
            return {"error": "No check-in record found for today. Please check in first."}, 400

        if existing_record.check_out:
            return {"error": "Employee has already checked out today"}, 400

        # Update check-out time
        existing_record.check_out = current_time
        db.session.commit()
        return {"message": "Check-out recorded successfully", "action": "checkout"}, 200

    except Exception as e:
        db.session.rollback()
        return {"error": f"Failed to record checkout: {str(e)}"}, 500


# ==============================================================================
#                           إدارة حسابات المستخدمين
# ==============================================================================

@app.route('/api/users', methods=['GET'])
@admin_required
def get_users():
    """
    [إضافة جديدة] - جلب قائمة بجميع حسابات المستخدمين مع الموظفين المرتبطين بهم.
    """
    try:
        # استعلام لجلب المستخدمين مع اسم الموظف المرتبط إن وجد
        users_with_employees = db.session.query(
            User, 
            EmployeeInfo.full_name
        ).outerjoin(
            EmployeeInfo, User.employee_id == EmployeeInfo.id
        ).order_by(User.Id).all()
        
        result = []
        for user, employee_name in users_with_employees:
            result.append({
                "id": user.Id,
                "email": user.Email,
                "role": user.Role,
                "employee_id": user.employee_id,
                "employee_name": employee_name or "Unrelated", # عرض "غير مرتبط" إذا لم يكن هناك موظف
                "last_login": user.last_login,
                "created_at": user.created_at
            })
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/employees/unlinked', methods=['GET'])
@admin_required
def get_unlinked_employees():
    """
    [إضافة جديدة] - جلب قائمة بالموظفين الذين ليس لديهم حساب مستخدم مرتبط بهم بعد.
    هذه الواجهة مهمة جداً لملء القائمة المنسدلة في نموذج إضافة حساب جديد.
    """
    try:
        # ايجاد جميع IDs الموظفين المرتبطين بالفعل بحسابات
        linked_employee_ids = db.session.query(User.employee_id).filter(User.employee_id.isnot(None)).all()
        # تحويل النتيجة إلى قائمة بسيطة من الـ IDs
        linked_ids = [item[0] for item in linked_employee_ids]

        # جلب جميع الموظفين الذين ليسوا ضمن قائمة الموظفين المرتبطين
        unlinked_employees = EmployeeInfo.query.filter(EmployeeInfo.id.notin_(linked_ids)).order_by(EmployeeInfo.full_name).all()
        
        result = [{"id": emp.id, "full_name": emp.full_name, "email": emp.email} for emp in unlinked_employees]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users', methods=['POST'])
@admin_required
def create_user():
    """
    [إضافة جديدة] - إنشاء حساب مستخدم جديد وربطه بموظف إذا تم تحديده.
    """
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')
        employee_id = data.get('employee_id')

        # التحقق من الحقول الإلزامية
        if not email or not password or not role:
            return jsonify({"error": "Email, password, and role are required"}), 400
        
        # التحقق من عدم وجود مستخدم بنفس البريد الإلكتروني
        if User.query.filter_by(Email=email).first():
            return jsonify({"error": "A user with this email already exists"}), 409

        # التحقق من أن الموظف (إذا تم تحديده) غير مرتبط بحساب آخر
        if employee_id:
            if User.query.filter_by(employee_id=employee_id).first():
                return jsonify({"error": "This employee is already linked to another user account"}), 409

        # تشفير كلمة المرور
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        new_user = User(
            Email=email,
            Password=hashed_password,
            Role=role,
            employee_id=employee_id
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({"success": True, "message": "User account created successfully"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    """
    [إضافة جديدة] - تحديث بيانات حساب مستخدم (مثل الدور أو الموظف المرتبط).
    """
    try:
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        
        if 'role' in data:
            user.Role = data['role']
        
        # السماح بربط الموظف أو فك ارتباطه
        if 'employee_id' in data:
            employee_id = data['employee_id']
            if employee_id:
                # التأكد أن الموظف غير مرتبط بحساب آخر (غير الحساب الحالي)
                existing_link = User.query.filter(User.employee_id == employee_id, User.Id != user_id).first()
                if existing_link:
                    return jsonify({"error": "This employee is already linked to another user account"}), 409
            user.employee_id = employee_id

        db.session.commit()
        return jsonify({"success": True, "message": "User account updated successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """
    [إضافة جديدة] - حذف حساب مستخدم.
    """
    try:
        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        return jsonify({"success": True, "message": "User account deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<int:user_id>/reset-password', methods=['POST'])
@admin_required
def reset_user_password(user_id):
    """
    [إضافة جديدة] - إعادة تعيين كلمة مرور مستخدم محدد.
    تقوم بتوليد كلمة مرور عشوائية جديدة، وتحديثها في قاعدة البيانات، ثم إرجاعها للمدير.
    """
    try:
        user = User.query.get_or_404(user_id)

        # توليد كلمة مرور عشوائية قوية
        new_password = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=12))
        
        # تشفير كلمة المرور الجديدة
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        
        # تحديث كلمة المرور في قاعدة البيانات
        user.Password = hashed_password
        db.session.commit()
        
        return jsonify({"success": True, "message": "Password reset successfully", "new_password": new_password}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# ==============================================================================
#                               FLASK API ENDPOINTS
# ==============================================================================

# --- Route لتغيير اللغة ---
@app.route('/change_language/<lang_code>')
def change_language(lang_code):
    # التأكد من أن اللغة المطلوبة مدعومة
    if lang_code in app.config['LANGUAGES']:
        # تخزين اللغة المختارة في جلسة المستخدم
        session['locale'] = lang_code
    # الرجوع إلى الصفحة السابقة التي كان فيها المستخدم
    # إذا لم تكن متوفرة، يتم التوجيه إلى لوحة التحكم الرئيسية
    return redirect(request.referrer or url_for('dashboard_redirect'))


# مسار اختبار الترجمة
@app.route('/test_translation')
def test_translation():
    from flask_babel import gettext as _
    test_text = _('System Settings')
    return f"English: System Settings, Arabic: {test_text}, Current locale: {get_locale()}"


# Debug endpoint to check i18n behavior server-side
@app.route('/_debug_i18n')
def _debug_i18n():
    # Return current session locale, get_locale(), and a translated sample
    current_session_locale = session.get('locale')
    locale_from_func = None
    try:
        locale_from_func = get_locale()
    except Exception:
        locale_from_func = 'error'

    # Sample string used in templates
    sample = gettext('System Settings')
    return jsonify({
        'session_locale': current_session_locale,
        'get_locale': locale_from_func,
        'sample_translation': sample
    })


@app.route('/api/employees', methods=['GET'])
@admin_required
def get_employees():
    """Returns a list of all employees."""
    try:
        employees = EmployeeInfo.query.all()
        result = []
        for emp in employees:
            # Check if employee has face embeddings
            face_embedding_count = FaceEmbedding.query.filter_by(employee_info_id=emp.id).count()

            result.append({
                "id": emp.id,
                "name": emp.full_name,
                "empid": f"EMP{emp.id:04d}",
                "department": emp.department or "N/A",
                "jobTitle": emp.position or "N/A",
                "hireDate": emp.hired_date or "N/A",
                "email": emp.email or "N/A",
                "status": emp.status or "active",
                "access_code": emp.access_code or "N/A",
                "face_registered": face_embedding_count > 0,
                "face_embeddings_count": face_embedding_count,
                "profile_pic": emp.photo, # Add profile picture filename
                "avatar_url": f"/static/profile_pics/{emp.photo}" if emp.photo else "/static/default-avatar.svg"
            })
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/reports/weekly', methods=['GET'])
@admin_required
def get_weekly_report():
    """
    Analyzes attendance data for the past week and returns a structured report.
    """
    try:
        today = date.today()
        start_of_week = today - timedelta(days=today.weekday()) # Monday
        end_of_week = start_of_week + timedelta(days=6) # Sunday, to get all records for the week

        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        # Get all attendance records for the entire week to process them in memory
        all_week_records = db.session.query(Attendance, EmployeeInfo.full_name).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).filter(
            Attendance.date.between(start_of_week.isoformat(), end_of_week.isoformat())
        ).all()

        # --- [NEW LOGIC] Process data day by day for the work week (Mon-Fri) ---
        final_summary = []
        for i in range(7):
            current_day_date = start_of_week + timedelta(days=i)
            day_name = current_day_date.strftime('%A')
            date_iso = current_day_date.isoformat()

            # Filter records for the current day from the list we already fetched
            records_for_day = [r for r in all_week_records if r.Attendance.date == date_iso]

            present_count = len([r for r in records_for_day if r.Attendance.status in ['Present', 'Late']])
            
            # Count explicit 'Absent' records for the day
            absent_count = len([r for r in records_for_day if r.Attendance.status == 'Absent'])

            # Hybrid logic: If no explicit absent records, calculate implicitly
            if absent_count == 0:
                # This is a fallback for days before the scheduler was implemented or if it fails
                implicit_absent = total_employees - present_count
                absent_count = max(0, implicit_absent)

            # Calculate average check-in time for the day
            avg_check_in_str = 'N/A'
            check_in_times = [datetime.strptime(r.Attendance.check_in, '%H:%M:%S').time() for r in records_for_day if r.Attendance.check_in]
            if check_in_times:
                total_seconds = sum(t.hour * 3600 + t.minute * 60 + t.second for t in check_in_times)
                avg_seconds = total_seconds / len(check_in_times)
                avg_time_obj = (datetime.min + timedelta(seconds=avg_seconds)).time()
                avg_check_in_str = avg_time_obj.strftime('%I:%M %p')

            final_summary.append({
                'day': day_name,
                'present': present_count,
                'absent': absent_count,
                'avg_check_in': avg_check_in_str
            })

        # --- Punctuality calculation (remains largely the same) ---
        employee_check_ins = {}
        for r in all_week_records:
            if r.Attendance.check_in:
                if r.full_name not in employee_check_ins:
                    employee_check_ins[r.full_name] = []
                employee_check_ins[r.full_name].append(datetime.strptime(r.Attendance.check_in, '%H:%M:%S').time())

        employee_punctuality = []
        for name, times in employee_check_ins.items():
            total_seconds = sum(t.hour * 3600 + t.minute * 60 + t.second for t in times)
            avg_seconds = total_seconds / len(times)
            employee_punctuality.append({'name': name, 'avg_check_in_seconds': avg_seconds})

        employee_punctuality.sort(key=lambda x: x['avg_check_in_seconds'])
        
        most_punctual = [emp['name'] for emp in employee_punctuality[:5]]
        least_punctual = [emp['name'] for emp in employee_punctuality[-5:][::-1]]

        return jsonify({
            'week_start': start_of_week.isoformat(),
            'week_end': (start_of_week + timedelta(days=4)).isoformat(), # Report still shows Mon-Fri
            'daily_summary': final_summary,
            'punctuality': {
                'most_punctual': most_punctual,
                'least_punctual': least_punctual
            }
        }), 200

    except Exception as e:
        import traceback
        print(f"Error in weekly report: {traceback.format_exc()}")
        return jsonify({"error": "Failed to generate weekly report", "details": str(e)}), 500

@app.route('/api/employees/<int:employee_id>', methods=['GET'])
def get_employee(employee_id):
    """Returns details for a single employee."""
    try:
        employee = EmployeeInfo.query.get_or_404(employee_id)
        face_embedding_count = FaceEmbedding.query.filter_by(employee_info_id=employee.id).count()
        
        return jsonify({
            "id": employee.id,
            "full_name": employee.full_name,
            "email": employee.email,
            "position": employee.position,
            "department": employee.department,
            "status": employee.status,
            "access_code": employee.access_code,
            "hired_date": employee.hired_date,
            "face_registered": face_embedding_count > 0,
            "profile_pic": employee.photo, # Add profile picture filename
            "avatar_url": f"/static/profile_pics/{employee.photo}" if employee.photo else "/static/default-avatar.svg"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    # [C-01 SECURITY FIX]
    # البحث عن المستخدم عن طريق البريد الإلكتروني
    user = User.query.filter_by(Email=email).first()

    # [C-01 SECURITY FIX]
    # التحقق من وجود المستخدم وصحة كلمة المرور باستخدام bcrypt
    # هذا يمنع تجاوز المصادقة الذي كان موجودًا سابقًا
    if not user or not bcrypt.check_password_hash(user.Password, password):
        return jsonify({"error": "Invalid email or password"}), 401

    # تحديث تاريخ آخر تسجيل دخول
    user.last_login = datetime.utcnow().isoformat()
    db.session.commit()

    # إنشاء توكن وصول للمستخدم المصادق عليه مع إضافة الصلاحية
    additional_claims = {"role": user.Role}
    access_token = create_access_token(identity=user.Email, additional_claims=additional_claims)
    
    # Return the token and user info. Frontend JS will store this in localStorage.
    response = jsonify({
        "access_token": access_token,
        "user": {
            "email": user.Email,
            "role": user.Role,
            "employee_id": user.employee_id
        }
    })
    set_access_cookies(response, access_token) # Set JWT as an http-only cookie
    return response, 200

@app.route('/api/register', methods=['POST'])
def register_user():
    """
    Create a new user account.
    For Postman testing: POST /api/register
    Body (JSON): {
        "email": "user@example.com",
        "password": "password123",
        "role": "employee",  // optional, defaults to "employee"
        "employee_id": 1     // optional, link to existing employee
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'employee')  # Default to employee
        employee_id = data.get('employee_id')
        
        # Validation
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
            
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters long"}), 400
            
        if role not in ['admin', 'employee']:
            return jsonify({"error": "Role must be either 'admin' or 'employee'"}), 400
            
        # Check if email already exists
        existing_user = User.query.filter_by(Email=email).first()
        if existing_user:
            return jsonify({"error": "Email already registered"}), 409
            
        # Validate employee_id if provided
        if employee_id:
            employee = EmployeeInfo.query.get(employee_id)
            if not employee:
                return jsonify({"error": "Employee ID not found"}), 404
                
            # Check if employee already has a user account
            existing_user_with_employee = User.query.filter_by(employee_id=employee_id).first()
            if existing_user_with_employee:
                return jsonify({"error": "Employee already has a user account"}), 409
        
        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create new user
        new_user = User(
            Email=email,
            Password=hashed_password,
            Role=role,
            employee_id=employee_id,
            is_active=1,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        # Return success response (don't include password)
        response_data = {
            "id": new_user.Id,
            "email": new_user.Email,
            "role": new_user.Role,
            "employee_id": new_user.employee_id,
            "is_active": new_user.is_active,
            "created_at": new_user.created_at
        }
        
        # Include employee info if linked
        if employee_id and employee:
            response_data["employee_info"] = {
                "full_name": employee.full_name,
                "position": employee.position,
                "department": employee.department
            }
        
        return jsonify({
            "success": True,
            "message": "User created successfully",
            "user": response_data
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to create user: {str(e)}"}), 500

@app.route('/api/attendance/recent', methods=['GET'])
def recent_attendance():
    """Returns the 20 most recent attendance records."""
    try:
        records = db.session.query(
            Attendance,
            EmployeeInfo.full_name.label('employee_name')
        ).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).order_by(
            Attendance.created_at.desc()
        ).limit(20).all()

        result = []
        for r in records:
            result.append({
                'id': r.Attendance.id,
                'employee_name': r.employee_name,
                'date': r.Attendance.date,
                'check_in': r.Attendance.check_in,
                'check_out': r.Attendance.check_out,
                'status': r.Attendance.status
            })
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/employees/<int:employee_id>', methods=['PUT'])
@admin_required
def update_employee(employee_id):
    employee = EmployeeInfo.query.get_or_404(employee_id)
    
    # This endpoint now primarily handles multipart/form-data
    employee.full_name = request.form.get('full_name', employee.full_name)
    employee.email = request.form.get('email', employee.email)
    employee.position = request.form.get('position', employee.position)
    
    # Handle department update
    department_id = request.form.get('department_id')
    if department_id:
        department = Department.query.get(department_id)
        if department:
            employee.department_id = department.id
            employee.department = department.name
    
    employee.hired_date = request.form.get('hired_date', employee.hired_date)
    employee.status = request.form.get('status', employee.status)
    
    # Handle photo upload
    if 'profile_pic' in request.files:
        photo_file = request.files['profile_pic']
        if photo_file and photo_file.filename != '':
            # Validate file type
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if '.' in photo_file.filename and photo_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                # Generate unique filename
                photo_filename = f"employee_{employee_id}_{int(time.time())}.{photo_file.filename.rsplit('.', 1)[1].lower()}"
                
                # Define the correct save directory
                photos_dir = os.path.join(BASE_PROJECT_DIR, "static", "profile_pics")
                os.makedirs(photos_dir, exist_ok=True)
                
                # Save the file
                photo_path = os.path.join(photos_dir, photo_filename)
                photo_file.save(photo_path)
                
                # Update employee photo field in the database
                employee.photo = photo_filename
            else:
                return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed"}), 400
    
    db.session.commit()
    
    return jsonify({"success": True, "message": "Employee updated successfully"}), 200
@app.route('/employee_photos/<filename>')
def serve_employee_photo(filename):
    """Serve employee photos from the employee_photos directory."""
    photos_dir = os.path.join(BASE_PROJECT_DIR, "employee_photos")
    return send_from_directory(photos_dir, filename)

# --- Employee Self-Service API Routes ---
@app.route('/api/employee/profile', methods=['GET'])
@employee_required # [C-01 SECURITY FIX] - تم تفعيل هذا المصادق لمنع الوصول غير المصرح به
def get_employee_profile():
    """
    Get the current employee's profile information (read-only).
    """
    try:
        user = request.current_user
        
        if not user.employee_id:
            return jsonify({"error": "No employee profile linked to this account"}), 404
            
        employee = EmployeeInfo.query.get(user.employee_id)
        if not employee:
            return jsonify({"error": "Employee profile not found"}), 404
            
        # Get department name
        department_name = None
        if employee.department_id:
            department = Department.query.get(employee.department_id)
            department_name = department.name if department else None
            
        return jsonify({
            "id": employee.id,
            "full_name": employee.full_name,
            "email": employee.email,
            "position": employee.position,
            "department": department_name or employee.department,
            "hired_date": employee.hired_date,
            "photo": employee.photo,
            "avatar_url": f"/static/profile_pics/{employee.photo}" if employee.photo else "/static/default-avatar.svg",
            "status": employee.status,
            "access_code": employee.access_code
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch profile: {str(e)}"}), 500

@app.route('/api/employee/attendance', methods=['GET'])
@employee_required
def get_employee_attendance():
    """
    Get the current employee's attendance records.
    """
    try:
        user = request.current_user
        
        if not user.employee_id:
            return jsonify({"error": "No employee profile linked to this account"}), 404
            
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Build query for employee's attendance records
        query = db.session.query(
            Attendance,
            EmployeeInfo.full_name
        ).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).filter(
            Attendance.employee_info_id == user.employee_id
        )
        
        # Apply date filters if provided
        if start_date:
            query = query.filter(Attendance.date >= start_date)
        if end_date:
            query = query.filter(Attendance.date <= end_date)
            
        # Order by date descending
        query = query.order_by(Attendance.date.desc(), Attendance.check_in.desc())
        
        # Paginate results
        total = query.count()
        records = query.offset((page - 1) * per_page).limit(per_page).all()
        
        # Format response
        attendance_data = []
        for record, employee_name in records:
            attendance_data.append({
                "id": record.id,
                "employee_name": employee_name,
                "date": record.date,
                "check_in_time": record.check_in,
                "check_out_time": record.check_out,
                "status": record.status,
                "hours_worked": calculate_hours_worked(record.check_in, record.check_out)
            })
        
        return jsonify({
            "attendance_records": attendance_data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        }), 200
        
    except Exception as e:
        print(f"ERROR in get_employee_attendance: {e}")
        return jsonify({"error": f"Failed to fetch attendance records: {str(e)}"}), 500

@app.route('/api/employee/attendance/summary', methods=['GET'])
@employee_required
def get_employee_attendance_summary():
    """
    Get attendance summary statistics for the current employee.
    """
    try:
        user = request.current_user
        
        if not user.employee_id:
            return jsonify({"error": "No employee profile linked to this account"}), 404
            
        # Get date range (default to current month)
        end_date = date.today()
        start_date = end_date.replace(day=1)
        
        # Override with query parameters if provided
        if request.args.get('start_date'):
            start_date = datetime.strptime(request.args.get('start_date'), '%Y-%m-%d').date()
        if request.args.get('end_date'):
            end_date = datetime.strptime(request.args.get('end_date'), '%Y-%m-%d').date()
            
        # Get attendance records for the period
        records = db.session.query(Attendance).filter(
            Attendance.employee_info_id == user.employee_id,
            Attendance.date.between(start_date.isoformat(), end_date.isoformat())
        ).all()
        
        # Calculate statistics
        total_days = len(records)
        present_days = len([r for r in records if r.status in ['Present', 'Late']])
        absent_days = len([r for r in records if r.status == 'Absent'])
        late_days = len([r for r in records if r.status == 'Late'])
        
        # Calculate total hours worked
        total_hours = sum([float(calculate_hours_worked(r.check_in, r.check_out) or 0) for r in records if r.check_in and r.check_out])
        
        # Calculate working days in the period (excluding weekends)
        working_days = 0
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                working_days += 1
            current_date += timedelta(days=1)
            
        attendance_rate = round((present_days / working_days) * 100, 1) if working_days > 0 else 0
        
        
        print(f"  Summary: Present: {present_days}, Absent: {absent_days}, Late: {late_days}, Total Hours: {total_hours:.2f}, Rate: {attendance_rate}%")
        return jsonify({
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "statistics": {
                "total_working_days": working_days,
                "total_recorded_days": total_days,
                "present_days": present_days,
                "absent_days": absent_days,
                "late_days": late_days,
                "total_hours_worked": round(total_hours, 2),
                "attendance_rate": attendance_rate
            }
        }), 200
        
    except Exception as e:
        print(f"ERROR in get_employee_attendance_summary: {e}")
        return jsonify({"error": f"Failed to fetch attendance summary: {str(e)}"}), 500

@app.route('/api/admin/employee/attendance/summary', methods=['GET'])
@admin_required
def get_admin_employee_attendance_summary():
    """
    Get attendance summary statistics for a specific employee (admin only).
    Query parameters:
    - employee_id: The employee ID to get summary for (required)
    - start_date: Filter from date (YYYY-MM-DD, optional, defaults to start of current month)
    - end_date: Filter to date (YYYY-MM-DD, optional, defaults to today)
    """
    try:
        employee_id = request.args.get('employee_id', type=int)
        if not employee_id:
            return jsonify({"error": "employee_id parameter is required"}), 400

        # Check if employee exists
        employee = EmployeeInfo.query.get(employee_id)
        if not employee:
            return jsonify({"error": "Employee not found"}), 404

        # Get date range (default to current month)
        end_date = date.today()
        start_date = end_date.replace(day=1)

        # Override with query parameters if provided
        if request.args.get('start_date'):
            start_date = datetime.strptime(request.args.get('start_date'), '%Y-%m-%d').date()
        if request.args.get('end_date'):
            end_date = datetime.strptime(request.args.get('end_date'), '%Y-%m-%d').date()

        # Get attendance records for the period
        records = db.session.query(Attendance).filter(
            Attendance.employee_info_id == employee_id,
            Attendance.date.between(start_date.isoformat(), end_date.isoformat())
        ).all()

        # Calculate statistics
        total_days = len(records)
        present_days = len([r for r in records if r.status in ['Present', 'Late']])
        absent_days = len([r for r in records if r.status == 'Absent'])
        late_days = len([r for r in records if r.status == 'Late'])

        # Calculate total hours worked
        total_hours = sum([float(calculate_hours_worked(r.check_in, r.check_out) or 0) for r in records if r.check_in and r.check_out])

        # Calculate working days in the period (excluding weekends)
        working_days = 0
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                working_days += 1
            current_date += timedelta(days=1)

        attendance_rate = round((present_days / working_days) * 100, 1) if working_days > 0 else 0

        return jsonify({
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "statistics": {
                "total_working_days": working_days,
                "total_recorded_days": total_days,
                "present_days": present_days,
                "absent_days": absent_days,
                "late_days": late_days,
                "total_hours_worked": round(total_hours, 2),
                "attendance_rate": attendance_rate
            }
        }), 200

    except Exception as e:
        print(f"ERROR in get_admin_employee_attendance_summary: {e}")
        return jsonify({"error": f"Failed to fetch attendance summary: {str(e)}"}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
@admin_required
def dashboard_stats():
    """Returns daily attendance statistics."""
    try:
        today = date.today().isoformat()
        
        # جلب العدد الإجمالي للموظفين النشطين
        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        # حساب عدد الموظفين الحاضرين أو المتأخرين
        present_count = Attendance.query.filter(
            Attendance.date == today,
            Attendance.status.in_(['Present', 'Late'])
        ).count()
        
        # حساب عدد الموظفين المسجلين كـ "متأخر" بشكل صريح
        late_count = Attendance.query.filter_by(date=today, status='Late').count()
        
        # [منطق جديد] أولاً، محاولة حساب الغياب من السجلات الصريحة التي أنشأها المجدول
        absent_count = Attendance.query.filter_by(date=today, status='Absent').count()

        # إذا لم تكن مهمة المجدول قد عملت بعد (عدد الغائبين المسجلين = 0)،
        # نلجأ إلى الطريقة الحسابية القديمة لتقديم تقدير لحظي للغياب خلال اليوم.
        if absent_count == 0:
            implicit_absent = total_employees - present_count
            absent_count = max(0, implicit_absent) # التأكد من أن القيمة ليست سالبة

        return jsonify({
            "present": present_count,
            "absent": absent_count,
            "late": late_count,
            "total_employees": total_employees
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard/weekly-summary', methods=['GET'])
@admin_required
def get_weekly_summary_for_dashboard():
    """
    Returns attendance data for the last 7 days for the dashboard chart.
    """
    try:
        # Get today's date
        today = date.today()
        
        # Calculate the date 7 days ago
        seven_days_ago = today - timedelta(days=6)
        
        # Query to get attendance counts grouped by date for the last 7 days
        summary = db.session.query(
            Attendance.date,
            func.count(Attendance.id).label('present_count')
        ).filter(
            Attendance.date.between(seven_days_ago.isoformat(), today.isoformat()),
            Attendance.status.in_(['Present', 'Late'])
        ).group_by(
            Attendance.date
        ).order_by(
            Attendance.date
        ).all()

        # Create a dictionary to hold the results, with all 7 days initialized to 0
        results = { (seven_days_ago + timedelta(days=i)).isoformat(): 0 for i in range(7) }
        
        # Populate the dictionary with the query results
        for record in summary:
            results[record.date] = record.present_count
            
        # Format the data for Chart.js
        chart_data = {
            'labels': [datetime.fromisoformat(d).strftime('%a') for d in results.keys()],
            'data': list(results.values())
        }
        
        return jsonify(chart_data), 200

    except Exception as e:
        return jsonify({"error": f"Failed to generate weekly summary: {str(e)}"}), 500
        
@app.route('/api/attendance/recognize-and-checkin', methods=['POST'])
def recognize_and_checkin():
    """
    Receives an image, recognizes the employee, and records their attendance.
    """
    # 1. Validate image was sent
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # 2. Process the uploaded image
    try:
        image_file = request.files['image']
        image_stream = image_file.read()
        numpy_image = np.frombuffer(image_stream, np.uint8)
        image_cv = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)

        print("🔍 معالجة الصورة المرفوعة للتعرف على الوجه...")
        processed_face = load_and_preprocess_image(image_cv, is_path=False)
        if processed_face is None:
            print("⚠️ [خطأ] لم يتم اكتشاف وجه في الصورة")
            return jsonify({"error": "No face detected in the image"}), 400

        print("🧠 توليد embedding للوجه...")
        new_embedding = get_embedding(processed_face)
        if new_embedding is None:
            print("⚠️ [خطأ] فشل في توليد embedding للوجه")
            return jsonify({"error": "Could not generate face embedding"}), 500

        print(f"✅ تم توليد embedding للوجه بنجاح. الشكل: {new_embedding.shape}")

    except Exception as e:
        print(f"⚠️ [خطأ] Error processing image: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    # 3. Recognize face using the new.ipynb approach
    try:
        print("🎯 بدء التعرف على الوجه من قاعدة البيانات...")
        recognized_employee, confidence = recognize_face_from_database(new_embedding, db.session)

        if recognized_employee is not None:
            # A confident match was found, record attendance
            print(f"✅ تم التعرف على الموظف: {recognized_employee.full_name}")
            response_data, status_code = record_attendance(recognized_employee.id)

            # Add recognition details to the response
            response_data['recognized_as'] = recognized_employee.full_name
            response_data['confidence'] = float(confidence)

            return jsonify(response_data), status_code
        else:
            # The person is not recognized
            print(f"❌ فشل التعرف على الوجه. الثقة: {confidence:.4f}")
            return jsonify({
                "error": "Face not recognized. Please ensure you are registered and try again.",
                "confidence": float(confidence)
            }), 404

    except Exception as e:
        print(f"⚠️ [خطأ] Error during face recognition: {e}")
        return jsonify({"error": f"Face recognition error: {str(e)}"}), 500
    
@app.route('/api/admins', methods=['POST'])
def create_admin():
    """
    Creates a new admin user.
    Expects a JSON payload with 'email' and 'password'.
    """
    # 1. Get data from the request
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # 2. Validate the input
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # 3. Check if the user already exists
    if User.query.filter_by(Email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409 # 409 Conflict

    try:
        # 4. Hash the password for security
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # 5. Create the new admin user object
        new_admin = User(
            Email=email,
            Password=hashed_password,
            Role='admin'  # Explicitly set the role to 'admin'
        )

        # 6. Add to the database and commit
        db.session.add(new_admin)
        db.session.commit()

        # 7. Return a success response
        return jsonify({
            "message": "Admin user created successfully",
            "user": {
                "Id": new_admin.Id,
                "Email": new_admin.Email,
                "Role": new_admin.Role
            }
        }), 201 # 201 Created

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to create admin user", "details": str(e)}), 500

@app.route('/api/reports/daily', methods=['GET'])
def get_daily_report():
    """
    Returns daily attendance report for a specific date.
    Query parameter: date (YYYY-MM-DD format, defaults to today)
    """
    try:
        report_date = request.args.get('date', date.today().isoformat())

        try:
            datetime.fromisoformat(report_date)
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        # Get all attendance records for the specified date (Present, Late, AND Absent)
        all_records_for_day = db.session.query(
            Attendance,
            EmployeeInfo.full_name,
            EmployeeInfo.department,
            EmployeeInfo.position
        ).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).filter(
            Attendance.date == report_date
        ).all()

        # --- [NEW HYBRID LOGIC] ---
        present_count = len([r for r in all_records_for_day if r.Attendance.status in ['Present', 'Late']])
        late_count = len([r for r in all_records_for_day if r.Attendance.status == 'Late'])
        
        # Get explicitly absent employees (from scheduler)
        explicit_absent_records = [r for r in all_records_for_day if r.Attendance.status == 'Absent']
        
        # Get implicitly absent employees (if scheduler hasn't run)
        recorded_employee_ids = {r.Attendance.employee_info_id for r in all_records_for_day}
        implicit_absent_query = EmployeeInfo.query.filter(
            EmployeeInfo.status == 'active',
            ~EmployeeInfo.id.in_(recorded_employee_ids)
        ).all()

        # Combine lists for the final report
        absent_employees_list = [
            {"name": r.full_name, "department": r.department or "N/A"}
            for r in explicit_absent_records
        ] + [
            {"name": emp.full_name, "department": emp.department or "N/A"}
            for emp in implicit_absent_query
        ]
        absent_count = len(absent_employees_list)
        # --- [END NEW LOGIC] ---

        # Format data for present/late employees
        attendance_data = []
        for r in all_records_for_day:
            if r.Attendance.status in ['Present', 'Late']:
                attendance_data.append({
                    "employee_name": r.full_name,
                    "department": r.department or "N/A",
                    "position": r.position or "N/A",
                    "check_in": r.Attendance.check_in or "N/A",
                    "check_out": r.Attendance.check_out or "N/A",
                    "status": r.Attendance.status or "N/A",
                    "hours_worked": calculate_hours_worked(r.Attendance.check_in, r.Attendance.check_out)
                })

        late_arrivals = [
            {"name": r.full_name, "department": r.department or "N/A", "check_in_time": r.Attendance.check_in}
            for r in all_records_for_day if r.Attendance.status == 'Late'
        ]

        return jsonify({
            "date": report_date,
            "summary": {
                "present": present_count,
                "absent": absent_count,
                "late": late_count,
                "total": total_employees
            },
            "attendance": attendance_data,
            "lateArrivals": late_arrivals,
            "absentEmployees": absent_employees_list
        }), 200

    except Exception as e:
        return jsonify({"error": "Failed to generate daily report", "details": str(e)})

def calculate_hours_worked(check_in, check_out):
    """Calculate hours worked between check-in and check-out times."""
    if not check_in or not check_out:
        return "N/A"

    try:
        check_in_time = datetime.strptime(check_in, '%H:%M:%S')
        check_out_time = datetime.strptime(check_out, '%H:%M:%S')

        # Handle case where check-out is next day
        if check_out_time < check_in_time:
            check_out_time += timedelta(days=1)

        time_diff = check_out_time - check_in_time
        hours = time_diff.total_seconds() / 3600
        return f"{hours:.2f}"
    except ValueError:
        return "N/A"

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current system settings."""
    try:
        settings = SystemSetting.query.get(1)
        if not settings:
            # If no settings row exists at all, create one with defaults.
            settings = SystemSetting(id=1)
            db.session.add(settings)
            db.session.commit()

        # Safely build the settings dictionary, providing a default for every field.
        settings_data = {
            "language": settings.language or 'en',
            "dateFormat": settings.date_format or 'MM/DD/YYYY',
            "timeFormat": settings.time_format or '12h',
            "workStartTime": settings.work_start_time or '08:00',
            "workEndTime": settings.work_end_time or '17:00',
            "lateTolerance": settings.max_late_minutes if settings.max_late_minutes is not None else 15,
            "autoCheckout": bool(settings.auto_checkout) if settings.auto_checkout is not None else False,
            "weekendDays": [int(x) for x in settings.weekend_days.split(',') if x] if settings.weekend_days else [5, 6],
            "emailNotifications": bool(settings.email_notifications) if settings.email_notifications is not None else True,
            "browserNotifications": bool(settings.browser_notifications) if settings.browser_notifications is not None else False,
            "notificationTypes": settings.notification_types.split(',') if settings.notification_types else ['late', 'absent', 'system'],
            "sessionTimeout": settings.session_timeout if settings.session_timeout is not None else 30,
            "passwordPolicy": settings.password_policy.split(',') if settings.password_policy else ['complexity', 'history'],
            "twoFactorAuth": bool(settings.two_factor_auth) if settings.two_factor_auth is not None else False,
            "apiAccess": bool(settings.api_access) if settings.api_access is not None else False,
            "attendanceDeadline": settings.attendance_deadline or '09:30',
            "checkoutStartTime": settings.checkout_start_time or '16:30',
            "enforceWorkEndTime": bool(settings.enforce_work_end_time) if settings.enforce_work_end_time is not None else False
        }
        return jsonify(settings_data), 200
    except Exception as e:
        # Log the error for better debugging
        import traceback
        print(f"❌ Error in get_settings: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Update system settings."""
    try:
        data = request.get_json()

        settings = SystemSetting.query.get(1)
        if not settings:
            # If settings with id=1 don't exist, create them
            settings = SystemSetting(id=1)
            db.session.add(settings)
            db.session.flush() # Ensure the new settings object is in the session

        # Update settings from request data
        if 'language' in data:
            settings.language = data['language']
        if 'dateFormat' in data:
            settings.date_format = data['dateFormat']
        if 'timeFormat' in data:
            settings.time_format = data['timeFormat']
        if 'workStartTime' in data:
            settings.work_start_time = data['workStartTime']
        if 'workEndTime' in data:
            settings.work_end_time = data['workEndTime']
        if 'lateTolerance' in data:
            try:
                # [FIX] Ensure lateTolerance is always an integer
                settings.max_late_minutes = int(data['lateTolerance'])
            except (ValueError, TypeError):
                # If the value is not a valid integer, return an error
                return jsonify({"error": "Invalid value for lateTolerance. It must be a whole number."}),
        if 'autoCheckout' in data:
            settings.auto_checkout = 1 if data['autoCheckout'] else 0
        if 'weekendDays' in data:
            settings.weekend_days = ','.join(map(str, data['weekendDays']))
        if 'emailNotifications' in data:
            settings.email_notifications = 1 if data['emailNotifications'] else 0
        if 'browserNotifications' in data:
            settings.browser_notifications = 1 if data['browserNotifications'] else 0
        if 'notificationTypes' in data:
            settings.notification_types = ','.join(data['notificationTypes'])
        if 'sessionTimeout' in data:
            settings.session_timeout = data['sessionTimeout']
        if 'passwordPolicy' in data:
            settings.password_policy = ','.join(data['passwordPolicy'])
        if 'twoFactorAuth' in data:
            settings.two_factor_auth = 1 if data['twoFactorAuth'] else 0
        if 'apiAccess' in data:
            settings.api_access = 1 if data['apiAccess'] else 0
        
        # Add specific attendance settings
        if 'attendanceDeadline' in data:
            settings.attendance_deadline = data['attendanceDeadline']
        if 'checkoutStartTime' in data:
            settings.checkout_start_time = data['checkoutStartTime']
        
        # [تعديل] حفظ قيمة صندوق الاختيار للسياسة الصارمة
        # يتم تحويل القيمة إلى Boolean قبل الحفظ في قاعدة البيانات
        if 'enforceWorkEndTime' in data:
            settings.enforce_work_end_time = bool(data['enforceWorkEndTime'])

        db.session.commit()
        return jsonify({"message": "Settings updated successfully"}), 200
    except Exception as e:
        db.session.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error updating settings: {str(e)}")
        print(f"❌ Full traceback: {error_details}")
        return jsonify({"error": f"Failed to update settings: {str(e)}", "details": error_details if app.debug else None}), 500

@app.route('/api/biometric/authenticate', methods=['POST'])
def biometric_authenticate():
    """
    Simulate biometric authentication and record attendance.
    In a real implementation, this would interface with biometric hardware.
    """
    try:
        data = request.get_json()
        fingerprint_data = data.get('fingerprint_data')  # This would be actual biometric data

        if not fingerprint_data:
            return jsonify({"error": "No biometric data provided"}), 400

        # In a real implementation, you would:
        # 1. Process the biometric data
        # 2. Compare against stored biometric templates
        # 3. Return the matched employee

        # For simulation, we'll randomly select an employee
        employees = EmployeeInfo.query.filter_by(status='active').all()
        if not employees:
            return jsonify({"error": "No active employees found"}), 404

        # Simulate 90% success rate
        if random.random() > 0.1:
            import random
            employee = random.choice(employees)

            # Record attendance
            response_data, status_code = record_attendance(employee.id)

            # Add biometric authentication details
            response_data['employee_name'] = employee.full_name
            response_data['authentication_method'] = 'biometric'
            response_data['timestamp'] = datetime.now().isoformat()

            return jsonify(response_data), status_code
        else:
            return jsonify({
                "error": "Biometric authentication failed",
                "message": "Fingerprint not recognized"
            }), 401

    except Exception as e:
        return jsonify({"error": f"Biometric authentication error: {str(e)}"}), 500

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Get all notifications for the current user."""
    try:
        notifications = Notification.query.order_by(Notification.created_at.desc()).all()

        result = []
        for notification in notifications:
            result.append({
                "id": notification.id,
                "title": notification.title,
                "message": notification.message,
                "type": notification.type,
                "priority": notification.priority,
                "read": bool(notification.read),
                "details": notification.details,
                "timestamp": notification.created_at,
                "related_employee_id": notification.related_employee_id
            })

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications/<int:notification_id>/read', methods=['PUT'])
def mark_notification_read(notification_id):
    """Mark a specific notification as read."""
    try:
        notification = Notification.query.get_or_404(notification_id)
        notification.read = 1
        db.session.commit()
        return jsonify({"message": "Notification marked as read"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications/mark-all-read', methods=['PUT'])
def mark_all_notifications_read():
    """Mark all notifications as read."""
    try:
        Notification.query.update({Notification.read: 1})
        db.session.commit()
        return jsonify({"message": "All notifications marked as read"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications', methods=['DELETE'])
def clear_all_notifications():
    """Clear all notifications."""
    try:
        Notification.query.delete()
        db.session.commit()
        return jsonify({"message": "All notifications cleared"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

def create_notification(title, message, notification_type, priority='medium', details=None, employee_id=None):
    """Helper function to create notifications."""
    try:
        notification = Notification(
            title=title,
            message=message,
            type=notification_type,
            priority=priority,
            details=details,
            related_employee_id=employee_id
        )
        db.session.add(notification)
        db.session.commit()
        return notification
    except Exception as e:
        db.session.rollback()
        print(f"Error creating notification: {e}")
        return None

def generate_unique_code(session, length=6):
    """Generate a unique access code for employees."""
    characters = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choices(characters, k=length))
        # Check if code already exists
        existing = session.query(EmployeeInfo).filter_by(access_code=code).first()
        if not existing:
            return code

@app.route('/api/attendance/code-checkin', methods=['POST'])
def code_checkin():
    """
    Handle attendance check-in/out using employee access code.
    """
    try:
        data = request.get_json()
        access_code = data.get('access_code')

        if not access_code:
            return jsonify({"error": "Access code is required"}), 400

        # Find employee by access code
        employee = EmployeeInfo.query.filter_by(access_code=access_code, status='active').first()

        if not employee:
            return jsonify({"error": "Invalid access code or employee not active"}), 404

        # Record attendance
        response_data, status_code = record_attendance(employee.id)

        # Add employee details to response
        response_data['recognized_as'] = employee.full_name
        response_data['authentication_method'] = 'access_code'
        response_data['timestamp'] = datetime.now().isoformat()

        # Create notification for successful check-in
        action_text = "checked in" if response_data.get('action') == 'checkin' else "checked out"
        create_notification(
            title=f"Code Check-in: {employee.full_name}",
            message=f"{employee.full_name} {action_text} using access code at {datetime.now().strftime('%H:%M')}",
            notification_type='attendance',
            priority='low',
            employee_id=employee.id
        )

        return jsonify(response_data), status_code

    except Exception as e:
        return jsonify({"error": f"Code check-in error: {str(e)}"}), 500

@app.route('/api/reports/daily/pdf', methods=['GET'])
@admin_required
def generate_daily_report_pdf():
    """
    Generate PDF for daily attendance report.
    Query parameter: date (YYYY-MM-DD format, defaults to today)
    """
    try:
        report_date = request.args.get('date', date.today().isoformat())

        try:
            datetime.fromisoformat(report_date)
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Get report data
        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        all_records_for_day = db.session.query(
            Attendance,
            EmployeeInfo.full_name,
            EmployeeInfo.department,
            EmployeeInfo.position
        ).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).filter(
            Attendance.date == report_date
        ).all()

        present_count = len([r for r in all_records_for_day if r.Attendance.status in ['Present', 'Late']])
        late_count = len([r for r in all_records_for_day if r.Attendance.status == 'Late'])

        explicit_absent_records = [r for r in all_records_for_day if r.Attendance.status == 'Absent']
        recorded_employee_ids = {r.Attendance.employee_info_id for r in all_records_for_day}
        implicit_absent_query = EmployeeInfo.query.filter(
            EmployeeInfo.status == 'active',
            ~EmployeeInfo.id.in_(recorded_employee_ids)
        ).all()

        absent_employees_list = [
            {"name": r.full_name, "department": r.department or "N/A"}
            for r in explicit_absent_records
        ] + [
            {"name": emp.full_name, "department": emp.department or "N/A"}
            for emp in implicit_absent_query
        ]
        absent_count = len(absent_employees_list)

        attendance_data = []
        for r in all_records_for_day:
            if r.Attendance.status in ['Present', 'Late']:
                attendance_data.append({
                    "employee_name": r.full_name,
                    "department": r.department or "N/A",
                    "position": r.position or "N/A",
                    "check_in": r.Attendance.check_in or "N/A",
                    "check_out": r.Attendance.check_out or "N/A",
                    "status": r.Attendance.status or "N/A",
                    "hours_worked": calculate_hours_worked(r.Attendance.check_in, r.Attendance.check_out)
                })

        late_arrivals = [
            {"name": r.full_name, "department": r.department or "N/A", "check_in_time": r.Attendance.check_in}
            for r in all_records_for_day if r.Attendance.status == 'Late'
        ]

        # Generate HTML for PDF
        html_content = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <title>تقرير الحضور اليومي - {report_date}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    direction: rtl;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-around;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                    flex: 1;
                    margin: 0 10px;
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                }}
                .present {{ background-color: #27ae60; color: white; }}
                .absent {{ background-color: #e74c3c; color: white; }}
                .late {{ background-color: #f39c12; color: white; }}
                .total {{ background-color: #3498db; color: white; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: right;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .section-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin: 30px 0 15px 0;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>تقرير الحضور اليومي</h1>
                <h2>التاريخ: {report_date}</h2>
            </div>

            <div class="summary">
                <div class="summary-card present">
                    <h3>الحاضرون</h3>
                    <p style="font-size: 24px; margin: 0;">{present_count}</p>
                </div>
                <div class="summary-card absent">
                    <h3>الغائبون</h3>
                    <p style="font-size: 24px; margin: 0;">{absent_count}</p>
                </div>
                <div class="summary-card late">
                    <h3>المتأخرون</h3>
                    <p style="font-size: 24px; margin: 0;">{late_count}</p>
                </div>
                <div class="summary-card total">
                    <h3>إجمالي الموظفين</h3>
                    <p style="font-size: 24px; margin: 0;">{total_employees}</p>
                </div>
            </div>

            <div class="section-title">تفاصيل الحضور</div>
            <table>
                <thead>
                    <tr>
                        <th>اسم الموظف</th>
                        <th>القسم</th>
                        <th>المنصب</th>
                        <th>وقت الحضور</th>
                        <th>وقت الانصراف</th>
                        <th>الحالة</th>
                        <th>ساعات العمل</th>
                    </tr>
                </thead>
                <tbody>
        """

        for record in attendance_data:
            html_content += f"""
                    <tr>
                        <td>{record['employee_name']}</td>
                        <td>{record['department']}</td>
                        <td>{record['position']}</td>
                        <td>{record['check_in']}</td>
                        <td>{record['check_out']}</td>
                        <td>{record['status']}</td>
                        <td>{record['hours_worked']}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>

            <div class="section-title">المتأخرون</div>
            <table>
                <thead>
                    <tr>
                        <th>اسم الموظف</th>
                        <th>القسم</th>
                        <th>وقت الحضور</th>
                    </tr>
                </thead>
                <tbody>
        """

        for late in late_arrivals:
            html_content += f"""
                    <tr>
                        <td>{late['name']}</td>
                        <td>{late['department']}</td>
                        <td>{late['check_in_time']}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>

            <div class="section-title">الغائبون</div>
            <table>
                <thead>
                    <tr>
                        <th>اسم الموظف</th>
                        <th>القسم</th>
                    </tr>
                </thead>
                <tbody>
        """

        for absent in absent_employees_list:
            html_content += f"""
                    <tr>
                        <td>{absent['name']}</td>
                        <td>{absent['department']}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """

        # Generate PDF
        pdf_buffer = generate_pdf_from_html(html_content)

        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_buffer)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=daily_report_{report_date}.pdf'
        return response

    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

@app.route('/api/reports/weekly/pdf', methods=['GET'])
@admin_required
def generate_weekly_report_pdf():
    """
    Generate PDF for weekly attendance report.
    """
    try:
        today = date.today()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        all_week_records = db.session.query(Attendance, EmployeeInfo.full_name).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).filter(
            Attendance.date.between(start_of_week.isoformat(), end_of_week.isoformat())
        ).all()

        final_summary = []
        for i in range(7):
            current_day_date = start_of_week + timedelta(days=i)
            day_name = current_day_date.strftime('%A')
            date_iso = current_day_date.isoformat()

            records_for_day = [r for r in all_week_records if r.Attendance.date == date_iso]

            present_count = len([r for r in records_for_day if r.Attendance.status in ['Present', 'Late']])

            absent_count = len([r for r in records_for_day if r.Attendance.status == 'Absent'])

            if absent_count == 0:
                implicit_absent = total_employees - present_count
                absent_count = max(0, implicit_absent)

            avg_check_in_str = 'N/A'
            check_in_times = [datetime.strptime(r.Attendance.check_in, '%H:%M:%S').time() for r in records_for_day if r.Attendance.check_in]
            if check_in_times:
                total_seconds = sum(t.hour * 3600 + t.minute * 60 + t.second for t in check_in_times)
                avg_seconds = total_seconds / len(check_in_times)
                avg_time_obj = (datetime.min + timedelta(seconds=avg_seconds)).time()
                avg_check_in_str = avg_time_obj.strftime('%I:%M %p')

            final_summary.append({
                'day': day_name,
                'present': present_count,
                'absent': absent_count,
                'avg_check_in': avg_check_in_str
            })

        # Generate HTML for PDF
        html_content = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <title>تقرير الحضور الأسبوعي</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    direction: rtl;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: center;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .summary {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>تقرير الحضور الأسبوعي</h1>
                <h2>من {start_of_week.strftime('%Y-%m-%d')} إلى {end_of_week.strftime('%Y-%m-%d')}</h2>
            </div>

            <div class="summary">
                <h3>ملخص الأسبوع</h3>
                <p>إجمالي الموظفين: {total_employees}</p>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>اليوم</th>
                        <th>الحاضرون</th>
                        <th>الغائبون</th>
                        <th>متوسط وقت الحضور</th>
                    </tr>
                </thead>
                <tbody>
        """

        for day in final_summary:
            html_content += f"""
                    <tr>
                        <td>{day['day']}</td>
                        <td>{day['present']}</td>
                        <td>{day['absent']}</td>
                        <td>{day['avg_check_in']}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """

        # Generate PDF
        pdf_buffer = generate_pdf_from_html(html_content)

        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_buffer)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=weekly_report_{start_of_week.strftime("%Y%m%d")}_{end_of_week.strftime("%Y%m%d")}.pdf'
        return response

    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

@app.route('/api/reports/monthly/pdf', methods=['GET'])
@admin_required
def generate_monthly_report_pdf():
    """
    Generate PDF for monthly attendance report.
    Query parameter: month (YYYY-MM format, defaults to current month)
    """
    try:
        report_month = request.args.get('month', date.today().strftime('%Y-%m'))

        year, month = map(int, report_month.split('-'))
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        working_days = 0
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                working_days += 1
            current_date += timedelta(days=1)

        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        attendance_records = db.session.query(
            Attendance,
            EmployeeInfo.full_name,
            EmployeeInfo.department
        ).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).filter(
            Attendance.date.between(start_date.isoformat(), end_date.isoformat())
        ).all()

        daily_trend = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                day_records = [r for r in attendance_records if r.Attendance.date == current_date.isoformat()]
                present_count = len([r for r in day_records if r.Attendance.status in ['Present', 'Late']])
                attendance_percentage = round((present_count / total_employees) * 100, 1) if total_employees > 0 else 0

                daily_trend.append({
                    "date": current_date.strftime('%m/%d'),
                    "attendancePercentage": attendance_percentage
                })
            current_date += timedelta(days=1)

        # Calculate employee summaries
        employee_summaries = {}
        for record in attendance_records:
            emp_name = record.full_name
            if emp_name not in employee_summaries:
                employee_summaries[emp_name] = {
                    'name': emp_name,
                    'department': record.department or 'Unknown',
                    'present': 0,
                    'late': 0,
                    'absent': 0,
                    'total_days': 0
                }

            employee_summaries[emp_name]['total_days'] += 1
            if record.Attendance.status in ['Present', 'Late']:
                employee_summaries[emp_name]['present'] += 1
            if record.Attendance.status == 'Absent':
                employee_summaries[emp_name]['absent'] += 1
            if record.Attendance.is_late == 1:
                employee_summaries[emp_name]['late'] += 1

        employee_list = []
        for emp_data in employee_summaries.values():
            attendance_percentage = round((emp_data['present'] / working_days) * 100, 1) if working_days > 0 else 0
            employee_list.append({
                "name": emp_data['name'],
                "department": emp_data['department'],
                "daysPresent": emp_data['present'],
                "daysAbsent": emp_data['absent'],
                "lateArrivals": emp_data['late'],
                "attendancePercentage": attendance_percentage
            })

        employee_list.sort(key=lambda x: x['attendancePercentage'], reverse=True)
        top_performers = employee_list[:5]
        concerns = [emp for emp in employee_list if emp['attendancePercentage'] < 75][-5:]

        total_late_arrivals = sum(emp['lateArrivals'] for emp in employee_list)
        perfect_attendance = len([emp for emp in employee_list if emp['attendancePercentage'] == 100])
        avg_attendance = round(sum(emp['attendancePercentage'] for emp in employee_list) / len(employee_list), 1) if employee_list else 0

        # Generate HTML for PDF
        html_content = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <title>تقرير الحضور الشهري - {report_month}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    direction: rtl;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-around;
                    margin-bottom: 30px;
                    flex-wrap: wrap;
                }}
                .summary-card {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                    flex: 1;
                    margin: 10px;
                    min-width: 200px;
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: right;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .section-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin: 30px 0 15px 0;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>تقرير الحضور الشهري</h1>
                <h2>الشهر: {report_month}</h2>
            </div>

            <div class="summary">
                <div class="summary-card">
                    <h3>أيام العمل</h3>
                    <p style="font-size: 24px; margin: 0;">{working_days}</p>
                </div>
                <div class="summary-card">
                    <h3>متوسط الحضور</h3>
                    <p style="font-size: 24px; margin: 0;">{avg_attendance}%</p>
                </div>
                <div class="summary-card">
                    <h3>إجمالي المتأخرين</h3>
                    <p style="font-size: 24px; margin: 0;">{total_late_arrivals}</p>
                </div>
                <div class="summary-card">
                    <h3>حضور مثالي</h3>
                    <p style="font-size: 24px; margin: 0;">{perfect_attendance}</p>
                </div>
            </div>

            <div class="section-title">أفضل الأداء</div>
            <table>
                <thead>
                    <tr>
                        <th>اسم الموظف</th>
                        <th>القسم</th>
                        <th>أيام الحضور</th>
                        <th>أيام الغياب</th>
                        <th>التأخيرات</th>
                        <th>نسبة الحضور</th>
                    </tr>
                </thead>
                <tbody>
        """

        for emp in top_performers:
            html_content += f"""
                    <tr>
                        <td>{emp['name']}</td>
                        <td>{emp['department']}</td>
                        <td>{emp['daysPresent']}</td>
                        <td>{emp['daysAbsent']}</td>
                        <td>{emp['lateArrivals']}</td>
                        <td>{emp['attendancePercentage']}%</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>

            <div class="section-title">الموظفين الذين يحتاجون متابعة</div>
            <table>
                <thead>
                    <tr>
                        <th>اسم الموظف</th>
                        <th>القسم</th>
                        <th>أيام الحضور</th>
                        <th>أيام الغياب</th>
                        <th>التأخيرات</th>
                        <th>نسبة الحضور</th>
                    </tr>
                </thead>
                <tbody>
        """

        for emp in concerns:
            html_content += f"""
                    <tr>
                        <td>{emp['name']}</td>
                        <td>{emp['department']}</td>
                        <td>{emp['daysPresent']}</td>
                        <td>{emp['daysAbsent']}</td>
                        <td>{emp['lateArrivals']}</td>
                        <td>{emp['attendancePercentage']}%</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """

        # Generate PDF
        pdf_buffer = generate_pdf_from_html(html_content)

        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_buffer)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=monthly_report_{report_month}.pdf'
        return response

    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

@app.route('/api/reports/monthly', methods=['GET'])
def get_monthly_report():
    """
    Returns monthly attendance report for a specific month.
    Query parameter: month (YYYY-MM format, defaults to current month)
    """
    try:
        # Get month from query parameter or use current month
        report_month = request.args.get('month', date.today().strftime('%Y-%m'))

        # Validate month format
        try:
            year, month = map(int, report_month.split('-'))
            start_date = date(year, month, 1)
            # Get last day of month
            if month == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, month + 1, 1) - timedelta(days=1)
        except ValueError:
            return jsonify({"error": "Invalid month format. Use YYYY-MM"}), 400

        # Calculate working days (excluding weekends)
        working_days = 0
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                working_days += 1
            current_date += timedelta(days=1)

        # Get total active employees
        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        # Get attendance records for the month
        attendance_records = db.session.query(
            Attendance,
            EmployeeInfo.full_name,
            EmployeeInfo.department
        ).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        ).filter(
            Attendance.date.between(start_date.isoformat(), end_date.isoformat())
        ).all()

        # Calculate daily attendance trend
        daily_trend = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only working days
                day_records = [r for r in attendance_records if r.Attendance.date == current_date.isoformat()]
                present_count = len([r for r in day_records if r.Attendance.status in ['Present', 'Late']])
                attendance_percentage = round((present_count / total_employees) * 100, 1) if total_employees > 0 else 0

                daily_trend.append({
                    "date": current_date.strftime('%m/%d'),
                    "attendancePercentage": attendance_percentage
                })
            current_date += timedelta(days=1)

        # Calculate department statistics
        department_stats = {}
        for record in attendance_records:
            dept = record.department or 'Unknown'
            if dept not in department_stats:
                department_stats[dept] = {'present': 0, 'total': 0}
            if record.Attendance.status in ['Present', 'Late']:
                department_stats[dept]['present'] += 1
            department_stats[dept]['total'] += 1

        dept_list = []
        for dept, stats in department_stats.items():
            percentage = round((stats['present'] / stats['total']) * 100, 1) if stats['total'] > 0 else 0
            dept_list.append({
                "department": dept,
                "attendancePercentage": percentage
            })

        # Calculate employee summaries
        employee_summaries = {}
        for record in attendance_records:
            emp_name = record.full_name
            if emp_name not in employee_summaries:
                employee_summaries[emp_name] = {
                    'name': emp_name,
                    'department': record.department or 'Unknown',
                    'present': 0,
                    'late': 0,
                    'absent': 0,
                    'total_days': 0
                }

            employee_summaries[emp_name]['total_days'] += 1
            if record.Attendance.status in ['Present', 'Late']:
                employee_summaries[emp_name]['present'] += 1
            if record.Attendance.status == 'Absent':
                employee_summaries[emp_name]['absent'] += 1

            if record.Attendance.is_late == 1:
                employee_summaries[emp_name]['late'] += 1

        # Convert to list and calculate percentages
        employee_list = []
        for emp_data in employee_summaries.values():
            attendance_percentage = round((emp_data['present'] / working_days) * 100, 1) if working_days > 0 else 0
            employee_list.append({
                "name": emp_data['name'],
                "department": emp_data['department'],
                "daysPresent": emp_data['present'],
                "daysAbsent": emp_data['absent'],
                "lateArrivals": emp_data['late'],
                "attendancePercentage": attendance_percentage
            })

        # Sort for top performers and concerns
        employee_list.sort(key=lambda x: x['attendancePercentage'], reverse=True)
        top_performers = employee_list[:5]
        concerns = [emp for emp in employee_list if emp['attendancePercentage'] < 75][-5:]

        # Calculate summary statistics
        total_late_arrivals = sum(emp['lateArrivals'] for emp in employee_list)
        perfect_attendance = len([emp for emp in employee_list if emp['attendancePercentage'] == 100])
        avg_attendance = round(sum(emp['attendancePercentage'] for emp in employee_list) / len(employee_list), 1) if employee_list else 0

        return jsonify({
            "month": report_month,
            "summary": {
                "totalWorkingDays": working_days,
                "averageAttendance": avg_attendance,
                "totalLateArrivals": total_late_arrivals,
                "perfectAttendance": perfect_attendance
            },
            "dailyTrend": daily_trend,
            "departmentStats": dept_list,
            "topPerformers": top_performers,
            "concerns": [{"name": c["name"], "issue": "Low attendance", "attendancePercentage": c["attendancePercentage"]} for c in concerns],
            "employeeSummary": employee_list
        }), 200

    except Exception as e:
        return jsonify({"error": "Failed to generate monthly report", "details": str(e)}), 500

# ==============================================================================
#                           ONBOARDING API ENDPOINTS
# ==============================================================================

def validate_attendance_time(action, employee_id):
    """
    Validate if attendance action is allowed based on time rules and existing records.
    Returns (is_valid, error_message)
    
    [تحسين] استخدام الدوال الجديدة safe_get_time_setting للحصول على أوقات الإعدادات بأمان
    """
    from datetime import datetime, time as dt_time

    current_time = datetime.now()  # الحصول على التاريخ والوقت الحاليين
    today = current_time.date()  # الحصول على التاريخ الحالي

    # [تحسين رئيسي] استخدام safe_get_time_setting للحصول على أوقات الإعدادات بأمان
    # هذا يضمن أن الصيغ غير الصالحة لن تكسر النظام
    checkin_deadline_str = safe_get_time_setting('attendance_deadline', '09:30:00')
    checkout_start_str = safe_get_time_setting('checkout_start_time', '16:30:00')
    work_end_str = safe_get_time_setting('work_end_time', '17:00:00')
    
    # جلب إعدادات النظام للتحقق من السياسة الصارمة
    settings = SystemSetting.query.get(1)
    enforce_strict_policy = settings.enforce_work_end_time if settings else False

    try:
        # [تحسين] تحويل الأوقات المعيارية إلى كائنات وقت للمقارنة
        checkin_deadline_time = datetime.strptime(checkin_deadline_str, '%H:%M:%S').time()
        checkout_start_time = datetime.strptime(checkout_start_str, '%H:%M:%S').time()
        work_end_time = datetime.strptime(work_end_str, '%H:%M:%S').time()
        
        print(f"ℹ️ أوقات الإعدادات المستخدمة:")
        print(f"   - الموعد النهائي للحضور: {checkin_deadline_str}")
        print(f"   - أبكر وقت انصراف: {checkout_start_str}")
        print(f"   - وقت انتهاء العمل: {work_end_str}")
        print(f"   - السياسة الصارمة: {enforce_strict_policy}")
        
    except (ValueError, TypeError) as e:
        # [تحسين] تسجيل الخطأ للمطورين مع استخدام قيم افتراضية آمنة
        print(f"⚠️ خطأ في تحليل أوقات الإعدادات: {e}")
        print(f"   - استخدام القيم الافتراضية للمتابعة...")
        
        checkin_deadline_time = dt_time(9, 30)    # 09:30 صباحاً
        checkout_start_time = dt_time(16, 30)     # 04:30 مساءً
        work_end_time = dt_time(17, 0)            # 05:00 مساءً
        enforce_strict_policy = False

    # التحقق من سجلات الحضور الموجودة لليوم
    existing_attendance = Attendance.query.filter(
        Attendance.employee_info_id == employee_id,
        Attendance.date == today.isoformat()
    ).first()

    if action == 'checkin':
        # قواعد تسجيل الحضور (Check-in)
        if existing_attendance and existing_attendance.check_in:
            return False, "لقد سجلت حضورك اليوم بالفعل. يُسمح بتسجيل حضور واحد فقط يومياً."

        # [تحسين] التحقق من الموعد النهائي لتسجيل الحضور
        current_time_obj = current_time.time()
        if current_time_obj > checkin_deadline_time:
            deadline_display = checkin_deadline_time.strftime('%H:%M')
            return False, f"انتهى الموعد النهائي لتسجيل الحضور. يرجى تسجيل الحضور قبل الساعة {deadline_display}."

    elif action == 'checkout':
        # قواعد تسجيل الانصراف (Check-out)
        if not existing_attendance or not existing_attendance.check_in:
            return False, "يجب تسجيل الحضور أولاً قبل تسجيل الانصراف."

        if existing_attendance.check_out:
            return False, "لقد سجلت انصرافك اليوم بالفعل."

        current_time_obj = current_time.time()
        
        # [تحسين رئيسي] منطق التحقق من وقت الانصراف المحسن باستخدام الإعدادات الجديدة
        if enforce_strict_policy:
            # السياسة الصارمة: الانصراف فقط بعد انتهاء الدوام الرسمي
            if current_time_obj < work_end_time:
                end_time_display = work_end_time.strftime('%H:%M')
                return False, f"السياسة الصارمة مفعلة. يُسمح بالانصراف فقط بعد انتهاء وقت العمل ({end_time_display})."
            print(f"✅ تم السماح بالانصراف (السياسة الصارمة): {current_time_obj} >= {work_end_time}")
        else:
            # السياسة المرنة: الانصراف بعد أبكر وقت مسموح
            if current_time_obj < checkout_start_time:
                start_time_display = checkout_start_time.strftime('%H:%M')
                return False, f"السياسة المرنة مفعلة. يُسمح بالانصراف فقط بعد الساعة {start_time_display}."
            print(f"✅ تم السماح بالانصراف (السياسة المرنة): {current_time_obj} >= {checkout_start_time}")

    # إذا وصلنا لهنا، فالإجراء مسموح
    return True, None

@app.route('/api/attendance/face-recognition', methods=['POST'])
def attendance_face_recognition():
    """
    Handle face recognition for attendance (entry/exit).
    Allows 3 attempts to recognize face for attendance.
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        attempt = int(request.form.get('attempt', 1))
        action = request.form.get('action', 'checkin')  # 'checkin' or 'checkout'

        # Process the uploaded image
        image_file = request.files['image']
        image_stream = image_file.read()
        numpy_image = np.frombuffer(image_stream, np.uint8)
        image_cv = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)

        processed_face = load_and_preprocess_image(image_cv, is_path=False)
        if processed_face is None:
            return jsonify({
                "recognized": False,
                "success": False,
                "error": "No face detected in the image",
                "attempt": attempt
            }), 400

        # Get face embedding
        new_embedding = get_embedding(processed_face)
        if new_embedding is None:
            return jsonify({
                "recognized": False,
                "success": False,
                "error": "Could not generate face embedding",
                "attempt": attempt
            }), 400

        # Recognize face from database
        recognized_employee, confidence = recognize_face_from_database(new_embedding, db.session)

        if recognized_employee is not None:
            # Validate attendance time and rules
            is_valid, error_message = validate_attendance_time(action, recognized_employee.id)
            if not is_valid:
                return jsonify({
                    "recognized": True,
                    "success": False,
                    "employee_name": recognized_employee.full_name,
                    "error": error_message,
                    "time_restriction": True,
                    "attempt": attempt
                }), 400

            # Face recognized and time is valid - record attendance
            if action == 'checkin':
                response_data, status_code = record_attendance(recognized_employee.id)
            else:  # checkout
                response_data, status_code = record_checkout(recognized_employee.id)

            # Check if the underlying action (check-in/out) was successful
            if status_code != 200:
                # Ensure error responses are standardized
                error_payload = {
                    "success": False,
                    "recognized": True,
                    "employee_name": recognized_employee.full_name,
                    "error": response_data.get("error", "An unknown error occurred during attendance recording.")
                }
                return jsonify(error_payload), status_code

            # If successful, enrich the response with recognition details
            response_data['success'] = True
            response_data['recognized'] = True
            response_data['employee_name'] = recognized_employee.full_name
            response_data['employee_id'] = recognized_employee.id
            response_data['confidence'] = float(confidence)
            response_data['attempt'] = attempt
            
            return jsonify(response_data), 200
        else:
            # Face not recognized
            return jsonify({
                "recognized": False,
                "success": False,
                "error": f"Face not recognized (confidence: {confidence:.3f})",
                "attempt": attempt,
                "confidence": float(confidence)
            }), 404

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Face recognition error: {str(e)}")
        print(f"❌ Full traceback: {error_details}")
        # attempt may not be defined here, so default to 1
        current_attempt = locals().get('attempt', 1)
        return jsonify({
            "recognized": False,
            "success": False,
            "error": f"Face recognition error: {str(e)}",
            "attempt": current_attempt,
            "details": error_details if app.debug else None
        }), 500

@app.route('/api/attendance/verify-access-code', methods=['POST'])
def verify_access_code_with_photo():
    """
    Verify access code and capture photo for security purposes.
    Used when face recognition fails after 3 attempts.
    """
    try:
        # Get access code from form data
        access_code = request.form.get('access_code', '').strip().upper()
        action = request.form.get('action', 'checkin')  # 'checkin' or 'checkout'

        if not access_code:
            return jsonify({"error": "Access code is required"}), 400

        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "Photo is required for security verification"}), 400

        # Find employee by access code
        employee = EmployeeInfo.query.filter_by(access_code=access_code, status='active').first()

        if not employee:
            return jsonify({"error": "Invalid access code"}), 400

        # Validate attendance time and rules
        is_valid, error_message = validate_attendance_time(action, employee.id)
        if not is_valid:
            return jsonify({"error": error_message, "time_restriction": True}), 400

        # Process and save the security photo
        image_file = request.files['image']
        image_stream = image_file.read()
        numpy_image = np.frombuffer(image_stream, np.uint8)
        image_cv = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)

        # Create security photos directory if it doesn't exist
        security_photos_dir = os.path.join(BASE_PROJECT_DIR, "security_photos")
        os.makedirs(security_photos_dir, exist_ok=True)

        # Create employee-specific directory
        employee_security_dir = os.path.join(security_photos_dir, employee.full_name.replace(" ", "_"))
        os.makedirs(employee_security_dir, exist_ok=True)

        # Save the security photo with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        photo_filename = f"access_code_entry_{timestamp}.jpg"
        photo_path = os.path.join(employee_security_dir, photo_filename)

        # Save the image
        cv2.imwrite(photo_path, image_cv)

        # Generate new access code for security
        session = SessionLocal()
        try:
            new_code = generate_unique_code(session)
            employee.access_code = new_code
            db.session.commit()
        finally:
            session.close()

        # Record attendance based on action
        if action == 'checkin':
            record_attendance(employee.id)
        else:  # checkout
            record_checkout(employee.id)

        return jsonify({
            "success": True,
            "employee_name": employee.full_name,
            "employee_id": employee.id,
            "new_access_code": new_code,
            "action": action,
            "message": f"Welcome {employee.full_name}! Attendance recorded via access code.",
            "security_photo_saved": True,
            "photo_path": photo_filename
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Access code verification failed: {str(e)}"}), 500

@app.route('/api/onboarding/generate-code', methods=['POST'])
def generate_verification_code():
    """Generate a unique verification code for onboarding."""
    try:
        # Generate a random 8-character code
        characters = string.ascii_uppercase + string.digits
        code = ''.join(random.choices(characters, k=8))

        # Create onboarding session
        session_id = f"verify_{int(time.time())}_{random.randint(1000, 9999)}"
        expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()

        onboarding_session = OnboardingSession(
            session_id=session_id,
            verification_code=code,
            expires_at=expires_at,
            status='code_generated'
        )
        db.session.add(onboarding_session)
        db.session.commit()

        return jsonify({
            "code": code,
            "session_id": session_id,
            "expires_in": 600  # 10 minutes in seconds
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to generate code: {str(e)}"}), 500

@app.route('/api/onboarding/verify-code', methods=['POST'])
def verify_onboarding_code():
    """Verify the entered code matches the generated code."""
    try:
        data = request.get_json()
        entered_code = data.get('entered_code', '').strip().upper()
        generated_code = data.get('generated_code', '').strip().upper()

        if not entered_code or not generated_code:
            return jsonify({"verified": False, "error": "Missing codes"}), 400

        if entered_code == generated_code:
            # Find and update the onboarding session
            session = OnboardingSession.query.filter_by(verification_code=generated_code).first()
            if session:
                session.code_verified = 1
                session.status = 'code_verified'
                db.session.commit()

            return jsonify({
                "verified": True,
                "message": "Code verified successfully"
            }), 200
        else:
            return jsonify({
                "verified": False,
                "error": "Code does not match"
            }), 400

    except Exception as e:
        return jsonify({"verified": False, "error": str(e)}), 500

@app.route('/api/onboarding/complete-setup', methods=['POST'])
def complete_onboarding_setup():
    """Complete the onboarding process by saving the final photo and creating the employee."""
    try:
        if 'photo' not in request.files:
            return jsonify({"error": "No photo provided"}), 400

        verification_code = request.form.get('verification_code')
        if not verification_code:
            return jsonify({"error": "Verification code required"}), 400

        # Find the onboarding session
        session = OnboardingSession.query.filter_by(
            verification_code=verification_code,
            code_verified=1
        ).first()

        if not session:
            return jsonify({"error": "Invalid or unverified session"}), 400

        # Check if session has expired
        if datetime.fromisoformat(session.expires_at) < datetime.now():
            return jsonify({"error": "Session has expired"}), 400

        # Save the photo
        photo_file = request.files['photo']
        photo_filename = f"employee_{int(time.time())}_{random.randint(1000, 9999)}.jpg"

        # Create employee photos directory if it doesn't exist
        photos_dir = os.path.join(BASE_PROJECT_DIR, "employee_photos")
        os.makedirs(photos_dir, exist_ok=True)

        photo_path = os.path.join(photos_dir, photo_filename)
        photo_file.save(photo_path)

        # Create the actual employee record
        new_employee = EmployeeInfo(
            full_name=f"New Employee {int(time.time())}",  # This should be updated by HR later
            status='active',
            access_code=generate_unique_code(db.session),
            photo=photo_filename,
            hired_date=date.today().isoformat()
        )
        db.session.add(new_employee)
        db.session.flush()

        # Update the onboarding session
        session.employee_id = new_employee.id
        session.photo_path = photo_path
        session.status = 'completed'

        db.session.commit()

        # Create welcome notification
        create_notification(
            title="Welcome to the Team!",
            message=f"New employee setup completed successfully. Employee ID: EMP{new_employee.id:04d}",
            notification_type='system',
            priority='medium',
            employee_id=new_employee.id
        )

        return jsonify({
            "success": True,
            "employee_id": new_employee.id,
            "access_code": new_employee.access_code,
            "message": "Onboarding completed successfully"
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Setup completion failed: {str(e)}"}), 500

@app.route('/api/onboarding/employee-info', methods=['GET'])
def get_onboarding_employee_info():
    """Get the most recently created employee info for the completion page."""
    try:
        # Get the most recent employee (last onboarded)
        latest_employee = EmployeeInfo.query.filter_by(status='active').order_by(EmployeeInfo.id.desc()).first()

        if not latest_employee:
            return jsonify({"error": "No employee found"}), 404

        return jsonify({
            "id": latest_employee.id,
            "full_name": latest_employee.full_name,
            "email": latest_employee.email,
            "position": latest_employee.position,
            "department": latest_employee.department,
            "access_code": latest_employee.access_code,
            "employee_id": f"EMP{latest_employee.id:04d}",
            "hired_date": latest_employee.hired_date,
            "status": latest_employee.status
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/onboarding/verify-old-code', methods=['POST'])
def verify_old_code():
    """Verify the employee's current access code and generate a new one."""
    try:
        data = request.get_json()
        old_code = data.get('old_code', '').strip().upper()

        if not old_code:
            return jsonify({"error": "Access code is required"}), 400

        # Find employee with this access code
        employee = EmployeeInfo.query.filter_by(access_code=old_code, status='active').first()

        if not employee:
            return jsonify({"error": "Invalid access code"}), 400

        # Generate new access code
        session = SessionLocal()
        try:
            new_code = generate_unique_code(session)

            # Update employee's access code
            employee.access_code = new_code
            db.session.commit()

            # Record attendance for successful access code verification
            response_data, status_code = record_attendance(employee.id)

            return jsonify({
                "success": True,
                "new_code": new_code,
                "employee_name": employee.full_name,
                "attendance_recorded": True,
                "message": f"Welcome {employee.full_name}! Attendance recorded successfully."
            }), 200

        finally:
            session.close()

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500

@app.route('/api/attendance/generate-new-code', methods=['POST'])
def generate_new_code_for_employee():
    """Generate a new access code for an employee after successful face recognition."""
    try:
        data = request.get_json()
        employee_name = data.get('employee_name', '').strip()

        if not employee_name:
            return jsonify({"error": "Employee name is required"}), 400

        # Find employee by name
        employee = EmployeeInfo.query.filter_by(full_name=employee_name, status='active').first()

        if not employee:
            return jsonify({"error": "Employee not found"}), 400

        # Generate new access code
        session = SessionLocal()
        try:
            new_code = generate_unique_code(session)

            # Update employee's access code
            employee.access_code = new_code
            db.session.commit()

            return jsonify({
                "success": True,
                "new_code": new_code,
                "employee_name": employee.full_name,
                "message": f"New access code generated for {employee.full_name}"
            }), 200

        finally:
            session.close()

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Code generation failed: {str(e)}"}), 500

@app.route('/api/attendance/history', methods=['GET'])
def get_attendance_history():
    """
    Get attendance history with filtering and pagination.
    Query parameters:
    - employee_id: Filter by specific employee
    - start_date: Filter from date (YYYY-MM-DD)
    - end_date: Filter to date (YYYY-MM-DD)
    - status: Filter by status (Present, Absent, Late)
    - page: Page number (default: 1)
    - per_page: Records per page (default: 50)
    """
    try:
        # Get query parameters
        employee_id = request.args.get('employee_id', type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        status = request.args.get('status')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)

        # Build query
        query = db.session.query(Attendance, EmployeeInfo).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        )

        # Apply filters
        if employee_id:
            query = query.filter(EmployeeInfo.id == employee_id)

        if start_date:
            query = query.filter(Attendance.date >= start_date)

        if end_date:
            query = query.filter(Attendance.date <= end_date)

        if status:
            query = query.filter(Attendance.status == status)

        # Order by date descending (most recent first)
        query = query.order_by(Attendance.date.desc(), Attendance.check_in.desc())

        # Paginate
        total = query.count()
        records = query.offset((page - 1) * per_page).limit(per_page).all()

        # Format results
        result = []
        for attendance, employee in records:
            # Calculate work duration if both check-in and check-out exist
            work_duration = None
            if attendance.check_in and attendance.check_out:
                try:
                    checkin_time = datetime.strptime(attendance.check_in, '%H:%M:%S')
                    checkout_time = datetime.strptime(attendance.check_out, '%H:%M:%S')
                    duration = checkout_time - checkin_time
                    hours = duration.seconds // 3600
                    minutes = (duration.seconds % 3600) // 60
                    work_duration = f"{hours}h {minutes}m"
                except:
                    work_duration = "Invalid"

            # Determine if late and calculate minutes based on settings
            is_late = attendance.is_late == 1
            late_minutes = 0
            if is_late and attendance.check_in:
                try:
                    # Fetch settings to get the correct work start time
                    settings = SystemSetting.query.get(1)
                    work_start_str = settings.work_start_time if settings and settings.work_start_time else '08:00:00'
                    
                    # [إصلاح] التعامل مع تنسيقات الوقت المختلفة (مع أو بدون الثواني)
                    if len(work_start_str.split(':')) == 2:  # إذا كان التنسيق HH:MM فقط
                        work_start_str += ':00'  # إضافة الثواني

                    checkin_time_obj = datetime.strptime(attendance.check_in, '%H:%M:%S')
                    work_start_time_obj = datetime.strptime(work_start_str, '%H:%M:%S')
                    
                    if checkin_time_obj > work_start_time_obj:
                        late_duration = checkin_time_obj - work_start_time_obj
                        late_minutes = late_duration.seconds // 60
                except Exception as e:
                    print(f"Could not calculate late minutes for history: {e}")
                    pass # Keep late_minutes as 0 if calculation fails

            result.append({
                "id": attendance.id,
                "employee_id": employee.id,
                "employee_name": employee.full_name,
                "employee_code": f"EMP{employee.id:04d}",
                "department": employee.department,
                "position": employee.position,
                "date": attendance.date,
                "check_in": attendance.check_in,
                "check_out": attendance.check_out,
                "status": attendance.status,
                "is_late": is_late,
                "late_minutes": late_minutes if is_late else 0,
                "work_duration": work_duration,
                "created_at": attendance.created_at
            })

        return jsonify({
            "success": True,
            "data": result,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch attendance history: {str(e)}"}), 500

@app.route('/api/attendance/summary', methods=['GET'])
def get_attendance_summary():
    """
    Get attendance summary statistics.
    Query parameters:
    - start_date: Filter from date (YYYY-MM-DD)
    - end_date: Filter to date (YYYY-MM-DD)
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Build base query
        query = db.session.query(Attendance)

        if start_date:
            query = query.filter(Attendance.date >= start_date)

        if end_date:
            query = query.filter(Attendance.date <= end_date)

        # Get counts by status
        total_records = query.count()
        present_count = query.filter(Attendance.status == 'Present').count()
        absent_count = query.filter(Attendance.status == 'Absent').count()
        late_count = query.filter(Attendance.is_late == 1).count()

        # Get unique employees count
        unique_employees = db.session.query(Attendance.employee_info_id).distinct().count()

        return jsonify({
            "success": True,
            "summary": {
                "total_records": total_records,
                "present_count": present_count,
                "absent_count": absent_count,
                "late_count": late_count,
                "unique_employees": unique_employees,
                "attendance_rate": round((present_count / total_records * 100) if total_records > 0 else 0, 2)
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch attendance summary: {str(e)}"}), 500

@app.route('/api/attendance/export', methods=['GET'])
def export_attendance_data():
    """
    Export attendance data as CSV.
    Same query parameters as history endpoint.
    """
    try:
        # Get query parameters
        employee_id = request.args.get('employee_id', type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        status = request.args.get('status')

        # Build query
        query = db.session.query(Attendance, EmployeeInfo).join(
            EmployeeInfo, Attendance.employee_info_id == EmployeeInfo.id
        )

        # Apply filters
        if employee_id:
            query = query.filter(EmployeeInfo.id == employee_id)

        if start_date:
            query = query.filter(Attendance.date >= start_date)

        if end_date:
            query = query.filter(Attendance.date <= end_date)

        if status:
            query = query.filter(Attendance.status == status)

        # Order by date descending
        query = query.order_by(Attendance.date.desc(), Attendance.check_in.desc())

        # Get all records (no pagination for export)
        records = query.all()

        # Create CSV content
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Employee ID', 'Employee Name', 'Employee Code', 'Department', 'Position',
            'Date', 'Check In', 'Check Out', 'Work Duration', 'Status', 'Late Minutes'
        ])

        # Write data
        for attendance, employee in records:
            # Calculate work duration
            work_duration = ''
            if attendance.check_in and attendance.check_out:
                try:
                    checkin_time = datetime.strptime(attendance.check_in, '%H:%M:%S')
                    checkout_time = datetime.strptime(attendance.check_out, '%H:%M:%S')
                    duration = checkout_time - checkin_time
                    hours = duration.seconds // 3600
                    minutes = (duration.seconds % 3600) // 60
                    work_duration = f"{hours}h {minutes}m"
                except:
                    work_duration = "Invalid"

            # Calculate late minutes based on settings
            late_minutes = 0
            if attendance.is_late == 1 and attendance.check_in:
                try:
                    # Fetch settings to get the correct work start time
                    settings = SystemSetting.query.get(1)
                    work_start_str = settings.work_start_time if settings and settings.work_start_time else '08:00:00'
                    
                    # [إصلاح] التعامل مع تنسيقات الوقت المختلفة (مع أو بدون الثواني)
                    if len(work_start_str.split(':')) == 2:  # إذا كان التنسيق HH:MM فقط
                        work_start_str += ':00'  # إضافة الثواني

                    checkin_time_obj = datetime.strptime(attendance.check_in, '%H:%M:%S')
                    work_start_time_obj = datetime.strptime(work_start_str, '%H:%M:%S')

                    if checkin_time_obj > work_start_time_obj:
                        late_duration = checkin_time_obj - work_start_time_obj
                        late_minutes = late_duration.seconds // 60
                except Exception as e:
                    print(f"Could not calculate late minutes for export: {e}")
                    pass # Keep late_minutes as 0 if calculation fails

            writer.writerow([
                employee.id,
                employee.full_name,
                f"EMP{employee.id:04d}",
                employee.department or '',
                employee.position or '',
                attendance.date,
                attendance.check_in or '',
                attendance.check_out or '',
                work_duration,
                attendance.status,
                late_minutes
            ])

        # Prepare response
        output.seek(0)
        
        # Get the CSV data as a string
        csv_data = output.getvalue()

        from flask import make_response
        
        # Encode the response to UTF-8 with BOM (Byte Order Mark) for Excel compatibility
        response = make_response(csv_data.encode('utf-8-sig'))
        
        # Set headers for CSV download
        response.headers['Content-Type'] = 'text/csv; charset=utf-8-sig'
        response.headers['Content-Disposition'] = f'attachment; filename=attendance_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        return response

    except Exception as e:
        return jsonify({"error": f"Failed to export attendance data: {str(e)}"}), 500

@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """
    [MODIFIED] - Get current user's full profile information, combining user and employee data.
    """
    try:
        current_user_email = get_jwt_identity()
        user = User.query.filter_by(Email=current_user_email).first()

        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404

        # Start with base user data
        profile_data = {
            "id": user.Id,
            "email": user.Email,
            "role": user.Role,
            "account_created": user.created_at,
            "last_login": user.last_login,
            "full_name": user.Email, # Fallback name
            "first_name": "",
            "last_name": "",
            "access_code": "N/A", # Changed from phone to access_code
            "department": "N/A",
            "position": "N/A",
            "employee_id": "N/A",
            "join_date": None,
            "avatar_url": "/static/default-avatar.svg",
            # Removed bio, address, emergency_contact
        }

        # If linked to an employee, enrich with employee data
        if user.employee_id:
            employee = EmployeeInfo.query.get(user.employee_id)
            if employee:
                name_parts = employee.full_name.split(' ', 1)
                first_name = name_parts[0]
                last_name = name_parts[1] if len(name_parts) > 1 else ""

                department_name = employee.department
                if employee.department_id:
                    dept = Department.query.get(employee.department_id)
                    if dept:
                        department_name = dept.name

                profile_data.update({
                    "full_name": employee.full_name,
                    "first_name": first_name,
                    "last_name": last_name,
                    "access_code": employee.access_code or "N/A", # Changed from phone to access_code
                    "department": department_name,
                    "position": employee.position or "",
                    "employee_id": f"EMP{employee.id:04d}",
                    "join_date": employee.hired_date,
                    "avatar_url": f"/static/profile_pics/{employee.photo}" if employee.photo else "/static/default-avatar.svg",
                    # Removed bio, address, emergency_contact
                })

        return jsonify({
            "success": True,
            "profile": profile_data
        }), 200

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": f"Failed to fetch profile: {str(e)}"}), 500

@app.route('/api/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """
    [MODIFIED] - Update current user's profile information.
    """
    try:
        current_user_email = get_jwt_identity()
        user = User.query.filter_by(Email=current_user_email).first()

        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404

        if not user.employee_id:
            return jsonify({"success": False, "error": "This account is not linked to an employee profile."}),

        employee = EmployeeInfo.query.get(user.employee_id)
        if not employee:
            return jsonify({"success": False, "error": "Associated employee profile not found"}), 404

        data = request.get_json()

        # Update EmployeeInfo model
        if 'first_name' in data and 'last_name' in data:
            employee.full_name = f"{data['first_name']} {data['last_name']}".strip()
        
        # Only update fields that are still present in the form
        if 'position' in data: employee.position = data['position']
        
        # Removed updates for phone, address, bio, emergency_contact as they are removed from frontend
        # Note: Department is not updated here as it's a more complex operation (linked table)
        # Note: Email is not updated here to prevent de-sync with user account identity

        db.session.commit()
        
        # Fetch the updated profile to return it
        # Re-using the logic from get_profile
        updated_user = User.query.filter_by(Email=current_user_email).first()
        updated_employee = EmployeeInfo.query.get(updated_user.employee_id)
        
        name_parts = updated_employee.full_name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        department_name = updated_employee.department
        if updated_employee.department_id:
            dept = Department.query.get(updated_employee.department_id)
            if dept:
                department_name = dept.name

        updated_profile_data = {
            "full_name": updated_employee.full_name,
            "first_name": first_name,
            "last_name": last_name,
            "access_code": updated_employee.access_code or "N/A", # Changed from phone to access_code
            "department": department_name,
            "position": updated_employee.position or "",
            # Removed bio, address, emergency_contact
        }

        return jsonify({
            "success": True,
            "message": "Profile updated successfully",
            "profile": updated_profile_data
        }), 200

    except Exception as e:
        db.session.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": f"Failed to update profile: {str(e)}"}), 500

@app.route('/api/profile/password', methods=['PUT'])
@jwt_required()
def change_password():
    """
    [MODIFIED] - Change the current user's (employee's) password.
    Requires current password, new password, and new password confirmation.
    """
    try:
        # Get the user identity from the JWT token
        current_user_email = get_jwt_identity()
        user = User.query.filter_by(Email=current_user_email).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        data = request.get_json()

        # التحقق من الحقول الإلزامية
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')

        if not all([current_password, new_password, confirm_password]):
            return jsonify({"error": "جميع حقول كلمة المرور مطلوبة"}), 400

        # التحقق من تطابق كلمة المرور الجديدة وتأكيدها
        if new_password != confirm_password:
            return jsonify({"error": "كلمة المرور الجديدة وتأكيدها غير متطابقين"}), 400

        # التحقق من طول كلمة المرور الجديدة (يمكن تعديل هذا الشرط حسب السياسة)
        if len(new_password) < 6: # يمكن تغيير هذا الرقم لسياسة أقوى
            return jsonify({"error": "كلمة المرور الجديدة يجب أن تكون 6 أحرف على الأقل"}), 400

        # التحقق من صحة كلمة المرور الحالية
        if not bcrypt.check_password_hash(user.Password, current_password):
            return jsonify({"error": "كلمة المرور الحالية غير صحيحة"}), 401

        # تشفير كلمة المرور الجديدة وتحديثها في قاعدة البيانات
        user.Password = bcrypt.generate_password_hash(new_password)
        db.session.commit()

        return jsonify({"success": True, "message": "تم تغيير كلمة المرور بنجاح"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"فشل في تغيير كلمة المرور: {str(e)}"}), 500

@app.route('/api/profile/avatar', methods=['POST'])
def upload_avatar():
    """
    Upload user avatar image.
    """
    try:
        if 'avatar' not in request.files:
            return jsonify({"error": "No avatar file provided"}), 400

        file = request.files['avatar']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not ('.' in file.filename and
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file type. Only PNG, JPG, JPEG, and GIF are allowed"}), 400

        # In a real application, you would:
        # 1. Save file to storage (local/cloud)
        # 2. Resize/optimize image
        # 3. Update database with new avatar URL
        # For demo purposes, we'll return a placeholder URL

        avatar_url = "/static/default-avatar.svg"

        return jsonify({
            "success": True,
            "message": "Avatar uploaded successfully",
            "avatar_url": avatar_url
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to upload avatar: {str(e)}"}), 500

@app.route('/api/departments', methods=['GET'])
def get_departments():
    """
    Get all departments with optional filtering and pagination.
    Query parameters:
    - search: Search by department name
    - active_only: Filter only active departments (true/false)
    - page: Page number (default: 1)
    - per_page: Records per page (default: 50)
    """
    try:
        # Get query parameters
        search = request.args.get('search', '').strip()
        active_only = request.args.get('active_only', 'false').lower() == 'true'
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)

        # Build query
        query = Department.query

        # Apply filters
        if search:
            query = query.filter(Department.name.ilike(f'%{search}%'))

        # Order by name
        query = query.order_by(Department.name.asc())

        # Get total count for pagination
        total = query.count()

        # Apply pagination
        departments = query.offset((page - 1) * per_page).limit(per_page).all()

        # Format results
        result = []
        for dept in departments:
            # Count employees in this department
            employee_count = EmployeeInfo.query.filter_by(department_id=dept.id, status='active').count()

            result.append({
                "id": dept.id,
                "name": dept.name,
                "employee_count": employee_count
            })

        return jsonify({
            "success": True,
            "departments": result,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch departments: {str(e)}"}), 500

@app.route('/api/departments', methods=['POST'])
def create_department():
    """
    Create a new department.
    """
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('name'):
            return jsonify({"error": "Department name is required"}), 400

        # Check if department name already exists
        existing_dept = Department.query.filter_by(name=data['name']).first()
        if existing_dept:
            return jsonify({"error": "Department name already exists"}), 400

        # Create new department
        department = Department(name=data['name'])

        db.session.add(department)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Department created successfully",
            "department": {
                "id": department.id,
                "name": department.name,
                "employee_count": 0
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to create department: {str(e)}"}), 500

@app.route('/api/departments/<int:department_id>', methods=['GET'])
def get_department(department_id):
    """
    Get a specific department by ID.
    """
    try:
        department = Department.query.get_or_404(department_id)

        # Count employees in this department
        employee_count = EmployeeInfo.query.filter_by(department_id=department.id, status='active').count()

        return jsonify({
            "success": True,
            "department": {
                "id": department.id,
                "name": department.name,
                "employee_count": employee_count
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch department: {str(e)}"}), 500

@app.route('/api/departments/<int:department_id>', methods=['PUT'])
def update_department(department_id):
    """
    Update a department.
    """
    try:
        department = Department.query.get_or_404(department_id)
        data = request.get_json()

        # Validate required fields
        if not data.get('name'):
            return jsonify({"error": "Department name is required"}), 400

        # Check if department name already exists (excluding current department)
        existing_dept = Department.query.filter(
            Department.name == data['name'],
            Department.id != department_id
        ).first()
        if existing_dept:
            return jsonify({"error": "Department name already exists"}), 400

        # Update department fields
        department.name = data['name']

        db.session.commit()

        # Count employees in this department
        employee_count = EmployeeInfo.query.filter_by(department_id=department.id, status='active').count()

        return jsonify({
            "success": True,
            "message": "Department updated successfully",
            "department": {
                "id": department.id,
                "name": department.name,
                "employee_count": employee_count
            }
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to update department: {str(e)}"}), 500

@app.route('/api/departments/<int:department_id>', methods=['DELETE'])
def delete_department(department_id):
    """
    Delete a department (soft delete by setting is_active to False).
    """
    try:
        department = Department.query.get_or_404(department_id)

        # Check if department has active employees
        employee_count = EmployeeInfo.query.filter_by(department_id=department.id, status='active').count()
        if employee_count > 0:
            return jsonify({
                "error": f"Cannot delete department. It has {employee_count} active employees. Please reassign employees first."
            }), 400

        # Hard delete the department
        db.session.delete(department)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Department deleted successfully"
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete department: {str(e)}"}), 500

@app.route('/api/departments/<int:department_id>/employees', methods=['GET'])
def get_department_employees(department_id):
    """
    Get all employees in a specific department.
    """
    try:
        department = Department.query.get_or_404(department_id)

        employees = EmployeeInfo.query.filter_by(department_id=department_id, status='active').all()

        result = []
        for emp in employees:
            result.append({
                "id": emp.id,
                "full_name": emp.full_name,
                "email": emp.email,
                "position": emp.position,
                "hired_date": emp.hired_date,
                "access_code": emp.access_code
            })

        return jsonify({
            "success": True,
            "department": {
                "id": department.id,
                "name": department.name
            },
            "employees": result,
            "total_employees": len(result)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch department employees: {str(e)}"}), 500

@app.route('/api/departments/stats', methods=['GET'])
def get_departments_stats():
    """
    Get department statistics for dashboard.
    """
    try:
        total_departments = Department.query.count()
        total_employees = EmployeeInfo.query.filter_by(status='active').count()

        # Get departments with employee counts
        departments_with_counts = db.session.query(
            Department.name,
            db.func.count(EmployeeInfo.id).label('employee_count')
        ).outerjoin(
            EmployeeInfo,
            (Department.id == EmployeeInfo.department_id) & (EmployeeInfo.status == 'active')
        ).group_by(Department.id, Department.name).all()

        # Find largest and smallest departments
        largest_dept = max(departments_with_counts, key=lambda x: x.employee_count) if departments_with_counts else None
        smallest_dept = min(departments_with_counts, key=lambda x: x.employee_count) if departments_with_counts else None

        return jsonify({
            "success": True,
            "stats": {
                "total_departments": total_departments,
                "total_employees": total_employees,
                "avg_employees_per_dept": round(total_employees / total_departments, 1) if total_departments > 0 else 0,
                "largest_department": {
                    "name": largest_dept.name,
                    "employee_count": largest_dept.employee_count
                } if largest_dept else None,
                "smallest_department": {
                    "name": smallest_dept.name,
                    "employee_count": smallest_dept.employee_count
                } if smallest_dept else None,
                "departments_breakdown": [
                    {
                        "name": dept.name,
                        "employee_count": dept.employee_count
                    } for dept in departments_with_counts
                ]
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch department stats: {str(e)}"}), 500

@app.route('/api/departments/list', methods=['GET'])
def get_departments_list():
    """
    Get simple list of departments for dropdowns/selects.
    Returns only id and name for each department.
    """
    try:
        departments = Department.query.order_by(Department.name.asc()).all()

        result = []
        for dept in departments:
            result.append({
                "id": dept.id,
                "name": dept.name
            })

        return jsonify({
            "success": True,
            "departments": result
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch departments list: {str(e)}"}), 500

@app.route('/api/employees/<int:employee_id>/face-embeddings', methods=['GET'])
def get_employee_face_embeddings(employee_id):
    """
    Get face embedding information for an employee.
    """
    try:
        # Check if employee exists
        employee = EmployeeInfo.query.get(employee_id)
        if not employee:
            return jsonify({"error": "Employee not found"}), 404

        # Get face embeddings
        embeddings = FaceEmbedding.query.filter_by(employee_info_id=employee_id).all()

        embedding_info = []
        for emb in embeddings:
            embedding_info.append({
                "id": emb.embedding_id,
                "image_source": emb.image_source_filename,
                "created_at": "N/A"  # Add timestamp if available
            })

        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "employee_name": employee.full_name,
            "embedding_count": len(embeddings),
            "embeddings": embedding_info
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch face embeddings: {str(e)}"}), 500

@app.route('/api/employees/<int:employee_id>/face-embeddings', methods=['DELETE'])
def delete_employee_face_embeddings(employee_id):
    """
    Delete all face embeddings for an employee (for re-registration).
    """
    try:
        # Check if employee exists
        employee = EmployeeInfo.query.get(employee_id)
        if not employee:
            return jsonify({"error": "Employee not found"}), 404

        # Delete all face embeddings for this employee
        deleted_count = FaceEmbedding.query.filter_by(employee_info_id=employee_id).delete()
        db.session.commit()

        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "employee_name": employee.full_name,
            "deleted_embeddings": deleted_count,
            "message": f"Deleted {deleted_count} face embeddings for {employee.full_name}"
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete face embeddings: {str(e)}"}), 500

@app.route('/api/employees/<int:employee_id>/face-registration/start', methods=['POST'])
def start_face_registration(employee_id):
    """
    Start face registration process for an employee.
    Creates a temporary session to track the registration progress.
    """
    try:
        # Check if employee exists
        employee = EmployeeInfo.query.get(employee_id)
        if not employee:
            return jsonify({"error": "Employee not found"}), 404

        # Check if employee already has face embeddings
        existing_embeddings = FaceEmbedding.query.filter_by(employee_info_id=employee_id).count()

        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "employee_name": employee.full_name,
            "existing_embeddings": existing_embeddings,
            "session_id": f"face_reg_{employee_id}_{int(time.time())}",
            "required_photos": 8,  # Minimum photos needed
            "message": "Face registration session started"
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to start face registration: {str(e)}"}), 500

@app.route('/api/employees/<int:employee_id>/face-registration/capture', methods=['POST'])
def capture_face_photo(employee_id):
    """
    Capture and process a single face photo for employee registration.
    Validates face detection and generates embedding.
    """
    try:
        # Check if employee exists
        employee = EmployeeInfo.query.get(employee_id)
        if not employee:
            return jsonify({"error": "Employee not found"}), 404

        # Check if image was provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        # Get additional parameters
        photo_index = request.form.get('photo_index', 1, type=int)
        session_id = request.form.get('session_id', '')

        # Read image data
        image_data = image_file.read()

        # Convert to PIL Image
        from PIL import Image
        import io
        pil_image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array for face detection
        import numpy as np
        image_array = np.array(pil_image)

        # Detect face using MTCNN
        try:
            faces = detector.detect_faces(image_array)

            if not faces:
                return jsonify({
                    "success": False,
                    "error": "No face detected in the image",
                    "quality_score": 0.0,
                    "photo_index": photo_index
                }), 400

            # Get the face with highest confidence
            best_face = max(faces, key=lambda x: x['confidence'])
            confidence = best_face['confidence']

            # Check if confidence is acceptable
            if confidence < 0.90:  # Require high confidence for registration
                return jsonify({
                    "success": False,
                    "error": f"Face detection confidence too low: {confidence:.3f}. Please ensure good lighting and clear face visibility.",
                    "quality_score": confidence,
                    "photo_index": photo_index
                }), 400

            # Extract face region
            box = best_face['box']
            x, y, w, h = box

            # Add padding around face
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_array.shape[1] - x, w + 2 * padding)
            h = min(image_array.shape[0] - y, h + 2 * padding)

            face_image = image_array[y:y+h, x:x+w]

            # Generate face embedding
            face_embedding = generate_face_embedding(face_image)

            if face_embedding is None:
                return jsonify({
                    "success": False,
                    "error": "Failed to generate face embedding",
                    "quality_score": confidence,
                    "photo_index": photo_index
                }), 400

            # Save image to temporary location
            import tempfile
            import os

            # Create temp directory for this session if it doesn't exist
            temp_dir = os.path.join(tempfile.gettempdir(), f"face_registration_{employee_id}")
            os.makedirs(temp_dir, exist_ok=True)

            # Save the original image
            image_filename = f"photo_{photo_index}_{int(time.time())}.jpg"
            image_path = os.path.join(temp_dir, image_filename)
            pil_image.save(image_path, 'JPEG', quality=95)

            return jsonify({
                "success": True,
                "photo_index": photo_index,
                "quality_score": confidence,
                "face_detected": True,
                "embedding_generated": True,
                "image_filename": image_filename,
                "temp_path": image_path,
                "message": f"Photo {photo_index} captured successfully with {confidence:.3f} confidence"
            }), 200

        except Exception as face_error:
            return jsonify({
                "success": False,
                "error": f"Face processing error: {str(face_error)}",
                "quality_score": 0.0,
                "photo_index": photo_index
            }), 400

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Face capture error: {str(e)}")
        print(f"❌ Full traceback: {error_details}")

        return jsonify({
            "success": False,
            "error": f"Face capture failed: {str(e)}",
            "details": error_details if app.debug else None
        }), 500

@app.route('/api/employees/<int:employee_id>/face-registration/complete', methods=['POST'])
def complete_face_registration(employee_id):
    """
    Complete face registration by saving all captured embeddings to database.
    """
    try:
        # Check if employee exists
        employee = EmployeeInfo.query.get(employee_id)
        if not employee:
            return jsonify({"error": "Employee not found"}), 404

        # Get session data
        data = request.get_json()
        session_id = data.get('session_id', '')
        captured_photos = data.get('captured_photos', [])

        if len(captured_photos) < 8:
            return jsonify({"error": "Minimum 5 photos required for registration"}), 400

        # Process each captured photo and save embeddings
        saved_embeddings = 0
        temp_dir = os.path.join(tempfile.gettempdir(), f"face_registration_{employee_id}")

        for photo_data in captured_photos:
            try:
                image_filename = photo_data.get('image_filename')
                image_path = photo_data.get('temp_path')

                if not image_path or not os.path.exists(image_path):
                    continue

                # Load and process the image again to generate final embedding
                from PIL import Image
                import numpy as np

                pil_image = Image.open(image_path)
                image_array = np.array(pil_image)

                # Detect face and generate embedding
                faces = detector.detect_faces(image_array)
                if not faces:
                    continue

                best_face = max(faces, key=lambda x: x['confidence'])
                box = best_face['box']
                x, y, w, h = box

                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image_array.shape[1] - x, w + 2 * padding)
                h = min(image_array.shape[0] - y, h + 2 * padding)

                face_image = image_array[y:y+h, x:x+w]
                face_embedding = generate_face_embedding(face_image)

                if face_embedding is not None:
                    # Save embedding to database
                    embedding_record = FaceEmbedding(
                        employee_info_id=employee_id,
                        embedding=face_embedding.tobytes(),
                        image_source_filename=image_filename
                    )
                    db.session.add(embedding_record)
                    saved_embeddings += 1

            except Exception as photo_error:
                print(f"⚠️ Error processing photo {photo_data}: {photo_error}")
                continue

        # Commit all embeddings
        db.session.commit()

        # Clean up temporary files
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")

        if saved_embeddings == 0:
            return jsonify({"error": "No valid face embeddings could be generated"}), 400

        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "employee_name": employee.full_name,
            "embeddings_saved": saved_embeddings,
            "total_photos_processed": len(captured_photos),
            "message": f"Face registration completed successfully with {saved_embeddings} embeddings"
        }), 200

    except Exception as e:
        db.session.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Face registration completion error: {str(e)}")
        print(f"❌ Full traceback: {error_details}")

        return jsonify({
            "error": f"Failed to complete face registration: {str(e)}",
            "details": error_details if app.debug else None
        }), 500

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_current_user_profile():
    try:
        current_user_email = get_jwt_identity()
        user = User.query.filter_by(Email=current_user_email).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        # If user is linked to an employee, use the employee's info
        if user.employee_id:
            employee = EmployeeInfo.query.get(user.employee_id)
            if employee:
                return jsonify({
                    "full_name": employee.full_name,
                    "profile_pic": employee.photo,
                    "avatar_url": f"/static/profile_pics/{employee.photo}" if employee.photo else "/static/default-avatar.svg"
                }), 200

        # Fallback for users not linked to an employee (e.g., admin)
        # Display a role-based name instead of the email address
        fallback_name = "User"
        if user.Role == 'admin':
            fallback_name = "Admin"
        
        return jsonify({
            "full_name": fallback_name,
            "profile_pic": None, # No specific photo for unlinked users
            "avatar_url": "/static/default-avatar.svg"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/employees', methods=['POST'])
@admin_required
def create_employee():
    """
    Create a new employee with profile picture.
    Handles multipart/form-data.
    """
    try:
        # Validate required form fields
        required_fields = ['full_name', 'email', 'department_id', 'position']
        for field in required_fields:
            if field not in request.form or not request.form[field].strip():
                return jsonify({"error": f"Missing or empty required field: {field}"}), 400

        # Extract and clean data from the form
        full_name = request.form['full_name'].strip()
        email = request.form['email'].strip().lower()
        department_id = request.form.get('department_id')
        position = request.form.get('position', '').strip()
        hired_date = request.form.get('hired_date', '').strip() or None

        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return jsonify({"error": "Invalid email format"}), 400

        # Check if email already exists
        if EmployeeInfo.query.filter_by(email=email).first():
            return jsonify({"error": "Email address already exists"}), 409

        # Validate department
        department = Department.query.get(department_id)
        if not department:
            return jsonify({"error": f"Department with ID {department_id} does not exist"}), 400

        # Generate unique access code
        access_code = generate_unique_code(db.session)

        # Create new employee record (without photo initially)
        new_employee = EmployeeInfo(
            full_name=full_name,
            email=email,
            department_id=department.id,
            department=department.name,
            position=position,
            hired_date=hired_date,
            access_code=access_code,
            status='active'
        )
        
        # Handle photo upload
        if 'profile_pic' in request.files:
            photo_file = request.files['profile_pic']
            if photo_file and photo_file.filename != '':
                allowed_extensions = {'png', 'jpg', 'jpeg'}
                if '.' in photo_file.filename and photo_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    # Save the employee first to get an ID
                    db.session.add(new_employee)
                    db.session.flush() # This assigns an ID to new_employee

                    # Generate a unique filename using the new employee's ID
                    photo_filename = f"employee_{new_employee.id}_{int(time.time())}.{photo_file.filename.rsplit('.', 1)[1].lower()}"
                    
                    # Define the correct save directory
                    photos_dir = os.path.join(BASE_PROJECT_DIR, "static", "profile_pics")
                    os.makedirs(photos_dir, exist_ok=True)
                    
                    # Save the file
                    photo_path = os.path.join(photos_dir, photo_filename)
                    photo_file.save(photo_path)
                    
                    # Update the employee record with the photo filename
                    new_employee.photo = photo_filename
                else:
                    return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed"}), 400

        # Add to session and commit
        db.session.add(new_employee)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Employee added successfully!",
            "employee_id": new_employee.id
        }), 201

    except Exception as e:
        db.session.rollback()
        import traceback
        print(f"Error creating employee: {traceback.format_exc()}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Other endpoints like check-in, check-out, etc., can be added here as needed.

# ==============================================================================
#                               MAIN PROGRAM EXECUTION
# ==============================================================================

# --- Decorator for Page Access Control ---
# --- Decorator for Page Access Control ---
# This decorator is now simplified as the @app.before_request handles initial token check
def page_access_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # The @app.before_request will handle the initial redirect if no token or invalid token.
        # If we reach here, it means a valid token was found and g.user_role is set.
        role = getattr(g, 'user_role', None)
        
        if not role:
            # This case should ideally be handled by @app.before_request, but as a fallback
            return redirect(url_for('login_page'))

        requested_path = request.path
        
        # Admin access
        if role == 'admin':
            return f(role=role, *args, **kwargs)
        
        # Employee access
        elif role == 'employee':
            allowed_employee_paths = ['/employee/dashboard', '/profile']
            if requested_path in allowed_employee_paths:
                return f(role=role, *args, **kwargs)
            else:
                # Employee trying to access a non-allowed page
                return redirect(url_for('employee_dashboard'))
        
        # Should not be reached if @app.before_request works correctly
        return redirect(url_for('login_page'))
            
    return decorated_function


@app.route('/login')
def login_page():
    """
    Renders the login.html page.
    """
    return render_template('login.html')

@app.route('/logout')
def logout_page():
    """
    Logs out the user by unsetting JWT cookies and redirecting to login.
    """
    response = make_response(redirect(url_for('login_page')))
    unset_jwt_cookies(response)
    return response

@app.before_request
def load_user_role_for_templates():
    """
    Loads user role from JWT (if present) into Flask's `g` object for template access.
    This runs before every request.
    """
    g.user_role = None # Default to None
    try:
        # Verify JWT in request. This will look in headers and cookies.
        # Set optional=True so it doesn't raise an error if no token is found.
        verify_jwt_in_request(optional=True)
        claims = get_jwt()
        if claims:
            g.user_role = claims.get("role")
    except Exception as e:
        # Log the error but don't prevent request from proceeding
        print(f"Error verifying JWT in before_request: {e}")
        g.user_role = None # Ensure role is None if token is invalid

# --- Decorator for Page Access Control ---
def page_access_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        role = getattr(g, 'user_role', None)
        requested_path = request.path
        
        # If no role is found (not logged in or invalid token), redirect to login
        if not role:
            return redirect(url_for('login_page'))

        # Admin access
        if role == 'admin':
            return f(role=role, *args, **kwargs)
        
        # Employee access
        elif role == 'employee':
            allowed_employee_paths = ['/employee/dashboard', '/profile']
            if requested_path in allowed_employee_paths:
                return f(role=role, *args, **kwargs)
            else:
                # Employee trying to access a non-allowed page
                return redirect(url_for('employee_dashboard'))
        
        # Should not be reached if roles are properly defined
        return redirect(url_for('login_page'))
            
    return decorated_function

@app.route('/dashboard')
@page_access_required
def dashboard_redirect(role):
    """
    Redirects to appropriate dashboard based on user role.
    """
    if role == 'admin':
        return render_template('dashboard.html', role=role)
    elif role == 'employee':
        return render_template('employee_dashboard.html', role=role)
    # Fallback, though page_access_required should handle this
    return redirect(url_for('login_page'))

@app.route('/employee/dashboard')
@page_access_required
def employee_dashboard(role):
    """
    Renders the employee dashboard page.
    """
    return render_template('employee_dashboard.html', role=role)

@app.route('/admin/dashboard')
@page_access_required
def admin_dashboard(role):
    """
    Renders the main admin dashboard page.
    """
    return render_template('dashboard.html', role=role)

@app.route('/reports/weekly')
@page_access_required
def weekly_report(role):
    """
    Renders the Weekly Attendance Report page.
    """
    return render_template('weekly_report.html', role=role)

@app.route('/employees')
@page_access_required
def employees_page(role):
    """
    Renders the Employees Management page.
    """
    return render_template('employees.html', role=role)

@app.route('/accounts')
@page_access_required
def accounts_page(role):
    """
    [إضافة جديدة] - يعرض صفحة إدارة حسابات المستخدمين.
    """
    return render_template('accounts.html', role=role)

@app.route('/employee/<int:employee_id>')
@page_access_required
def employee_detail_page(role, employee_id):
    """
    Renders the Employee Details page.
    """
    return render_template('employee_detail.html', role=role)

@app.route('/attendance/history')
@page_access_required
def attendance_history_page(role):
    """
    Renders the Attendance History page for admin.
    """
    return render_template('attendance_history.html', role=role)

@app.route('/profile')
@page_access_required
def profile_page(role):
    """
    Renders the Profile page for user profile management.
    """
    return render_template('profile.html', role=role)

@app.route('/departments')
@page_access_required
def departments_page(role):
    """
    Renders the Departments management page.
    """
    return render_template('departments.html', role=role)

@app.route('/settings')
@page_access_required
def settings_page(role):
    """
    Renders the System Settings page.
    """
    return render_template('settings.html', role=role)

@app.route('/checkin')
def checkin_page():
    """
    Renders the Face Recognition Check-in page.
    """
    return render_template('checkin.html')

@app.route('/reports/daily')
@page_access_required
def daily_report_page(role):
    """
    Renders the Daily Attendance Report page.
    """
    return render_template('daily_report.html', role=role)

@app.route('/biometric')
def biometric_scan_page():
    """
    Renders the Biometric Scan page.
    """
    return render_template('biometric_scan.html')

@app.route('/notifications')
@page_access_required
def notifications_page(role):
    """
    Renders the Notifications page.
    """
    return render_template('notifications.html', role=role)

@app.route('/code-checkin')
def code_checkin_page():
    """
    Renders the Code Check-In page.
    """
    return render_template('code_checkin.html')

@app.route('/help')
@page_access_required
def help_page(role):
    """
    Renders the Help & About page.
    """
    return render_template('help.html', role=role)

@app.route('/reports/monthly')
@page_access_required
def monthly_report_page(role):
    """
    Renders the Monthly Report page.
    """
    return render_template('monthly_report.html', role=role)

# ==============================================================================
#                           ONBOARDING ROUTES
# ==============================================================================

@app.route('/welcome')
@app.route('/')
def welcome_page():
    """
    Renders the Welcome landing page - first page users see.
    """
    return render_template('welcome.html')

@app.route('/onboarding/face-setup')
def face_setup_page():
    """
    Renders the Face Recognition Setup page.
    """
    return render_template('face_setup.html')

@app.route('/onboarding/code-verification')
def code_verification_page():
    """
    Renders the Code Verification page.
    """
    return render_template('code_verification.html')

@app.route('/onboarding/complete')
def onboarding_complete_page():
    """
    Renders the Onboarding Complete page.
    """
    return render_template('onboarding_complete.html')

@app.route('/employees/<int:employee_id>/face-registration')
def face_registration_page(employee_id):
    """
    Renders the Face Registration page for a specific employee.
    """
    # Verify employee exists
    employee = EmployeeInfo.query.get_or_404(employee_id)
    return render_template('face_registration.html')

@app.route('/debug_babel')
def debug_babel():
    from flask_babel import gettext as _
    current_locale = get_locale()
    babel_info = {
        'current_locale': str(current_locale),  # Convert Locale object to string
        'babel_default_locale': str(babel.default_locale) if hasattr(babel, 'default_locale') else 'N/A',
        'available_locales': list(app.config['LANGUAGES'].keys()),
        'test_translation': _('System Settings'),
        'translation_works': _('System Settings') != 'System Settings'
    }
    return jsonify(babel_info)


if __name__ == '__main__':
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'reset-db':
            reset_database()
            print("✅ Database reset complete. You can now run the application normally.")
            sys.exit(0)
        elif sys.argv[1] == 'init-db':
            initialize_database()
            sys.exit(0)

    # Step 1: Initialize the database and create tables if they don't exist.
    print("\n--- Initializing Database ---")
    initialize_database()

    # Step 2: Check if the database has any employee data.
    # We use an app_context to allow database queries before the server starts.
    with app.app_context():
        try:
            employee_count = EmployeeInfo.query.count()
            if employee_count == 0:
                print("\nDatabase is empty. Starting one-time employee enrollment.")
                # Step 3: If no employees, run the enrollment process.
                enroll_employees_from_folder(EMPLOYEE_FOLDER_TRAINING)

                # Step 4: After enrollment, test recognition from "New employees" folder (like new.ipynb)
                print(f"\n🔍 جاري مقارنة الموظفين من مجلد الاختبار: {NEW_EMPLOYEE_FOLDER_TESTING} ...")
                database_embeddings = load_all_known_embeddings_from_db()

                if not database_embeddings:
                    print("⚠️ لا يمكن المتابعة بدون تضمينات مسجلة.")
                else:
                    test_results = test_recognition_from_folder(NEW_EMPLOYEE_FOLDER_TESTING, database_embeddings)

                    print("\n📊 --- ملخص نتائج اختبار المجلد ---")
                    for true_person_name, (recognized_name, score, code_if_known, status) in test_results.items():
                        print(f"  الموظف (الهوية الحقيقية): {true_person_name}")
                        print(f"    تم التعرف عليه كـ: {recognized_name} (التشابه: {score:.4f}) | الكود (إذا معروف): {code_if_known}")
                        print(f"    الحالة: {status}")
                        if status == "Face Recognized":
                            if recognized_name.lower() == true_person_name.lower():
                                print("    ✅ النتيجة: تعرف صحيح.")
                            else:
                                print(f"    ❌ النتيجة: تعرف خاطئ (كان يجب أن يكون {true_person_name}).")
                        else:
                            print(f"    ❌ النتيجة: {status}")
            else:
                print(f"\n✅ Found {employee_count} employees in the database. Skipping enrollment.")
        except Exception as e:
            print(f"⚠️ Error checking employee count: {e}")
            print("Proceeding without enrollment check...")

    # Step 4: Start the Flask web server.
    # This will make the API endpoints available.
    print("\n--- Starting Flask Web Server ---")

    # [جديد] تهيئة وبدء تشغيل مجدول المهام الخلفي
    # هذا المجدول مسؤول عن المهام التلقائية مثل تسجيل الغياب.
    try:
        from scheduler import initialize_scheduler
        scheduler = initialize_scheduler()
        print("✅ إعداد المجدول اكتمل بنجاح.")
    except Exception as e:
        print(f"❌ فشل في تهيئة المجدول: {e}")

    print("API will be available at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
