
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, send_from_directory
import cv2
import threading
import time
import json
import numpy as np
from datetime import datetime, date
import os
import csv
from io import StringIO
import requests
from dotenv import load_dotenv
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

load_dotenv()
app = Flask(__name__)

# Database and authentication setup
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
database_url = os.getenv('SUPABASE_DATABASE_URL', os.getenv('DATABASE_URL', 'sqlite:///attendance.db'))
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# PostgreSQL specific configuration for Supabase
if database_url.startswith('postgresql'):
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

from models import db, User, Department, Position, Attendance, Evidence, SystemLog, migrate_csv_data

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
# Configuración desde variables de entorno
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
FLASK_ENV = os.getenv('FLASK_ENV', 'production')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def create_tables():
    """Create database tables and migrate data if needed"""
    with app.app_context():
        try:
            db.create_all()
            print("✅ Conexión a base de datos exitosa")
        except Exception as e:
            print(f"❌ Error conectando a la base de datos: {e}")
            print("💡 Verifica tu configuración de SUPABASE_DATABASE_URL en .env")
            return False
        return True

        # Create default admin user if none exists
        if not User.query.filter_by(role='admin').first():
            admin = User(
                username='admin',
                email='admin@company.com',
                first_name='Admin',
                last_name='User',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)

        # Create default departments and positions
        if not Department.query.first():
            departments = [
                Department(name='IT', description='Information Technology'),
                Department(name='HR', description='Human Resources'),
                Department(name='Sales', description='Sales and Marketing'),
                Department(name='Operations', description='Operations'),
            ]
            for dept in departments:
                db.session.add(dept)

        if not Position.query.first():
            positions = [
                Position(name='Manager', description='Department Manager'),
                Position(name='Supervisor', description='Team Supervisor'),
                Position(name='Employee', description='Regular Employee'),
                Position(name='Intern', description='Internship Position'),
            ]
            for pos in positions:
                db.session.add(pos)

        try:
            db.session.commit()
            # Migrate CSV data if database is empty
            if not Attendance.query.first():
                migrate_csv_data()
        except Exception as e:
            db.session.rollback()
            print(f"Error initializing database: {e}")

# Verificar que la API key esté configurada
if not ASSEMBLYAI_API_KEY:
    print("❌ ERROR: ASSEMBLYAI_API_KEY no encontrada en variables de entorno")
    print("💡 Crea un archivo .env con tu API key")
    exit(1)

# Estado del sistema
class SystemState:
    WAITING = "waiting"
    DETECTING_FACE = "detecting_face"
    FACE_DETECTED = "face_detected"
    REGISTERING = "registering"
    COMPLETED = "completed"

system_state = SystemState.WAITING
current_user_data = {}
last_registered_user = {}

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.last_frame = None
    
    def initialize_camera(self):
        """Inicializar cámara"""
        try:
            # cv2.CAP_DSHOW is Windows only, removing it for Linux compatibility
            if os.name == 'nt':
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return True
        except Exception as e:
            print(f"Error cámara: {e}")
        return False
    
    def get_frame(self):
        """Obtener frame de la cámara"""
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    return frame
            except Exception as e:
                print(f"Error capturando frame: {e}")
        
        # Frame de simulación
        return self.create_test_frame()
    
    def create_test_frame(self):
        """Crear frame de prueba"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 60
        cv2.putText(frame, "CAMARA NO DISPONIBLE", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Usando modo simulacion", (50, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    
    def release(self):
        if self.cap:
            self.cap.release()

class FaceDetector:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            self.face_cascade = None
    
    def detect_face(self, frame):
        """Detectar si hay un rostro en el frame"""
        if self.face_cascade is None:
            return frame, True  # En simulación, siempre detecta rostro
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        # Dibujar rectángulo alrededor del rostro
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, "ROSTRO DETECTADO", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar instrucciones
        if len(faces) > 0:
            cv2.putText(frame, "✓ Rostro detectado - Puede continuar", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Acercarse a la camara para deteccion", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, len(faces) > 0

class AttendanceManager:
    @staticmethod
    def register_attendance(user_id, user_data=None):
        """Registrar asistencia en base de datos"""
        today = date.today()
        now = datetime.now()

        # Check if user already checked in today
        existing = Attendance.query.filter_by(user_id=user_id, date=today).first()

        if existing and not existing.check_out_time:
            # Prevent check-out immediately after check-in (minimum 1 minute)
            time_since_checkin = now - existing.check_in_time
            if time_since_checkin.total_seconds() < 60:  # 60 seconds = 1 minute
                print(f"⚠️ Check-out rechazado: muy pronto después del check-in ({time_since_checkin.total_seconds():.1f}s)")
                return existing  # Return existing record without check-out
    
            # Check out
            existing.check_out_time = now
            existing.updated_at = now
            db.session.commit()
    
            # Log action
            SystemLog.log_action(user_id, 'check_out', f'Checked out at {now.strftime("%H:%M:%S")}', request.remote_addr if request else None)
    
            print(f"✅ Check-out registrado para usuario {user_id}")
            return existing
        elif not existing:
            # Check in
            attendance = Attendance(
                user_id=user_id,
                date=today,
                check_in_time=now,
                status='present'
            )

            # Add legacy data if provided (for migration)
            if user_data:
                notes = []
                if user_data.get('edad'): notes.append(f"Edad: {user_data['edad']}")
                if user_data.get('sexo'): notes.append(f"Sexo: {user_data['sexo']}")
                if user_data.get('celular'): notes.append(f"Celular: {user_data['celular']}")
                attendance.notes = ', '.join(notes)

            db.session.add(attendance)
            db.session.commit()

            # Log action
            SystemLog.log_action(user_id, 'check_in', f'Checked in at {now.strftime("%H:%M:%S")}', request.remote_addr if request else None)

            print(f"✅ Check-in registrado para usuario {user_id}")
            return attendance

        return existing

    @staticmethod
    def get_user_attendance(user_id, date_from=None, date_to=None):
        """Obtener registros de asistencia de un usuario"""
        query = Attendance.query.filter_by(user_id=user_id)
        if date_from:
            query = query.filter(Attendance.date >= date_from)
        if date_to:
            query = query.filter(Attendance.date <= date_to)
        return query.order_by(Attendance.date.desc()).all()

    @staticmethod
    def get_all_records():
        """Obtener todos los registros para exportación"""
        attendances = Attendance.query.join(User).order_by(Attendance.date.desc(), Attendance.check_in_time.desc()).all()

        records = []
        for att in attendances:
            record = {
                'Fecha': att.date.strftime('%Y-%m-%d'),
                'Hora_Ingreso': att.check_in_time.strftime('%H:%M:%S') if att.check_in_time else '',
                'Hora_Salida': att.check_out_time.strftime('%H:%M:%S') if att.check_out_time else '',
                'Nombre': att.user.first_name,
                'Apellido': att.user.last_name,
                'Correo': att.user.email,
                'Departamento': att.user.department.name if att.user.department else '',
                'Posicion': att.user.position.name if att.user.position else '',
                'Estado': att.status,
                'Horas_Trabajadas': f"{att.total_hours:.2f}" if att.total_hours > 0 else '',
                'Timestamp': att.created_at.isoformat()
            }
            records.append(record)

        return records

    @staticmethod
    def has_records():
        """Verificar si hay registros"""
        return Attendance.query.count() > 0

    @staticmethod
    def get_today_stats():
        """Obtener estadísticas del día"""
        today = date.today()
        attendances = Attendance.query.filter_by(date=today).all()

        stats = {
            'total_checkins': len([a for a in attendances if a.check_in_time]),
            'total_checkouts': len([a for a in attendances if a.check_out_time]),
            'present_today': len(attendances),
            'late_arrivals': len([a for a in attendances if a.status == 'late'])
        }
        return stats

# Cliente AssemblyAI personalizado
class SimpleTranscriber:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.enabled = bool(api_key and api_key != "tu_api_key_aqui")
        print(f"🔧 AssemblyAI configurado: {self.enabled}")
    
    def transcribe_audio(self, audio_file_path):
        """Transcripción simple usando la API REST de AssemblyAI"""
        if not self.enabled:
            return {"error": "AssemblyAI no configurado"}
        
        try:
            print("📤 Subiendo audio a AssemblyAI...")
            
            # Leer archivo de audio
            with open(audio_file_path, 'rb') as audio_file:
                # Subir archivo
                upload_response = requests.post(
                    'https://api.assemblyai.com/v2/upload',
                    headers={'authorization': self.api_key},
                    data=audio_file.read(),
                    timeout=30
                )
                
                if upload_response.status_code != 200:
                    error_msg = f"Error en upload ({upload_response.status_code}): {upload_response.text}"
                    print(f"❌ {error_msg}")
                    return {"error": error_msg}
                
                upload_url = upload_response.json()['upload_url']
                print("✅ Audio subido correctamente")
                
                # Solicitar transcripción
                transcript_response = requests.post(
                    'https://api.assemblyai.com/v2/transcript',
                    headers={
                        'authorization': self.api_key,
                        'content-type': 'application/json'
                    },
                    json={
                        'audio_url': upload_url,
                        'language_code': 'es'
                    },
                    timeout=30
                )
                
                if transcript_response.status_code != 200:
                    error_msg = f"Error en transcripción ({transcript_response.status_code}): {transcript_response.text}"
                    print(f"❌ {error_msg}")
                    return {"error": error_msg}
                
                transcript_id = transcript_response.json()['id']
                print(f"🆔 ID de transcripción: {transcript_id}")
                
                # Polling para obtener resultado
                return self.wait_for_transcription(transcript_id)
                
        except requests.exceptions.Timeout:
            error_msg = "Timeout en la conexión con AssemblyAI"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error en transcripción: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def wait_for_transcription(self, transcript_id, timeout=60):
        """Esperar a que la transcripción esté lista"""
        start_time = time.time()
        polling_interval = 2
        
        while time.time() - start_time < timeout:
            try:
                # Verificar estado
                polling_response = requests.get(
                    f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                    headers={'authorization': self.api_key},
                    timeout=30
                )
                
                if polling_response.status_code != 200:
                    return {"error": f"Error en polling: {polling_response.text}"}
                
                polling_result = polling_response.json()
                status = polling_result['status']
                
                if status == 'completed':
                    print("✅ Transcripción completada")
                    return {
                        'success': True,
                        'text': polling_result['text'],
                        'confidence': polling_result.get('confidence', 1.0),
                        'duration': polling_result.get('audio_duration', 0)
                    }
                elif status == 'error':
                    error_msg = polling_result.get('error', 'Error desconocido en transcripción')
                    print(f"❌ Error en transcripción: {error_msg}")
                    return {"error": error_msg}
                elif status == 'processing':
                    print("⏳ Procesando audio...")
                elif status == 'queued':
                    print("📋 Audio en cola...")
                
                time.sleep(polling_interval)
                polling_interval = min(polling_interval * 1.5, 5)  # Backoff exponencial
                
            except requests.exceptions.Timeout:
                print("⏰ Timeout en polling, reintentando...")
                time.sleep(polling_interval)
            except Exception as e:
                return {"error": f"Error en polling: {str(e)}"}
        
        return {"error": f"Timeout después de {timeout} segundos"}

class VoiceFormManager:
    def __init__(self, assemblyai_client):
        self.client = assemblyai_client
        self.field_keywords = {
            'nombre': ['nombre', 'llámame', 'me llamo', 'mi nombre es', 'soy'],
            'apellido': ['apellido', 'apellidos', 'mis apellidos'],
            'edad': ['edad', 'años', 'tengo', 'mi edad es'],
            'correo': ['correo', 'email', 'mail', 'correo electrónico'],
            'celular': ['celular', 'teléfono', 'número', 'contacto', 'móvil']
        }
    
    def process_voice_input(self, audio_file_path, current_field=None):
        """Procesar entrada de voz y extraer información relevante"""
        if not self.client.enabled:
            return {"error": "AssemblyAI no disponible"}
        
        # Transcribir audio
        result = self.client.transcribe_audio(audio_file_path)
        if "error" in result:
            return result
        
        text = result["text"].lower().strip()
        print(f"📝 Texto transcrito: {text}")
        
        # Si se especifica un campo, usar directamente
        if current_field:
            return {
                'success': True,
                'field': current_field,
                'value': self.clean_text(text),
                'confidence': result.get('confidence', 1.0),
                'original_text': text
            }
        
        # Detectar comandos especiales
        command_result = self.detect_commands(text)
        if command_result:
            return command_result
        
        # Intentar detectar campo automáticamente
        return self.auto_detect_field(text, result.get('confidence', 1.0))
    
    def detect_commands(self, text):
        """Detectar comandos de voz especiales"""
        if any(cmd in text for cmd in ['siguiente', 'continuar', 'next', 'adelante']):
            return {'success': True, 'action': 'next_field'}
        elif any(cmd in text for cmd in ['anterior', 'atrás', 'back', 'regresar']):
            return {'success': True, 'action': 'previous_field'}
        elif any(cmd in text for cmd in ['enviar', 'registrar', 'finalizar', 'terminar']):
            return {'success': True, 'action': 'submit_form'}
        elif any(cmd in text for cmd in ['limpiar', 'borrar', 'reset', 'empezar de nuevo']):
            return {'success': True, 'action': 'clear_form'}
        elif any(cmd in text for cmd in ['ayuda', 'help', 'asistencia']):
            return {'success': True, 'action': 'show_help'}
        
        return None
    
    def auto_detect_field(self, text, confidence):
        """Detectar automáticamente el campo basado en palabras clave"""
        for field, keywords in self.field_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    value = self.extract_value_after_keyword(text, keyword)
                    return {
                        'success': True,
                        'field': field,
                        'value': value,
                        'confidence': confidence,
                        'original_text': text,
                        'action': 'fill_field'
                    }
        
        # Si no se detecta campo específico, devolver texto completo
        return {
            'success': True,
            'field': 'unknown',
            'value': self.clean_text(text),
            'confidence': confidence,
            'original_text': text,
            'action': 'unknown'
        }
    
    def extract_value_after_keyword(self, text, keyword):
        """Extraer valor después de una palabra clave"""
        if keyword in text:
            start_pos = text.find(keyword) + len(keyword)
            value = text[start_pos:].strip()
            # Limpiar valor
            value = self.clean_text(value)
            return value
        return text
    
    def clean_text(self, text):
        """Limpiar y formatear texto"""
        # Remover palabras comunes no necesarias
        common_words = ['es', 'de', 'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'que', 'en']
        words = text.split()
        filtered_words = [word for word in words if word not in common_words]
        
        # Unir y capitalizar
        cleaned = ' '.join(filtered_words).title()
        
        # Limpiar puntuación extra
        cleaned = cleaned.replace('.', '').replace(',', '').strip()
        
        return cleaned

# Inicializar componentes
camera = Camera()
face_detector = FaceDetector()
attendance_manager = AttendanceManager()

# Inicializar AssemblyAI
assemblyai_client = SimpleTranscriber(ASSEMBLYAI_API_KEY)
voice_manager = VoiceFormManager(assemblyai_client)

# Variables globales
current_frame = None
face_detected = False
frame_lock = threading.Lock()

def process_frames():
    """Procesamiento de frames en segundo plano"""
    global current_frame, face_detected
    
    while True:
        try:
            frame = camera.get_frame()
            
            if system_state == SystemState.DETECTING_FACE:
                # Solo detectar rostros cuando estamos en modo detección
                processed_frame, face_found = face_detector.detect_face(frame)
                face_detected = face_found
            else:
                processed_frame = frame
                # Mostrar información del estado
                if system_state == SystemState.WAITING:
                    cv2.putText(processed_frame, "SISTEMA DE ASISTENCIA", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(processed_frame, "Esperando registro...", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                elif system_state == SystemState.FACE_DETECTED:
                    cv2.putText(processed_frame, "✓ ROSTRO VERIFICADO", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(processed_frame, "Complete el formulario", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                elif system_state == SystemState.REGISTERING:
                    cv2.putText(processed_frame, "REGISTRANDO DATOS...", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                elif system_state == SystemState.COMPLETED:
                    cv2.putText(processed_frame, "✓ REGISTRO COMPLETADO", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(processed_frame, "Puede continuar", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            with frame_lock:
                current_frame = processed_frame
            
            time.sleep(0.033)
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            time.sleep(1)

def generate_frames():
    """Generar stream de video"""
    while True:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

# Rutas de la aplicación (legacy - now handled by protected routes below)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Old registro route removed - now handled by protected route below

@app.route('/detectar_rostro')
@login_required
def detectar_rostro():
    """Iniciar detección de rostro"""
    global system_state
    system_state = SystemState.DETECTING_FACE

    # Inicializar cámara si no está activa
    if not camera.cap or not camera.cap.isOpened():
        camera.initialize_camera()

    # Check if user needs to complete profile
    needs_profile_completion = not (
        current_user.first_name and
        current_user.last_name and
        current_user.email
    )

    return render_template('detectar_rostro.html', needs_profile_completion=needs_profile_completion)

@app.route('/formulario')
@login_required
def formulario():
    """Formulario de datos personales - Solo para completar perfil inicial"""
    global system_state
    if face_detected:
        system_state = SystemState.FACE_DETECTED
        assemblyai_enabled = assemblyai_client.enabled

        # Check if user has complete profile
        has_complete_profile = (
            current_user.first_name and
            current_user.last_name and
            current_user.email and
            current_user.department and
            current_user.position
        )

        return render_template('formulario.html',
                             assemblyai_enabled=assemblyai_enabled,
                             has_complete_profile=has_complete_profile)
    else:
        return redirect(url_for('detectar_rostro'))

@app.route('/completado')
@login_required
def completado():
    """Vista de registro completado"""
    global system_state
    system_state = SystemState.COMPLETED

    # Get the latest attendance for current user
    today = date.today()
    latest_attendance = Attendance.query.filter_by(user_id=current_user.id, date=today).order_by(Attendance.created_at.desc()).first()

    return render_template('completado.html', user_data=last_registered_user, attendance=latest_attendance)

# API endpoints
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/face_status')
def api_face_status():
    """API para verificar estado de detección facial"""
    return jsonify({
        'face_detected': face_detected,
        'system_state': system_state
    })

@app.route('/api/register', methods=['POST'])
@login_required
def api_register():
    """API para registrar asistencia"""
    global system_state, last_registered_user

    try:
        # Check if this is a profile update or attendance registration
        user_data = request.json or {}

        # If user data is provided, update profile first
        if user_data and any(key in user_data for key in ['nombre', 'apellido', 'email', 'departamento', 'posicion']):
            # Update user profile
            if 'nombre' in user_data:
                current_user.first_name = user_data['nombre']
            if 'apellido' in user_data:
                current_user.last_name = user_data['apellido']
            if 'email' in user_data:
                current_user.email = user_data['email']
            if 'edad' in user_data:
                current_user.phone = user_data.get('edad', '')  # Using phone field for age for now
            if 'departamento' in user_data:
                dept = Department.query.filter_by(name=user_data['departamento']).first()
                if not dept:
                    dept = Department(name=user_data['departamento'])
                    db.session.add(dept)
                    db.session.flush()
                current_user.department_id = dept.id
            if 'posicion' in user_data:
                pos = Position.query.filter_by(name=user_data['posicion']).first()
                if not pos:
                    pos = Position(name=user_data['posicion'])
                    db.session.add(pos)
                    db.session.flush()
                current_user.position_id = pos.id

            db.session.commit()
            SystemLog.log_action(current_user.id, 'profile_update', 'Updated profile information', request.remote_addr)

        # Registrar asistencia usando el usuario actual
        system_state = SystemState.REGISTERING
        time.sleep(1)  # Pequeña pausa para efecto visual

        record = AttendanceManager.register_attendance(current_user.id)
        last_registered_user = {
            'id': record.id,
            'date': record.date.strftime('%Y-%m-%d'),
            'check_in_time': record.check_in_time.strftime('%H:%M:%S') if record.check_in_time else None,
            'check_out_time': record.check_out_time.strftime('%H:%M:%S') if record.check_out_time else None,
            'status': record.status,
            'user': current_user.full_name,
            'total_hours': record.total_hours
        }
        system_state = SystemState.COMPLETED

        action = 'check_out' if record.check_out_time else 'check_in'
        return jsonify({
            'success': True,
            'message': f'{"Check-out" if record.check_out_time else "Check-in"} registrado correctamente',
            'record': last_registered_user,
            'action': action,
            'redirect_url': url_for('evidencia', attendance_id=record.id) if not record.check_out_time else None
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_csv')
@login_required
def api_export_csv():
    """Exportar registros a CSV según permisos"""
    if current_user.is_admin():
        # Admin exporta todos los registros
        records = AttendanceManager.get_all_records()
    elif current_user.is_manager():
        # Manager exporta registros de su departamento
        department_users = User.query.filter_by(department_id=current_user.department_id).all()
        user_ids = [u.id for u in department_users]
        attendances = Attendance.query.filter(Attendance.user_id.in_(user_ids)).all()

        records = []
        for att in attendances:
            record = {
                'Fecha': att.date.strftime('%Y-%m-%d'),
                'Hora_Ingreso': att.check_in_time.strftime('%H:%M:%S') if att.check_in_time else '',
                'Hora_Salida': att.check_out_time.strftime('%H:%M:%S') if att.check_out_time else '',
                'Nombre': att.user.first_name,
                'Apellido': att.user.last_name,
                'Correo': att.user.email,
                'Departamento': att.user.department.name if att.user.department else '',
                'Posicion': att.user.position.name if att.user.position else '',
                'Estado': att.status,
                'Horas_Trabajadas': f"{att.total_hours:.2f}" if att.total_hours > 0 else '',
                'Timestamp': att.created_at.isoformat()
            }
            records.append(record)
    else:
        # Employee exporta solo sus registros
        attendances = AttendanceManager.get_user_attendance(current_user.id)
        records = []
        for att in attendances:
            record = {
                'Fecha': att.date.strftime('%Y-%m-%d'),
                'Hora_Ingreso': att.check_in_time.strftime('%H:%M:%S') if att.check_in_time else '',
                'Hora_Salida': att.check_out_time.strftime('%H:%M:%S') if att.check_out_time else '',
                'Nombre': att.user.first_name,
                'Apellido': att.user.last_name,
                'Correo': att.user.email,
                'Departamento': att.user.department.name if att.user.department else '',
                'Posicion': att.user.position.name if att.user.position else '',
                'Estado': att.status,
                'Horas_Trabajadas': f"{att.total_hours:.2f}" if att.total_hours > 0 else '',
                'Timestamp': att.created_at.isoformat()
            }
            records.append(record)
    
    if not records:
        # Devolver mensaje amigable en lugar de error
        return """
        <html>
            <head>
                <title>Sin Registros</title>
                <style>
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        text-align: center;
                    }
                    .container {
                        background: rgba(255,255,255,0.1);
                        padding: 40px;
                        border-radius: 20px;
                        backdrop-filter: blur(10px);
                        max-width: 500px;
                    }
                    h2 { margin-bottom: 20px; font-size: 2em; }
                    p { margin-bottom: 25px; line-height: 1.6; }
                    button {
                        padding: 12px 25px;
                        border: none;
                        border-radius: 25px;
                        font-size: 1em;
                        font-weight: 600;
                        cursor: pointer;
                        margin: 0 10px;
                        background: #667eea;
                        color: white;
                        transition: all 0.3s ease;
                    }
                    button:hover {
                        background: #5a6fd8;
                        transform: scale(1.05);
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>📭 No hay registros para exportar</h2>
                    <p>No se encontraron registros de asistencia en el sistema.</p>
                    <p>Realice algunos registros primero y luego intente exportar.</p>
                    <button onclick="window.history.back()">Volver</button>
                    <button onclick="window.location.href='/'">Ir al Inicio</button>
                </div>
            </body>
        </html>
        """
    
    # Crear CSV en memoria
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    
    response = Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=asistencia_completa.csv"}
    )
    
    return response

@app.route('/api/records')
@login_required
def api_records():
    """Obtener registros según permisos del usuario"""
    if current_user.is_admin():
        # Admin ve todos los registros
        records = AttendanceManager.get_all_records()
    elif current_user.is_manager():
        # Manager ve registros de su departamento
        department_users = User.query.filter_by(department_id=current_user.department_id).all()
        user_ids = [u.id for u in department_users]
        attendances = Attendance.query.filter(Attendance.user_id.in_(user_ids)).all()

        records = []
        for att in attendances:
            record = {
                'Fecha': att.date.strftime('%Y-%m-%d'),
                'Hora_Ingreso': att.check_in_time.strftime('%H:%M:%S') if att.check_in_time else '',
                'Hora_Salida': att.check_out_time.strftime('%H:%M:%S') if att.check_out_time else '',
                'Nombre': att.user.first_name,
                'Apellido': att.user.last_name,
                'Correo': att.user.email,
                'Departamento': att.user.department.name if att.user.department else '',
                'Posicion': att.user.position.name if att.user.position else '',
                'Estado': att.status,
                'Horas_Trabajadas': f"{att.total_hours:.2f}" if att.total_hours > 0 else '',
                'Timestamp': att.created_at.isoformat()
            }
            records.append(record)
    else:
        # Employee ve solo sus propios registros
        records = AttendanceManager.get_user_attendance(current_user.id)
        # Convertir a formato de lista como los otros
        formatted_records = []
        for att in records:
            record = {
                'Fecha': att.date.strftime('%Y-%m-%d'),
                'Hora_Ingreso': att.check_in_time.strftime('%H:%M:%S') if att.check_in_time else '',
                'Hora_Salida': att.check_out_time.strftime('%H:%M:%S') if att.check_out_time else '',
                'Nombre': att.user.first_name,
                'Apellido': att.user.last_name,
                'Correo': att.user.email,
                'Departamento': att.user.department.name if att.user.department else '',
                'Posicion': att.user.position.name if att.user.position else '',
                'Estado': att.status,
                'Horas_Trabajadas': f"{att.total_hours:.2f}" if att.total_hours > 0 else '',
                'Timestamp': att.created_at.isoformat()
            }
            formatted_records.append(record)
        records = formatted_records

    return jsonify({
        'records': records,
        'total': len(records),
        'has_records': len(records) > 0,
        'user_role': current_user.role
    })

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """Transcribir audio usando AssemblyAI"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo de audio'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'})
    
    # Guardar archivo temporal
    filename = f"temp_audio_{int(time.time())}.wav"
    try:
        audio_file.save(filename)
        
        # Obtener campo específico si se proporciona
        current_field = request.form.get('field', None)
        
        # Transcribir con AssemblyAI
        result = voice_manager.process_voice_input(filename, current_field)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error procesando audio: {str(e)}'})
    
    finally:
        # Limpiar archivo temporal
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass

@app.route('/reset_system')
def reset_system():
    """Resetear sistema para nuevo registro"""
    global system_state, face_detected, current_user_data
    system_state = SystemState.WAITING
    face_detected = False
    current_user_data = {}
    return jsonify({'success': True, 'message': 'Sistema reiniciado'})

@app.route('/api/system_status')
def api_system_status():
    """Estado del sistema"""
    return jsonify({
        'assemblyai_enabled': assemblyai_client.enabled,
        'camera_available': camera.cap is not None and camera.cap.isOpened(),
        'face_detected': face_detected,
        'system_state': system_state,
        'total_records': len(attendance_manager.get_all_records())
    })

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password) and user.is_active:
            login_user(user)
            SystemLog.log_action(user.id, 'login', f'Login from {request.remote_addr}', request.remote_addr)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))

        flash('Usuario o contraseña incorrectos', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    SystemLog.log_action(current_user.id, 'logout', f'Logout from {request.remote_addr}', request.remote_addr)
    logout_user()
    return redirect(url_for('login'))

# Protected routes
@app.route('/')
@login_required
def index():
    """Vista principal - Menú de inicio"""
    has_records = AttendanceManager.has_records()
    stats = AttendanceManager.get_today_stats()

    # Check if user has checked in today but not checked out
    today = date.today()
    current_attendance = Attendance.query.filter_by(
        user_id=current_user.id,
        date=today
    ).first()

    show_checkout = current_attendance and current_attendance.check_in_time and not current_attendance.check_out_time

    return render_template('index.html',
                         has_records=has_records,
                         stats=stats,
                         show_checkout=show_checkout,
                         current_attendance=current_attendance)

@app.route('/admin')
@login_required
def admin():
    """Panel de administración"""
    if not current_user.is_admin():
        flash('Acceso denegado. Se requieren permisos de administrador.', 'error')
        return redirect(url_for('index'))

    users = User.query.all()
    departments = Department.query.all()
    positions = Position.query.all()
    return render_template('admin.html', users=users, departments=departments, positions=positions)

@app.route('/admin/user/create', methods=['POST'])
@login_required
def admin_create_user():
    """Crear nuevo usuario (solo admin)"""
    if not current_user.is_admin():
        return jsonify({'success': False, 'error': 'Acceso denegado'})

    try:
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        role = request.form.get('role', 'employee')
        department_id = request.form.get('department_id')
        position_id = request.form.get('position_id')

        # Validar datos
        if not all([username, email, password, first_name, last_name]):
            flash('Todos los campos son obligatorios', 'error')
            return redirect(url_for('admin'))

        # Verificar si usuario ya existe
        if User.query.filter_by(username=username).first():
            flash('El nombre de usuario ya existe', 'error')
            return redirect(url_for('admin'))

        if User.query.filter_by(email=email).first():
            flash('El correo electrónico ya está registrado', 'error')
            return redirect(url_for('admin'))

        # Crear usuario
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            role=role,
            department_id=int(department_id) if department_id else None,
            position_id=int(position_id) if position_id else None
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        # Log action
        SystemLog.log_action(current_user.id, 'create_user', f'Created user {username}', request.remote_addr)

        flash(f'Usuario {username} creado exitosamente', 'success')
        return redirect(url_for('admin'))

    except Exception as e:
        db.session.rollback()
        flash(f'Error al crear usuario: {str(e)}', 'error')
        return redirect(url_for('admin'))

@app.route('/registro')
@login_required
def registro():
    """Vista de registro de asistencia"""
    global system_state
    system_state = SystemState.WAITING

    # Check if user already checked in today
    today = date.today()
    existing_attendance = Attendance.query.filter_by(user_id=current_user.id, date=today).first()
    already_checked_in = existing_attendance and existing_attendance.check_in_time and not existing_attendance.check_out_time

    return render_template('registro.html', already_checked_in=already_checked_in, attendance=existing_attendance)

@app.route('/evidencia/<int:attendance_id>')
@login_required
def evidencia(attendance_id):
    """Vista para subir evidencia de actividades"""
    attendance = Attendance.query.filter_by(id=attendance_id, user_id=current_user.id).first()
    if not attendance:
        flash('Registro de asistencia no encontrado', 'error')
        return redirect(url_for('index'))

    # Get existing evidence
    existing_evidence = Evidence.query.filter_by(attendance_id=attendance_id).all()

    return render_template('evidencia.html', attendance=attendance, existing_evidence=existing_evidence)

@app.route('/api/evidence/upload', methods=['POST'])
@login_required
def upload_evidence():
    """API para subir evidencia"""
    try:
        attendance_id = request.form.get('attendance_id')
        evidence_type = request.form.get('type')
        title = request.form.get('title', '')
        content = request.form.get('content', '')

        # Verify attendance belongs to user
        attendance = Attendance.query.filter_by(id=attendance_id, user_id=current_user.id).first()
        if not attendance:
            return jsonify({'success': False, 'error': 'Registro no encontrado'})

        # Handle file upload for photos
        filename = None
        if evidence_type == 'photo' and 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(f"{current_user.id}_{attendance_id}_{int(time.time())}_{file.filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

        # Create evidence record
        evidence = Evidence(
            attendance_id=attendance_id,
            type=evidence_type,
            title=title,
            content=content,
            file_path=filename  # Store only filename, not full path
        )

        db.session.add(evidence)
        db.session.commit()

        # Log action
        SystemLog.log_action(current_user.id, 'upload_evidence', f'Uploaded {evidence_type} evidence for attendance {attendance_id}', request.remote_addr)

        return jsonify({
            'success': True,
            'message': 'Evidencia subida correctamente',
            'evidence': {
                'id': evidence.id,
                'type': evidence.type,
                'title': evidence.title,
                'content': evidence.content,
                'file_path': evidence.file_path,
                'created_at': evidence.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/evidence/<int:evidence_id>', methods=['DELETE'])
@login_required
def delete_evidence(evidence_id):
    """Eliminar evidencia"""
    try:
        evidence = Evidence.query.filter_by(id=evidence_id).first()
        if not evidence:
            return jsonify({'success': False, 'error': 'Evidencia no encontrada'})

        # Check if evidence belongs to user's attendance
        attendance = Attendance.query.filter_by(id=evidence.attendance_id, user_id=current_user.id).first()
        if not attendance:
            return jsonify({'success': False, 'error': 'No autorizado'})

        # Delete file if exists
        if evidence.file_path:
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], evidence.file_path)
            if os.path.exists(full_path):
                os.remove(full_path)

        db.session.delete(evidence)
        db.session.commit()

        # Log action
        SystemLog.log_action(current_user.id, 'delete_evidence', f'Deleted evidence {evidence_id}', request.remote_addr)

        return jsonify({'success': True, 'message': 'Evidencia eliminada'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create database tables and migrate data
    if not create_tables():
        print("❌ No se pudo conectar a la base de datos. Revisa la configuración.")
        exit(1)

    # Iniciar procesamiento de frames
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()

    db_type = "Supabase PostgreSQL" if os.getenv('SUPABASE_DATABASE_URL') else "SQLite"
    print(f"🚀 Sistema de Control de Asistencia Mejorado con {db_type}")
    print("🌐 Accede a: http://localhost:5000")
    print("📊 Características:")
    print("   - Autenticación de usuarios con roles")
    print(f"   - Base de datos {db_type} para almacenamiento")
    print("   - Detección facial para verificación")
    print("   - Check-in/Check-out con seguimiento de horas")
    print("   - Dashboard administrativo")
    print("   - Sistema de logs y auditoría")
    if assemblyai_client.enabled:
        print("   - ✅ AssemblyAI integrado para comandos de voz")
        print("   - 🔑 API Key configurada correctamente")
    else:
        print("   - ❌ AssemblyAI no configurado")
        print("   - 💡 Verifica que la API key sea válida")

    try:
        app.run(host='0.0.0.0', port=5000, debug=DEBUG, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Cerrando sistema...")
        camera.release()
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        camera.release()