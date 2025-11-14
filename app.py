
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import threading
import time
import json
import numpy as np
from datetime import datetime
import os
import csv
from io import StringIO
import requests  
from dotenv import load_dotenv 
load_dotenv()
app = Flask(__name__)
# Configuración desde variables de entorno
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY') 
FLASK_ENV = os.getenv('FLASK_ENV', 'production')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

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
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
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
    def __init__(self):
        self.data_file = 'asistencia_registros.csv'
        self.ensure_csv_headers()
    
    def ensure_csv_headers(self):
        """Asegurar que el CSV tenga los encabezados correctos"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Fecha', 'Hora_Ingreso', 'Nombre', 'Apellido', 'Edad', 
                    'Sexo', 'Correo', 'Celular', 'Timestamp'
                ])
    
    def register_attendance(self, user_data):
        """Registrar asistencia en CSV"""
        timestamp = datetime.now()
        
        record = {
            'Fecha': timestamp.strftime('%Y-%m-%d'),
            'Hora_Ingreso': timestamp.strftime('%H:%M:%S'),
            'Nombre': user_data.get('nombre', ''),
            'Apellido': user_data.get('apellido', ''),
            'Edad': user_data.get('edad', ''),
            'Sexo': user_data.get('sexo', ''),
            'Correo': user_data.get('correo', ''),
            'Celular': user_data.get('celular', ''),
            'Timestamp': timestamp.isoformat()
        }
        
        # Guardar en CSV
        with open(self.data_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)
        
        print(f"✅ Asistencia registrada: {user_data.get('nombre', '')} {user_data.get('apellido', '')}")
        return record
    
    def get_all_records(self):
        """Obtener todos los registros"""
        records = []
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                records = list(reader)
        return records
    
    def has_records(self):
        """Verificar si hay registros"""
        return os.path.exists(self.data_file) and len(self.get_all_records()) > 0

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

# Rutas de la aplicación
@app.route('/')
def index():
    """Vista principal - Menú de inicio"""
    has_records = attendance_manager.has_records()
    return render_template('index.html', has_records=has_records)

@app.route('/registro')
def registro():
    """Vista de registro de asistencia"""
    global system_state
    system_state = SystemState.WAITING
    return render_template('registro.html')

@app.route('/detectar_rostro')
def detectar_rostro():
    """Iniciar detección de rostro"""
    global system_state
    system_state = SystemState.DETECTING_FACE
    
    # Inicializar cámara si no está activa
    if not camera.cap or not camera.cap.isOpened():
        camera.initialize_camera()
    
    return render_template('detectar_rostro.html')

@app.route('/formulario')
def formulario():
    """Formulario de datos personales"""
    global system_state
    if face_detected:
        system_state = SystemState.FACE_DETECTED
        assemblyai_enabled = assemblyai_client.enabled
        return render_template('formulario.html', assemblyai_enabled=assemblyai_enabled)
    else:
        return redirect(url_for('detectar_rostro'))

@app.route('/completado')
def completado():
    """Vista de registro completado"""
    global system_state
    system_state = SystemState.COMPLETED
    
    # Pasar los datos del último usuario registrado
    return render_template('completado.html', user_data=last_registered_user)

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
def api_register():
    """API para registrar asistencia"""
    global system_state, current_user_data, last_registered_user
    
    try:
        user_data = request.json
        current_user_data = user_data
        
        # Validar datos requeridos
        required_fields = ['nombre', 'apellido']
        for field in required_fields:
            if not user_data.get(field):
                return jsonify({'success': False, 'error': f'Falta el campo: {field}'})
        
        # Registrar asistencia
        system_state = SystemState.REGISTERING
        time.sleep(1)  # Pequeña pausa para efecto visual
        
        record = attendance_manager.register_attendance(user_data)
        last_registered_user = record
        system_state = SystemState.COMPLETED
        
        return jsonify({
            'success': True, 
            'message': 'Asistencia registrada correctamente',
            'record': record
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_csv')
def api_export_csv():
    """Exportar registros a CSV"""
    records = attendance_manager.get_all_records()
    
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
def api_records():
    """Obtener todos los registros"""
    records = attendance_manager.get_all_records()
    return jsonify({
        'records': records, 
        'total': len(records), 
        'has_records': len(records) > 0
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

if __name__ == '__main__':
    # Iniciar procesamiento de frames
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    print("🚀 Sistema de Control de Asistencia Mejorado")
    print("🌐 Accede a: http://localhost:5000")
    print("📊 Características:")
    print("   - Detección facial para verificación")
    print("   - Formulario completo de datos personales")
    print("   - Exportación a CSV")
    print("   - Interfaz de múltiples vistas")
    if assemblyai_client.enabled:
        print("   - ✅ AssemblyAI integrado para comandos de voz")
        print("   - 🔑 API Key configurada correctamente")
    else:
        print("   - ❌ AssemblyAI no configurado")
        print("   - 💡 Verifica que la API key sea válida")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Cerrando sistema...")
        camera.release()
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        camera.release()