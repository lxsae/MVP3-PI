import requests
import time
import os
from datetime import datetime

class AssemblyAIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ASSEMBLYAI_API_KEY')
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {
            "authorization": self.api_key,
            "content-type": "application/json"
        }
        self.enabled = bool(self.api_key and self.api_key != "effa893c8d2342b4a09333f6dfbce42e")
    
    def upload_audio(self, audio_file_path):
        """Subir archivo de audio a AssemblyAI"""
        if not self.enabled:
            return {"error": "AssemblyAI no configurado"}
        
        try:
            # Subir archivo
            with open(audio_file_path, "rb") as f:
                upload_response = requests.post(
                    f"{self.base_url}/upload",
                    headers=self.headers,
                    data=f.read()
                )
            
            if upload_response.status_code != 200:
                return {"error": f"Error en upload: {upload_response.text}"}
            
            upload_url = upload_response.json()["upload_url"]
            return {"success": True, "upload_url": upload_url}
            
        except Exception as e:
            return {"error": f"Error subiendo audio: {str(e)}"}
    
    def transcribe_audio(self, audio_file_path):
        """Transcribir archivo de audio completo"""
        if not self.enabled:
            return {"error": "AssemblyAI no configurado"}
        
        try:
            # Primero subir el archivo
            upload_result = self.upload_audio(audio_file_path)
            if "error" in upload_result:
                return upload_result
            
            # Solicitar transcripción
            transcript_response = requests.post(
                f"{self.base_url}/transcript",
                headers=self.headers,
                json={
                    "audio_url": upload_result["upload_url"],
                    "language_code": "es"
                }
            )
            
            if transcript_response.status_code != 200:
                return {"error": f"Error en transcripción: {transcript_response.text}"}
            
            transcript_id = transcript_response.json()["id"]
            
            # Esperar por el resultado
            return self.wait_for_transcription(transcript_id)
            
        except Exception as e:
            return {"error": f"Error en transcripción: {str(e)}"}
    
    def wait_for_transcription(self, transcript_id, timeout=300):
        """Esperar a que la transcripción esté lista"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Verificar estado
            status_response = requests.get(
                f"{self.base_url}/transcript/{transcript_id}",
                headers=self.headers
            )
            
            if status_response.status_code != 200:
                return {"error": f"Error verificando estado: {status_response.text}"}
            
            status_data = status_response.json()
            status = status_data["status"]
            
            if status == "completed":
                return {
                    "success": True,
                    "text": status_data["text"],
                    "words": status_data.get("words", []),
                    "confidence": status_data.get("confidence", 1.0),
                    "duration": status_data.get("audio_duration", 0)
                }
            elif status == "error":
                return {"error": status_data.get("error", "Error desconocido en transcripción")}
            
            # Esperar antes de verificar de nuevo
            time.sleep(2)
        
        return {"error": "Timeout esperando por transcripción"}
    
    def transcribe_realtime(self, audio_chunk):
        """Transcripción en tiempo real (para futura implementación)"""
        # Esta función sería para streaming, pero requiere WebSocket
        # Por ahora usamos transcripción por archivo
        return {"error": "Transcripción en tiempo real no implementada"}

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
        common_words = ['es', 'de', 'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o']
        words = text.split()
        filtered_words = [word for word in words if word not in common_words]
        
        # Unir y capitalizar
        cleaned = ' '.join(filtered_words).title()
        
        # Limpiar puntuación extra
        cleaned = cleaned.replace('.', '').replace(',', '').strip()
        
        return cleaned

# Cliente simplificado para transcripción básica
class SimpleTranscriber:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.enabled = bool(api_key and api_key != "tu_api_key_aqui")
    
    def transcribe_audio(self, audio_file_path):
        """Transcripción simple usando la API REST"""
        if not self.enabled:
            return {"error": "AssemblyAI no configurado"}
        
        try:
            # Leer archivo de audio
            with open(audio_file_path, 'rb') as audio_file:
                # Subir archivo
                upload_response = requests.post(
                    'https://api.assemblyai.com/v2/upload',
                    headers={'authorization': self.api_key},
                    data=audio_file.read()
                )
                
                if upload_response.status_code != 200:
                    return {"error": f"Error en upload: {upload_response.text}"}
                
                upload_url = upload_response.json()['upload_url']
                
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
                    }
                )
                
                if transcript_response.status_code != 200:
                    return {"error": f"Error en transcripción: {transcript_response.text}"}
                
                transcript_id = transcript_response.json()['id']
                
                # Polling para obtener resultado
                while True:
                    polling_response = requests.get(
                        f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                        headers={'authorization': self.api_key}
                    )
                    
                    if polling_response.status_code != 200:
                        return {"error": f"Error en polling: {polling_response.text}"}
                    
                    polling_result = polling_response.json()
                    
                    if polling_result['status'] == 'completed':
                        return {
                            'success': True,
                            'text': polling_result['text'],
                            'confidence': polling_result.get('confidence', 1.0),
                            'duration': polling_result.get('audio_duration', 0)
                        }
                    elif polling_result['status'] == 'error':
                        return {"error": polling_result.get('error', 'Error desconocido')}
                    
                    time.sleep(2)
                    
        except Exception as e:
            return {"error": f"Error en transcripción: {str(e)}"}