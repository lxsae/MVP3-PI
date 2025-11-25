# 🚀 Guía de Despliegue - Sistema de Control de Asistencia

Este documento describe cómo desplegar el proyecto en diferentes entornos.

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- PostgreSQL (opcional, por defecto usa SQLite)
- Cuenta en AssemblyAI (para transcripción de voz)

## 🛠️ Configuración del Entorno

1. **Clonar el repositorio y entrar al directorio:**
   ```bash
   git clone <url-del-repo>
   cd MVP3-PI
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto (puedes copiar `.env.example`):
   ```env
   # Clave secreta para sesiones (cambiar en producción)
   SECRET_KEY=tu_secreto_super_seguro
   
   # API Key de AssemblyAI (Obligatoria para voz)
   ASSEMBLYAI_API_KEY=tu_api_key_aqui
   
   # Base de datos (Opcional, por defecto SQLite local)
   # SUPABASE_DATABASE_URL=postgresql://usuario:password@host:port/dbname
   
   # Modo Debug (False en producción)
   DEBUG=False
   ```

## 🚀 Despliegue en Render.com (Opción Gratuita)

Render es excelente para proyectos Python/Flask.

1. Crea una cuenta en [Render.com](https://render.com).
2. Haz clic en **"New"** -> **"Web Service"**.
3. Conecta tu repositorio de GitHub.
4. Configura el servicio:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. En la sección **"Environment Variables"**, añade las variables de tu archivo `.env`:
   - `ASSEMBLYAI_API_KEY`
   - `SECRET_KEY`
   - `PYTHON_VERSION` (ej. 3.9.0)

## ☁️ Despliegue en Heroku

1. Instala Heroku CLI y haz login: `heroku login`
2. Crea la app:
   ```bash
   heroku create nombre-de-tu-app
   ```
3. Añade buildpack de apt (para librerías de sistema si son necesarias, como `libgl1` para OpenCV):
   ```bash
   heroku buildpacks:add --index 1 heroku-community/apt
   ```
   *Nota: Crea un archivo `Aptfile` con el contenido `libgl1-mesa-glx` y `libglib2.0-0` si tienes problemas con OpenCV.*

4. Configura variables:
   ```bash
   heroku config:set ASSEMBLYAI_API_KEY=tu_api_key
   heroku config:set SECRET_KEY=tu_secreto
   ```
5. Despliega:
   ```bash
   git push heroku main
   ```

## 🐧 Despliegue en Servidor Linux (VPS / EC2)

1. Instala dependencias del sistema (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
   ```
2. Sigue los pasos de "Configuración del Entorno".
3. Usa **Gunicorn** como servidor de producción:
   ```bash
   pip install gunicorn
   gunicorn --workers 3 --bind 0.0.0.0:8000 app:app
   ```
4. (Opcional) Configura Nginx como proxy reverso para manejar HTTPS y servir estáticos.

## 📝 Notas Importantes

- **OpenCV en la nube:** Algunas plataformas pueden dar error con `cv2` si faltan librerías gráficas. Asegúrate de instalar `opencv-python-headless` en lugar de `opencv-python` en `requirements.txt` si no necesitas mostrar ventanas de GUI en el servidor (que es lo normal en servidores web).
- **HTTPS:** Para que el micrófono funcione en dispositivos móviles y navegadores modernos, **ES OBLIGATORIO usar HTTPS** (excepto en localhost). Render y Heroku lo proveen automáticamente.

