import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_very_secret_key_here_change_it'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///detections.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app/static/results')
    YOLO_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'yolov8n.pt') # Путь к модели

    # Убедимся, что папка для загрузок существует
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)