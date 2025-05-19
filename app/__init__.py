from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from ultralytics import YOLO
import os
from .config import Config

db = SQLAlchemy()
bcrypt = Bcrypt()
model = None  # Инициализируем позже


def create_app(config_class=Config):
    global model

    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    bcrypt.init_app(app)

    # Загрузка модели YOLO
    if os.path.exists(app.config['YOLO_MODEL_PATH']):
        try:
            model = YOLO(app.config['YOLO_MODEL_PATH'])
            print(f"YOLO model loaded from {app.config['YOLO_MODEL_PATH']}")
        except Exception as e:
            print(f"Error loading YOLO model from path: {e}")
            model = None

    if model is None:  # Если локальная не загрузилась или не существует
        print(f"Attempting to download and load default YOLO model (yolov8n.pt)...")
        try:
            model = YOLO('yolov8n.pt')  # Попытка скачать и загрузить стандартную
            print("Default YOLO model yolov8n.pt loaded successfully.")
        except Exception as e:
            print(f"Could not download or load default YOLO model: {e}")
            model = None  # Устанавливаем в None, если загрузка не удалась

    # Убедимся, что папка для загрузок существует
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'originals'), exist_ok=True)

    # Импорт и регистрация Blueprint
    from .routes import main_bp  # Используем относительный импорт
    app.register_blueprint(main_bp)

    from . import models  # Модели можно импортировать здесь

    with app.app_context():
        db.create_all()  # Создаем таблицы, если их нет

    return app