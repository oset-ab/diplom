# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from ultralytics import YOLO
import os
from .config import Config # Assuming Config class is in app/config.py
from datetime import datetime
import logging

db = SQLAlchemy()
bcrypt = Bcrypt()
model = None # Global model variable

def create_app(config_class=Config):
    global model # Allow modification of the global model variable

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configure logging (using app.logger)
    if not app.debug:
        app.logger.setLevel(logging.INFO)
    else:
        app.logger.setLevel(logging.DEBUG) # More verbose in debug

    db.init_app(app)
    bcrypt.init_app(app)

    # --- YOLO Model Loading ---
    yolo_path = app.config['YOLO_MODEL_PATH']
    if os.path.exists(yolo_path):
        try:
            model = YOLO(yolo_path)
            app.logger.info(f"YOLO model loaded from {yolo_path}")
        except Exception as e:
            app.logger.error(f"Error loading YOLO model from path {yolo_path}: {e}", exc_info=True)
            model = None # Ensure model is None if local load fails
    else:
        app.logger.warning(f"Local YOLO model path not found: {yolo_path}")

    if model is None: # If local model failed or path didn't exist
        app.logger.info("Attempting to download and load default YOLO model (yolov8n.pt)...")
        try:
            model = YOLO('yolov8n.pt') # Standard model name
            app.logger.info("Default YOLO model yolov8n.pt loaded successfully.")
        except Exception as e:
            app.logger.error(f"Could not download or load default YOLO model: {e}", exc_info=True)
            model = None # Explicitly set to None if all attempts fail

    # --- Folder Creation ---
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    originals_path = os.path.join(app.config['UPLOAD_FOLDER'], 'originals')
    os.makedirs(originals_path, exist_ok=True)
    app.logger.debug(f"Upload folder ensured: {app.config['UPLOAD_FOLDER']}")
    app.logger.debug(f"Originals folder ensured: {originals_path}")

    # --- Blueprint Registration ---
    # This is where the routes defined in app/routes.py (using main_bp) are connected to the app.
    # Crucially, 'from .routes import main_bp' should only effectively happen ONCE
    # and main_bp itself should be defined only ONCE (in routes.py).
    from .routes import main_bp  # Import the blueprint instance
    app.register_blueprint(main_bp) # Register it ONCE

    # --- Debug: Print URL Map (as you had before for diagnosing BuildError) ---
    with app.app_context():
        app.logger.info("---- REGISTERED URL RULES ----")
        rules_found_target = False
        for rule in app.url_map.iter_rules():
            app.logger.info(f"Endpoint: {rule.endpoint}, Path: {rule.rule}, Methods: {list(rule.methods)}")
            if rule.endpoint == 'main.download_media_report':
                rules_found_target = True
        if not rules_found_target:
            app.logger.error("CRITICAL: Endpoint 'main.download_media_report' STILL NOT FOUND in url_map after registration attempt!")
        else:
            app.logger.info("INFO: Endpoint 'main.download_media_report' WAS found in url_map.")
        app.logger.info("------------------------------")
    # --- End Debug ---

    from . import models # Import models to ensure SQLAlchemy knows about them

    @app.context_processor
    def inject_now():
        return {'now': datetime.utcnow()}

    with app.app_context():
        db.create_all() # Create database tables if they don't exist
        app.logger.info("Database tables created/ensured.")

    return app