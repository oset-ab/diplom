# app/models.py
from datetime import datetime
from app import db, bcrypt

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

    media_files = db.relationship('Media', backref='uploader', lazy=True)
    logs = db.relationship('Log', backref='user_event', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Media(db.Model):
    __tablename__ = 'media'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False) # Original uploaded name or generated stream video name
    file_path = db.Column(db.String(300), nullable=False) # Path to the primary file (original or stream video)
    upload_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    media_type = db.Column(db.String(50), nullable=False) # 'image', 'video' (uploaded), 'stream_capture' (now a video)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Detections can now be detailed (for images, stream_capture videos) or summary (for uploaded processed videos)
    detections = db.relationship('Detection', backref='media_source', lazy=True, cascade="all, delete-orphan")

    @property
    def media_type_rus(self):
        if self.media_type == 'image':
            return 'Изображение'
        elif self.media_type == 'video': # Uploaded video, processed into another video
            return 'Видео (загруженное)'
        elif self.media_type == 'stream_capture': # Live stream recorded as video
            return 'Видео (захват стрима)'
        return self.media_type

    def __repr__(self):
        return f'<Media {self.filename}>'

class Detection(db.Model):
    __tablename__ = 'detections'
    id = db.Column(db.Integer, primary_key=True)
    media_id = db.Column(db.Integer, db.ForeignKey('media.id', ondelete='CASCADE'), nullable=False)
    object_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False) # Actual confidence OR count for uploaded video summary
    bbox_x = db.Column(db.Integer, nullable=False) # -2 for uploaded video summary, actual for others
    bbox_y = db.Column(db.Integer, nullable=False)
    bbox_width = db.Column(db.Integer, nullable=False)
    bbox_height = db.Column(db.Integer, nullable=False)
    detected_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # For images: result_img_... (snapshot with bboxes)
    # For uploaded videos: processed_... (the fully processed video file itself, detections are summary)
    # For stream_captures (now videos): the stream_capture_USERID_TIMESTAMP.mp4 file (detections are detailed)
    result_frame_filename = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<Detection {self.object_class} on Media ID {self.media_id}>'

class Log(db.Model):
    # ... (no changes to Log model) ...
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    event = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    def __repr__(self):
        return f'<Log {self.event} at {self.timestamp}>'

def add_log_entry(event, details=None, user_id=None):
    # ... (no changes to add_log_entry) ...
    try:
        log_entry = Log(event=event, details=details, user_id=user_id)
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        # In a real app, use Flask's logger: current_app.logger.error(f"Error adding log entry: {e}")
        print(f"Error adding log entry: {e}")
        db.session.rollback()