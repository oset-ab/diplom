# app/models.py
from datetime import datetime
from app import db, bcrypt

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, nullable=False, default=False) # <--- НОВОЕ ПОЛЕ

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
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)  # Путь относительно корня проекта или полный
    upload_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    media_type = db.Column(db.String(50), nullable=False)  # 'image', 'video', 'stream_record'
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Связь для каскадного удаления или просто для доступа
    detections = db.relationship('Detection', backref='media_source', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Media {self.filename}>'


class Detection(db.Model):
    __tablename__ = 'detections'
    id = db.Column(db.Integer, primary_key=True)
    media_id = db.Column(db.Integer, db.ForeignKey('media.id'), nullable=False)
    object_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bbox_x = db.Column(db.Integer, nullable=False)
    bbox_y = db.Column(db.Integer, nullable=False)
    bbox_width = db.Column(db.Integer, nullable=False)
    bbox_height = db.Column(db.Integer, nullable=False)
    detected_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # Дополнительно, если нужно сохранить результат кадра для видео/стрима
    result_frame_filename = db.Column(db.String(255), nullable=True)  # Имя файла с кадром с детекцией

    def __repr__(self):
        return f'<Detection {self.object_class} on Media ID {self.media_id}>'


class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    event = db.Column(db.String(100), nullable=False)  # "Login", "Upload", "StreamStart", "DetectionError"
    details = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Может быть Null, если системное событие

    def __repr__(self):
        return f'<Log {self.event} at {self.timestamp}>'


# Вспомогательная функция для логирования
def add_log_entry(event, details=None, user_id=None):
    try:
        log_entry = Log(event=event, details=details, user_id=user_id)
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        print(f"Error adding log entry: {e}")
        db.session.rollback()