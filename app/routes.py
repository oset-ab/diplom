# app/routes.py
from flask import (
    render_template, Response, request, redirect, url_for,
    jsonify, session, send_from_directory, flash, current_app, abort, Blueprint
)
from werkzeug.utils import secure_filename
import cv2
import os
import time
from datetime import datetime
from functools import wraps
import io  # Для CSV
import csv  # Для CSV

from app import db, bcrypt, model  # Импортируем из пакета app
from app.models import User, Media, Detection, Log, add_log_entry

main_bp = Blueprint('main', __name__)

# --- Глобальные переменные для стрима ---
cap = None
stream_active = False
stream_start_time = 0
stream_detected_objects_for_db = {}  # Словарь для сбора детекций {class_name: count}
stream_last_annotated_frame = None  # Для сохранения последнего кадра


# --- Декораторы ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Пожалуйста, войдите в систему для доступа к этой странице.', 'warning')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Пожалуйста, войдите в систему.', 'warning')
            return redirect(url_for('main.login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            # flash('У вас нет прав доступа к этой странице.', 'danger') # Можно убрать, чтобы не показывать явно
            abort(403)  # Ошибка "Forbidden"
        return f(*args, **kwargs)

    return decorated_function


# --- Маршруты аутентификации и обычные ---
@main_bp.route('/')
def welcome():
    return render_template('welcome.html')


@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким именем уже существует.', 'danger')
            return redirect(url_for('main.register'))

        new_user = User(username=username)
        new_user.set_password(password)

        # ВРЕМЕННЫЙ КОД: Сделать первого зарегистрированного пользователя админом
        # ЗАКОММЕНТИРУЙТЕ ИЛИ УДАЛИТЕ ПОСЛЕ СОЗДАНИЯ ПЕРВОГО АДМИНА!
        # if User.query.count() == 0:
        #     new_user.is_admin = True
        #     flash('Вы зарегистрированы как первый пользователь и назначены администратором.', 'info')

        db.session.add(new_user)
        db.session.commit()
        add_log_entry(event="UserRegistered", details=f"User: {username}", user_id=new_user.id)
        flash('Регистрация успешна! Теперь вы можете войти.', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html')


@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin  # Сохраняем статус админа в сессии
            add_log_entry(event="UserLogin", user_id=user.id)
            flash('Вход выполнен успешно!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            add_log_entry(event="LoginFailed", details=f"Attempted username: {username}")
            flash('Неверное имя пользователя или пароль.', 'danger')
            return redirect(url_for('main.login'))
    return render_template('login.html')


@main_bp.route('/logout')
@login_required
def logout():
    user_id = session.get('user_id')
    session.clear()
    if user_id:
        add_log_entry(event="UserLogout", user_id=user_id)
    flash('Вы вышли из системы.', 'info')
    return redirect(url_for('main.login'))


@main_bp.route('/dashboard')
@login_required
def dashboard():
    # user = User.query.get(session['user_id']) # Не обязательно, если username уже в сессии
    return render_template('dashboard.html',
                           username=session.get('username'),
                           is_admin=session.get('is_admin', False))


# --- Обработка видеопотока ---
def gen_frames(current_user_id):  # Принимаем user_id
    global stream_active, stream_start_time, cap, stream_detected_objects_for_db, stream_last_annotated_frame

    if model is None:
        print("YOLO model not loaded, cannot start stream.")
        # ... (код ошибки модели) ...
        return

    stream_active = True
    stream_start_time = time.time()
    stream_detected_objects_for_db.clear()
    stream_last_annotated_frame = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        # ... (код ошибки камеры) ...
        stream_active = False
        return

    # УБИРАЕМ ЛОГИРОВАНИЕ НАЧАЛА СТРИМА ОТСЮДА, ТАК КАК НЕТ КОНТЕКСТА ПРИЛОЖЕНИЯ
    # add_log_entry("StreamStart", user_id=current_user_id)
    print(f"Stream generation started for user: {current_user_id} (logging of start deferred or omitted)")

    while stream_active:
        success, frame = cap.read()
        if not success:
            print("Stream: failed to grab frame")
            break

        results = model(frame, verbose=False)

        annotated_frame_to_show = frame.copy()
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            annotated_frame_for_save = results[0].plot()
            annotated_frame_to_show = annotated_frame_for_save.copy()
            stream_last_annotated_frame = annotated_frame_for_save.copy()

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    object_class = model.names[class_id]
                    stream_detected_objects_for_db[object_class] = stream_detected_objects_for_db.get(object_class,
                                                                                                      0) + 1
        # else: # Если детекций нет, stream_last_annotated_frame не обновляется
        # pass

        ret, buffer = cv2.imencode('.jpg', annotated_frame_to_show)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if cap and cap.isOpened():
        cap.release()
    print("Stream generation loop ended.")


@main_bp.route('/video_feed')
@login_required
def video_feed():
    if model is None:
        flash("YOLO модель не загружена. Видеопоток не может быть запущен.", "danger")
        return redirect(url_for('main.dashboard'))

    user_id_for_stream = session.get('user_id')

    # Логируем старт стрима здесь, до того как Response начнет стримить
    # Это будет лог "попытки" старта стрима, фактический старт камеры и генерации кадров будет в gen_frames
    add_log_entry("StreamFeedRequested", f"User ID: {user_id_for_stream} requesting video feed.", user_id_for_stream)

    return Response(gen_frames(current_user_id=user_id_for_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@main_bp.route('/end_stream_session', methods=['POST'])
@login_required  # Здесь session доступна
def end_stream_session():
    global stream_active, stream_start_time, cap, stream_detected_objects_for_db, stream_last_annotated_frame
    print("Received end_stream_session request.")

    user_id_from_session = session['user_id']  # Гарантированно есть из-за @login_required

    was_ever_started = stream_start_time > 0
    stream_active = False

    if cap and cap.isOpened():
        cap.release()
        cap = None
        print("Camera released.")

    duration = 0
    if was_ever_started:
        duration = time.time() - stream_start_time
        print(f"Stream duration: {duration:.2f}s. Detected objects during stream: {stream_detected_objects_for_db}")

    if was_ever_started and duration > 1 and (
            stream_detected_objects_for_db or stream_last_annotated_frame is not None):
        try:
            timestamp_obj = datetime.fromtimestamp(stream_start_time)
            timestamp_str = timestamp_obj.strftime('%Y%m%d_%H%M%S')

            # Используем user_id_from_session для имени файла
            stream_media_filename_base = f"stream_capture_{user_id_from_session}_{timestamp_str}"

            media_filepath_for_db = "N/A_stream_no_frame_saved"
            result_image_filename_for_db = None

            if stream_last_annotated_frame is not None:
                result_image_filename_for_db = stream_media_filename_base + ".jpg"
                saved_frame_path = os.path.join(current_app.config['UPLOAD_FOLDER'], result_image_filename_for_db)
                cv2.imwrite(saved_frame_path, stream_last_annotated_frame)
                media_filepath_for_db = saved_frame_path
                print(f"Saved stream frame to: {saved_frame_path}")
            else:
                print("No annotated frame to save for this stream session.")

            media_entry = Media(
                filename=result_image_filename_for_db if result_image_filename_for_db else stream_media_filename_base,
                file_path=media_filepath_for_db,
                media_type='stream_capture',
                user_id=user_id_from_session,  # Используем user_id из сессии
                upload_time=timestamp_obj
            )
            db.session.add(media_entry)
            db.session.flush()

            if stream_detected_objects_for_db:
                for obj_class, count in stream_detected_objects_for_db.items():
                    detection = Detection(
                        media_id=media_entry.id,
                        object_class=obj_class,
                        confidence=-1,
                        bbox_x=-1, bbox_y=-1, bbox_width=-1, bbox_height=-1,
                        detected_at=datetime.now(),
                        result_frame_filename=result_image_filename_for_db
                    )
                    db.session.add(detection)

            db.session.commit()
            objects_count_str = ", ".join([f"{k}({v})" for k, v in stream_detected_objects_for_db.items()])
            log_details = f"Media ID: {media_entry.id}, Duration: {duration:.2f}s, Objects: {objects_count_str}"
            add_log_entry("StreamProcessed", log_details, user_id_from_session)  # Используем user_id из сессии
            flash(
                f"Сессия стрима сохранена. {('Обнаружено: ' + objects_count_str) if objects_count_str else 'Объектов не обнаружено.'}",
                "success")
            response_message = 'Stream ended and data saved.'
        except Exception as e:
            db.session.rollback()
            print(f"Error saving stream session to DB: {e}")
            add_log_entry("StreamProcessingError", f"Error: {e}", user_id_from_session)  # Используем user_id из сессии
            flash("Ошибка при сохранении сессии стрима в БД.", "danger")
            response_message = 'Stream ended, but error saving data.'
    elif was_ever_started:
        add_log_entry("StreamEnd", f"Duration: {duration:.2f}s. No significant detections or frame to save.",
                      user_id_from_session)  # Используем user_id из сессии
        response_message = 'Stream ended. No data to save (too short, no detections, or no frame).'
        flash("Сессия стрима завершена. Нет данных для сохранения (слишком короткая или нет детекций/кадра).", "info")
    else:
        response_message = 'Stream was not active or already stopped.'
        print("Stream was not considered active for saving.")

    stream_start_time = 0
    stream_detected_objects_for_db.clear()
    stream_last_annotated_frame = None
    print("Global stream variables reset.")

    return jsonify({'status': 'success', 'message': response_message, 'duration': round(duration, 2)})


# --- Загрузка медиа ---
@main_bp.route('/upload_media', methods=['POST'])
@login_required
def upload_media():
    if model is None:
        flash("YOLO модель не загружена. Обработка медиа невозможна.", "danger")
        return jsonify({'error': 'YOLO model not loaded'}), 500

    if 'media' not in request.files:
        flash('Файл не был предоставлен.', 'warning')
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['media']
    if file.filename == '':
        flash('Файл не выбран.', 'warning')
        return jsonify({'error': 'No selected file'}), 400

    # Инициализация переменных для return jsonify на случай ошибки до их определения
    unique_filename_for_response = "N/A"
    result_filename_for_response = "N/A"
    media_id_for_response = None
    objects_str_for_response = "N/A"
    duration_for_response = 0
    message_for_response = "Ошибка обработки файла."
    result_url_for_response = None

    if file:
        original_filename = secure_filename(file.filename)  # Имя файла от пользователя
        original_save_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'originals')
        os.makedirs(original_save_dir, exist_ok=True)

        base, ext = os.path.splitext(original_filename)
        unique_filename_for_response = f"{base}_{session.get('user_id', 0)}_{int(time.time())}{ext}"
        original_filepath = os.path.join(original_save_dir, unique_filename_for_response)

        try:
            file.save(original_filepath)
        except Exception as e:
            flash(f"Ошибка сохранения файла: {e}", "danger")
            return jsonify({'error': f'Failed to save file: {e}'}), 500

        media_type_str = ''
        if original_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            media_type_str = 'image'
        elif original_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            media_type_str = 'video'
        else:
            try:
                os.remove(original_filepath)  # Удаляем, если сохранили, но тип не тот
            except OSError:
                pass  # Файл мог не сохраниться
            flash('Неподдерживаемый тип файла.', 'danger')
            return jsonify({'error': 'Unsupported file type'}), 400

        media_entry = Media(
            filename=unique_filename_for_response,  # Сохраняем уникальное имя оригинала
            file_path=original_filepath,  # Путь к оригиналу
            media_type=media_type_str,
            user_id=session['user_id']
        )
        db.session.add(media_entry)
        db.session.commit()
        media_id_for_response = media_entry.id

        add_log_entry("MediaUpload", f"Media ID: {media_id_for_response}, Filename: {unique_filename_for_response}",
                      session['user_id'])

        start_processing_time = time.time()

        result_filename_for_response = 'result_' + unique_filename_for_response
        result_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], result_filename_for_response)

        detected_objects_summary = {}
        annotated_frame = None  # Инициализируем

        try:
            if media_type_str == 'image':
                img = cv2.imread(original_filepath)
                if img is None:
                    flash('Не удалось прочитать изображение.', 'danger')
                    add_log_entry("ProcessingError", f"Failed to read image: {unique_filename_for_response}",
                                  session['user_id'])
                    db.session.delete(media_entry)
                    db.session.commit()
                    return jsonify({'error': 'Could not read image file'}), 500

                results = model(img, verbose=False)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = img.copy()  # Копия оригинала, если нет детекций

                cv2.imwrite(result_filepath, annotated_frame)

                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        obj_class = model.names[class_id]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        detection = Detection(
                            media_id=media_id_for_response,
                            object_class=obj_class,
                            confidence=round(confidence, 4),
                            bbox_x=x1, bbox_y=y1,
                            bbox_width=x2 - x1, bbox_height=y2 - y1,
                            result_frame_filename=result_filename_for_response
                        )
                        db.session.add(detection)
                        detected_objects_summary[obj_class] = detected_objects_summary.get(obj_class, 0) + 1
                db.session.commit()

            elif media_type_str == 'video':
                # Обрабатываем только первый кадр для видео
                video_cap = cv2.VideoCapture(original_filepath)
                success, frame = video_cap.read()
                if success:
                    results = model(frame, verbose=False)
                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame.copy()

                    cv2.imwrite(result_filepath, annotated_frame)

                    for r in results:
                        for box in r.boxes:
                            # ... (аналогично image, сохраняем детекции) ...
                            class_id = int(box.cls[0])
                            obj_class = model.names[class_id]
                            confidence = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detection = Detection(
                                media_id=media_id_for_response,
                                object_class=obj_class,
                                confidence=round(confidence, 4),
                                bbox_x=x1, bbox_y=y1,
                                bbox_width=x2 - x1, bbox_height=y2 - y1,
                                result_frame_filename=result_filename_for_response
                            )
                            db.session.add(detection)
                            detected_objects_summary[obj_class] = detected_objects_summary.get(obj_class, 0) + 1
                    db.session.commit()
                else:
                    flash('Не удалось прочитать первый кадр видеофайла.', 'danger')
                    add_log_entry("ProcessingError", f"Failed to read video frame: {unique_filename_for_response}",
                                  session['user_id'])
                    db.session.delete(media_entry)
                    db.session.commit()
                    return jsonify({'error': 'Could not read video file frame'}), 500
                video_cap.release()
        except Exception as e_proc:
            db.session.rollback()  # Откатываем детекции, если ошибка в процессе
            flash(f"Ошибка во время обработки медиа: {e_proc}", "danger")
            add_log_entry("ProcessingError", f"Media ID: {media_id_for_response}, Error: {e_proc}", session['user_id'])
            # Запись Media остается, но без детекций
            return jsonify({'error': f'Error during media processing: {e_proc}'}), 500

        duration_for_response = time.time() - start_processing_time
        add_log_entry("MediaProcessed", f"Media ID: {media_id_for_response}, Duration: {duration_for_response:.2f}s",
                      session['user_id'])

        objects_str_for_response = ", ".join([f"{k}({v})" for k, v in
                                              detected_objects_summary.items()]) if detected_objects_summary else "Объекты не найдены"

        message_for_response = f'Файл {unique_filename_for_response} обработан. {("Найдено: " + objects_str_for_response) if detected_objects_summary else "Объекты не найдены."}'
        flash(message_for_response, 'success')
        result_url_for_response = url_for('main.results_file',
                                          filename=result_filename_for_response) if result_filename_for_response else None

        return jsonify({
            'message': message_for_response,
            'original_filename': unique_filename_for_response,
            'result_filename': result_filename_for_response,
            'media_id': media_id_for_response,
            'detected_objects': objects_str_for_response,
            'processing_time': round(duration_for_response, 2),
            'result_url': result_url_for_response
        })

    # Этот блок маловероятен, если предыдущие проверки на file и file.filename пройдены
    flash('Произошла непредвиденная ошибка при загрузке файла.', 'danger')
    return jsonify({
        'error': 'Unexpected error during file upload',
        'message': message_for_response,
        'original_filename': unique_filename_for_response,
        'result_filename': result_filename_for_response,
        'media_id': media_id_for_response,
        'detected_objects': objects_str_for_response,
        'processing_time': round(duration_for_response, 2),
        'result_url': result_url_for_response
    }), 500


# --- Просмотр результатов и история ---
@main_bp.route('/results/<path:filename>')
@login_required
def results_file(filename):
    # Проверка, существует ли файл, прежде чем отдавать
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        abort(404)
    # Дополнительная проверка прав доступа, если нужна (например, админ может смотреть все, юзер только свои)
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@main_bp.route('/history')
@login_required
def history_page():
    page = request.args.get('page', 1, type=int)
    user_media_query = Media.query.filter_by(user_id=session['user_id'])

    media_type_filter = request.args.get('media_type')
    if media_type_filter:
        user_media_query = user_media_query.filter(Media.media_type == media_type_filter)

    user_media = user_media_query.order_by(Media.upload_time.desc()) \
        .paginate(page=page, per_page=5, error_out=False)  # Уменьшил per_page для теста

    media_with_detections = []
    for m_item in user_media.items:
        detections = Detection.query.filter_by(media_id=m_item.id).all()
        detected_summary = {}
        first_result_frame = None

        if m_item.media_type == 'stream_capture':
            # Для стрима файл результата это сам m_item.filename (если он есть и сохранен)
            if m_item.filename and m_item.filename != f"stream_data_{m_item.upload_time.strftime('%Y%m%d_%H%M%S')}" and \
                    os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], m_item.filename)):
                first_result_frame = m_item.filename

            for det in detections:
                detected_summary[det.object_class] = detected_summary.get(det.object_class, 0) + 1
        else:  # Для image и video
            # Ищем первый result_frame_filename из детекций
            for det in detections:
                detected_summary[det.object_class] = detected_summary.get(det.object_class, 0) + 1
                if not first_result_frame and det.result_frame_filename:
                    first_result_frame = det.result_frame_filename

            # Если детекций не было, или у них нет result_frame_filename,
            # но это image/video, ищем файл 'result_' + m_item.filename
            if not first_result_frame:
                potential_result_file = 'result_' + m_item.filename
                if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], potential_result_file)):
                    first_result_frame = potential_result_file

        media_with_detections.append({
            'media': m_item,
            'detections_summary_str': ", ".join(
                [f"{k}({v})" for k, v in detected_summary.items()]) if detected_summary else "Объекты не найдены",
            'first_result_frame': first_result_frame,
            'is_stream_capture': m_item.media_type == 'stream_capture'
        })

    return render_template('history_page.html',
                           media_items=media_with_detections,
                           pagination=user_media,
                           current_media_type_filter=media_type_filter)


# --- Отчеты ---
@main_bp.route('/download_report/media/<int:media_id>')
@login_required
def download_media_report(media_id):
    media_item = Media.query.get_or_404(media_id)
    if media_item.user_id != session['user_id'] and not session.get('is_admin'):
        abort(403)

    detections = Detection.query.filter_by(media_id=media_id).all()
    if not detections:
        flash("Нет детекций для этого медиафайла.", "info")
        return redirect(request.referrer or url_for('main.history_page'))

    si = io.StringIO()
    cw = csv.writer(si)

    header = ['ID Детекции', 'Класс Объекта']
    if media_item.media_type != 'stream_capture':
        header.extend(['Уверенность', 'X', 'Y', 'Ширина', 'Высота'])
    header.extend(['Время Детекции', 'Файл результата кадра'])
    cw.writerow(header)

    for det in detections:
        row = [det.id, det.object_class]
        if media_item.media_type != 'stream_capture':
            row.extend([
                f"{det.confidence:.2f}" if det.confidence is not None and det.confidence != -1 else "N/A",
                det.bbox_x if det.bbox_x != -1 else "N/A",
                det.bbox_y if det.bbox_y != -1 else "N/A",
                det.bbox_width if det.bbox_width != -1 else "N/A",
                det.bbox_height if det.bbox_height != -1 else "N/A",
            ])
        row.extend([
            det.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            det.result_frame_filename or "N/A"
        ])
        cw.writerow(row)

    output = si.getvalue()
    response = Response(output, mimetype="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=report_media_{media_item.filename}_{media_id}.csv"
    return response


# --- Админ-панель ---
@main_bp.route('/admin')
@admin_required
def admin_dashboard():
    users = User.query.order_by(User.username).all()
    return render_template('admin/dashboard.html', users=users)


@main_bp.route('/admin/user_history/<int:user_id>')
@admin_required
def admin_user_history(user_id):
    target_user = User.query.get_or_404(user_id)
    page = request.args.get('page', 1, type=int)

    user_media_query = Media.query.filter_by(user_id=target_user.id)
    media_type_filter = request.args.get('media_type')
    if media_type_filter:
        user_media_query = user_media_query.filter(Media.media_type == media_type_filter)

    user_media = user_media_query.order_by(Media.upload_time.desc()) \
        .paginate(page=page, per_page=5, error_out=False)  # Уменьшил per_page

    media_with_detections = []
    # Копируем логику сборки media_with_detections из history_page
    for m_item in user_media.items:
        detections = Detection.query.filter_by(media_id=m_item.id).all()
        detected_summary = {}
        first_result_frame = None
        if m_item.media_type == 'stream_capture':
            if m_item.filename and m_item.filename != f"stream_data_{m_item.upload_time.strftime('%Y%m%d_%H%M%S')}" and \
                    os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], m_item.filename)):
                first_result_frame = m_item.filename
            for det in detections:
                detected_summary[det.object_class] = detected_summary.get(det.object_class, 0) + 1
        else:
            for det in detections:
                detected_summary[det.object_class] = detected_summary.get(det.object_class, 0) + 1
                if not first_result_frame and det.result_frame_filename:
                    first_result_frame = det.result_frame_filename
            if not first_result_frame:
                potential_result_file = 'result_' + m_item.filename
                if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], potential_result_file)):
                    first_result_frame = potential_result_file

        media_with_detections.append({
            'media': m_item,
            'detections_summary_str': ", ".join(
                [f"{k}({v})" for k, v in detected_summary.items()]) if detected_summary else "Объекты не найдены",
            'first_result_frame': first_result_frame,
            'is_stream_capture': m_item.media_type == 'stream_capture'
        })

    return render_template('admin/user_history_page.html',
                           target_user=target_user,
                           media_items=media_with_detections,
                           pagination=user_media,
                           current_media_type_filter=media_type_filter)


@main_bp.route('/admin/download_user_report/<int:user_id>')
@admin_required
def admin_download_user_report(user_id):
    target_user = User.query.get_or_404(user_id)
    # Получаем все медиа пользователя, затем для каждого медиа его детекции
    media_items_by_user = Media.query.filter_by(user_id=user_id).order_by(Media.upload_time.asc()).all()

    if not media_items_by_user:
        flash(f"Нет данных для отчета по пользователю {target_user.username}.", "info")
        return redirect(url_for('main.admin_user_history', user_id=user_id))

    si = io.StringIO()
    cw = csv.writer(si)

    # Заголовок для полного отчета
    cw.writerow(['ID Медиа', 'Имя Файла Оригинала', 'Тип Медиа', 'Время Загрузки Медиа',
                 'ID Детекции', 'Класс Объекта', 'Уверенность', 'X', 'Y', 'Ширина', 'Высота',
                 'Время Детекции Объекта', 'Файл кадра с результатом детекции'])

    for media in media_items_by_user:
        detections_for_media = Detection.query.filter_by(media_id=media.id).all()
        if not detections_for_media:
            # Записываем информацию о медиа, даже если детекций нет
            cw.writerow([media.id, media.filename, media.media_type, media.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
                         'N/A', 'Нет детекций', '', '', '', '', '', '', ''])
        else:
            for det in detections_for_media:
                is_stream_det = media.media_type == 'stream_capture'
                cw.writerow([
                    media.id, media.filename, media.media_type, media.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
                    det.id, det.object_class,
                    f"{det.confidence:.2f}" if not is_stream_det and det.confidence is not None else (
                        "N/A (Stream)" if is_stream_det else "N/A"),
                    det.bbox_x if not is_stream_det and det.bbox_x != -1 else "N/A",
                    det.bbox_y if not is_stream_det and det.bbox_y != -1 else "N/A",
                    det.bbox_width if not is_stream_det and det.bbox_width != -1 else "N/A",
                    det.bbox_height if not is_stream_det and det.bbox_height != -1 else "N/A",
                    det.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
                    det.result_frame_filename or "N/A"
                ])

    output = si.getvalue()
    response = Response(output, mimetype="text/csv")
    # Убедимся, что имя файла корректно для разных ОС
    safe_username = "".join(c if c.isalnum() else "_" for c in target_user.username)
    report_filename = f"full_report_user_{safe_username}_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers[
        "Content-Disposition"] = f"attachment; filename=\"{report_filename}\""  # Кавычки для имен с пробелами
    return response

# API эндпоинты (если были, то здесь)
# @main_bp.route('/api/history_data') ...
# @main_bp.route('/api/logs') ...