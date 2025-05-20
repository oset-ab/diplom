# app/routes.py
from flask import (
    render_template, Response, request, redirect, url_for,
    jsonify, session, send_from_directory, flash, current_app, abort, Blueprint,
    stream_with_context, make_response
)
from werkzeug.utils import secure_filename
import cv2
import os
import time
from datetime import datetime
from functools import wraps
import io
import csv
import threading
import json
import base64
import subprocess
import re

from app import db, bcrypt, model
from app.models import User, Media, Detection, Log, add_log_entry

main_bp = Blueprint('main', __name__)

_stream_lock = threading.Lock()
_active_stream_viewers = set()
_global_cap = None
_user_stream_session_starts = {}
_user_video_writers = {}
_user_session_detections = {}
_user_session_filenames = {}

DEFAULT_FPS = 25.0
LIVE_STREAM_RECORD_FPS = 10.0


# --- Вспомогательная функция для постобработки видео с FFMPEG ---
def post_process_video_with_ffmpeg(video_path, logger):
    logger.info(f"FFMPEG: Entered post_process_video_with_ffmpeg for: {video_path}")
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        logger.warning(f"FFMPEG: SKIPPED - File {video_path} is missing or empty before processing.")
        return False
    original_size = os.path.getsize(video_path)
    logger.info(f"FFMPEG: Original file size for {video_path}: {original_size} bytes.")
    temp_fixed_video_path = video_path + "_tempfixed.mp4"
    success = False
    try:
        if os.path.exists(temp_fixed_video_path):
            logger.warning(f"FFMPEG: Removing existing temp file: {temp_fixed_video_path}")
            os.remove(temp_fixed_video_path)
        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-c:v', 'libx264',  # Перекодировать видео
            '-preset', 'ultrafast',  # Максимально быстро для теста (качество ниже)
            '-profile:v', 'baseline',  # Самый совместимый профиль
            '-pix_fmt', 'yuv420p',  # Для максимальной совместимости
            '-c:a', 'aac',  # Перекодировать аудио (если есть)
            '-movflags', '+faststart',
            temp_fixed_video_path
        ]
        logger.info(f"FFMPEG: Running command: {' '.join(ffmpeg_command)}")
        process = subprocess.run(ffmpeg_command, check=False, capture_output=True, text=True, timeout=300)
        if process.returncode == 0:
            if os.path.exists(temp_fixed_video_path) and os.path.getsize(temp_fixed_video_path) > 0:
                logger.info(
                    f"FFMPEG: Temp file {temp_fixed_video_path} created successfully. Size: {os.path.getsize(temp_fixed_video_path)} bytes.")
                try:
                    os.remove(video_path)
                    os.rename(temp_fixed_video_path, video_path)
                    logger.info(f"FFMPEG: Video {video_path} successfully post-processed and replaced.")
                    success = True
                except Exception as e_file_op:
                    logger.error(f"FFMPEG: ERROR during file rename/delete for {video_path}: {e_file_op}",
                                 exc_info=True)
                    if not os.path.exists(video_path) and os.path.exists(temp_fixed_video_path):
                        os.rename(temp_fixed_video_path, video_path)
                    elif os.path.exists(temp_fixed_video_path):
                        os.remove(temp_fixed_video_path)
            else:
                logger.error(
                    f"FFMPEG: Temp file {temp_fixed_video_path} NOT created or empty after successful ffmpeg command for {video_path}.")
                if os.path.exists(temp_fixed_video_path): os.remove(temp_fixed_video_path)
        else:
            logger.error(f"FFMPEG: PROCESSING FAILED for {video_path}. Return code: {process.returncode}")
            logger.error(f"FFMPEG stderr: {process.stderr.strip() if process.stderr else 'N/A'}")
            if os.path.exists(temp_fixed_video_path): os.remove(temp_fixed_video_path)
    except subprocess.TimeoutExpired:
        logger.error(f"FFMPEG: Timeout for {video_path}.")
        if os.path.exists(temp_fixed_video_path): os.remove(temp_fixed_video_path)
    except FileNotFoundError:
        logger.error("FFMPEG: Command 'ffmpeg' not found. Ensure it's installed and in system PATH.")
    except Exception as e:
        logger.error(f"FFMPEG: Exception during post-processing for {video_path}: {e}", exc_info=True)
        if os.path.exists(temp_fixed_video_path): os.remove(temp_fixed_video_path)
    logger.info(f"FFMPEG: Exiting post_process_video_with_ffmpeg for {video_path}. Success: {success}")
    return success


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
        if not user or not user.is_admin: abort(403)
        return f(*args, **kwargs)

    return decorated_function


# --- Маршруты аутентификации и обычные ---
@main_bp.route('/')
def welcome(): return render_template('welcome.html')


@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        username = request.form['username'];
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким именем уже существует.', 'danger');
            return redirect(url_for('main.register'))
        new_user = User(username=username);
        new_user.set_password(password)
        db.session.add(new_user);
        db.session.commit()
        add_log_entry(event="UserRegistered", details=f"User: {username}", user_id=new_user.id)
        flash('Регистрация успешна! Теперь вы можете войти.', 'success');
        return redirect(url_for('main.login'))
    return render_template('register.html')


@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        username = request.form['username'];
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id;
            session['username'] = user.username;
            session['is_admin'] = user.is_admin
            add_log_entry(event="UserLogin", user_id=user.id)
            flash('Вход выполнен успешно!', 'success');
            return redirect(url_for('main.dashboard'))
        else:
            add_log_entry(event="LoginFailed", details=f"Attempted username: {username}")
            flash('Неверное имя пользователя или пароль.', 'danger')
    return render_template('login.html')


@main_bp.route('/logout')
@login_required
def logout():
    user_id = session.get('user_id')
    with _stream_lock:
        if user_id in _active_stream_viewers: _active_stream_viewers.discard(user_id)
        writer = _user_video_writers.pop(user_id, None)
        if writer and writer.isOpened(): writer.release()
        _user_session_detections.pop(user_id, None);
        _user_session_filenames.pop(user_id, None)
        _user_stream_session_starts.pop(user_id, None);
        manage_camera_capture()
    session.clear()
    if user_id: add_log_entry(event="UserLogout", user_id=user_id)
    flash('Вы вышли из системы.', 'info');
    return redirect(url_for('main.login'))


@main_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'), is_admin=session.get('is_admin', False))


# --- Обработка видеопотока с камеры ---
def manage_camera_capture():
    global _global_cap
    with _stream_lock:
        if _active_stream_viewers and (_global_cap is None or not _global_cap.isOpened()):
            _global_cap = cv2.VideoCapture(0)
            if _global_cap.isOpened():
                current_app.logger.info("Global camera started.")
            else:
                current_app.logger.error("Error: Could not open global camera."); _global_cap = None
        elif not _active_stream_viewers and (_global_cap is not None and _global_cap.isOpened()):
            _global_cap.release();
            _global_cap = None
            current_app.logger.info("Global camera released as no viewers are active.")


def generate_camera_frames_for_user(user_id_generating_for):
    global _global_cap, _user_video_writers, _user_session_detections, _user_session_filenames, model
    current_app.logger.info(f"Attempting to start stream generation for user: {user_id_generating_for}")
    if model is None: current_app.logger.error(f"User {user_id_generating_for}: YOLO model not loaded."); return
    user_writer = None;
    video_session_filename_for_this_session = None
    with _stream_lock:
        _active_stream_viewers.add(user_id_generating_for)
        _user_stream_session_starts[user_id_generating_for] = time.time()
        _user_session_detections[user_id_generating_for] = []
    manage_camera_capture()
    with _stream_lock:
        if _global_cap and _global_cap.isOpened():
            frame_w = int(_global_cap.get(cv2.CAP_PROP_FRAME_WIDTH));
            frame_h = int(_global_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_w == 0 or frame_h == 0: frame_w, frame_h = 640, 480; current_app.logger.warning(
                f"User {user_id_generating_for}: Camera 0x0, using defaults {frame_w}x{frame_h}.")
            s_dt = datetime.fromtimestamp(_user_stream_session_starts[user_id_generating_for])
            fn_base = f"stream_capture_{user_id_generating_for}_{s_dt.strftime('%Y%m%d_%H%M%S')}"
            video_session_filename_for_this_session = f"{fn_base}.mp4"
            vid_path = os.path.join(current_app.config['UPLOAD_FOLDER'], video_session_filename_for_this_session)
            _user_session_filenames[user_id_generating_for] = video_session_filename_for_this_session
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            try:
                uw_instance = cv2.VideoWriter(vid_path, fourcc, float(LIVE_STREAM_RECORD_FPS), (frame_w, frame_h))
                if not uw_instance.isOpened():
                    current_app.logger.error(f"User {user_id_generating_for}: Failed VideoWriter at {vid_path}.")
                else:
                    user_writer = uw_instance; _user_video_writers[
                        user_id_generating_for] = user_writer; current_app.logger.info(
                        f"User {user_id_generating_for}: VideoWriter opened: {video_session_filename_for_this_session}")
            except Exception as e:
                current_app.logger.error(f"User {user_id_generating_for}: VideoWriter Exception: {e}", exc_info=True)
        else:
            current_app.logger.warning(f"User {user_id_generating_for}: Global cam N/A for VideoWriter.")
    current_app.logger.info(f"Stream loop starting for user: {user_id_generating_for}")
    try:
        while True:
            f_cam = None;
            ann_disp = None;
            ann_vid = None;
            curr_dets = []
            with _stream_lock:
                if user_id_generating_for not in _active_stream_viewers: break
                if _global_cap is None or not _global_cap.isOpened(): break
                success, f_cam = _global_cap.read()
            if not success or f_cam is None: time.sleep(0.1); continue
            results = model(f_cam, verbose=False);
            ann_disp = f_cam.copy();
            ann_vid = f_cam.copy()
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                ann_plot = results[0].plot();
                ann_disp = ann_plot.copy();
                ann_vid = ann_plot.copy()
                for r in results:
                    for box in r.boxes:
                        clsid = int(box.cls[0]);
                        obj_cls = model.names[clsid];
                        conf = float(box.conf[0]);
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        curr_dets.append({"object_class": obj_cls, "confidence": conf, "bbox_x": x1, "bbox_y": y1,
                                          "bbox_width": x2 - x1, "bbox_height": y2 - y1,
                                          "detected_at": datetime.utcnow()})
            if user_writer and user_writer.isOpened(): user_writer.write(ann_vid)
            if curr_dets:
                with _stream_lock:
                    if user_id_generating_for in _user_session_detections: _user_session_detections[
                        user_id_generating_for].extend(curr_dets)
            ret, buf = cv2.imencode('.jpg', ann_disp)
            if not ret: current_app.logger.error(f"User {user_id_generating_for}: JPG encode fail."); continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    except GeneratorExit:
        current_app.logger.info(f"GeneratorExit for user {user_id_generating_for}.")
    except Exception as e:
        current_app.logger.error(f"Exception in generate_camera_frames for {user_id_generating_for}: {e}",
                                 exc_info=True)
    finally:
        current_app.logger.info(
            f"Finally for user {user_id_generating_for}'s stream (video: {video_session_filename_for_this_session}).")
        if user_writer and user_writer.isOpened():
            user_writer.release()
            current_app.logger.info(
                f"User {user_id_generating_for}: VideoWriter for {video_session_filename_for_this_session or 'Unk'} released.")
            if video_session_filename_for_this_session:
                full_vid_path = os.path.join(current_app.config['UPLOAD_FOLDER'],
                                             video_session_filename_for_this_session)
                if os.path.exists(full_vid_path) and os.path.getsize(full_vid_path) > 0:
                    current_app.logger.info(
                        f"User {user_id_generating_for}: Attempting ffmpeg for live stream: {full_vid_path}")
                    post_process_video_with_ffmpeg(full_vid_path, current_app.logger)
                else:
                    current_app.logger.warning(
                        f"User {user_id_generating_for}: FFMPEG skipped for {full_vid_path}: File missing/empty AFTER release.")
        else:
            if video_session_filename_for_this_session:
                fp_fallback = os.path.join(current_app.config['UPLOAD_FOLDER'], video_session_filename_for_this_session)
                if os.path.exists(fp_fallback) and os.path.getsize(fp_fallback) > 0:
                    current_app.logger.warning(
                        f"User {user_id_generating_for}: VideoWriter N/A. FFMPEG on existing: {fp_fallback}")
                    post_process_video_with_ffmpeg(fp_fallback, current_app.logger)
                else:
                    current_app.logger.warning(
                        f"User {user_id_generating_for}: VideoWriter N/A AND file {fp_fallback} missing/empty. FFMPEG skipped.")
        with _stream_lock:
            _active_stream_viewers.discard(user_id_generating_for); _user_video_writers.pop(user_id_generating_for,
                                                                                            None)
        manage_camera_capture()
        current_app.logger.info(f"Stream generation cleanup ended for user {user_id_generating_for}.")


@main_bp.route('/video_feed')
@login_required
def video_feed():
    if model is None: flash("YOLO модель не загружена.", "danger"); return redirect(url_for('main.dashboard'))
    user_id = session['user_id'];
    add_log_entry("StreamFeedRequested", f"User ID: {user_id}", user_id)
    return Response(stream_with_context(generate_camera_frames_for_user(user_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@main_bp.route('/end_stream_session', methods=['POST'])
@login_required
def end_stream_session():
    user_id = session['user_id'];
    current_app.logger.info(f"end_stream_session POST from user {user_id}.")
    with _stream_lock:
        _active_stream_viewers.discard(user_id);
        writer = _user_video_writers.pop(user_id, None)
        if writer and writer.isOpened(): writer.release(); current_app.logger.warning(
            f"User {user_id}: VideoWriter force-closed in end_session.")
        # FFMPEG for forced close - attempt if file exists
        temp_fn = _user_session_filenames.get(user_id)  # use .get()
        if writer and temp_fn:  # if writer existed (even if now closed) and filename is known
            path_to_chk = os.path.join(current_app.config['UPLOAD_FOLDER'], temp_fn)
            if os.path.exists(path_to_chk) and os.path.getsize(path_to_chk) > 0:
                current_app.logger.info(
                    f"User {user_id}: Attempting ffmpeg in end_session (after force close check) for {path_to_chk}")
                post_process_video_with_ffmpeg(path_to_chk, current_app.logger)

        session_vid_file = _user_session_filenames.pop(user_id, None);
        session_dets = _user_session_detections.pop(user_id, [])
        user_sess_start = _user_stream_session_starts.pop(user_id, 0)
    manage_camera_capture()
    duration = 0;
    saved_vid_url = None;
    media_id_log = None
    if user_sess_start > 0: duration = time.time() - user_sess_start
    resp_msg = "Stream session ended."
    full_vid_path = None
    if session_vid_file: full_vid_path = os.path.join(current_app.config['UPLOAD_FOLDER'], session_vid_file)

    if session_vid_file and full_vid_path and os.path.exists(full_vid_path) and os.path.getsize(full_vid_path) > 0:
        try:
            media_ts = datetime.fromtimestamp(user_sess_start if user_sess_start > 0 else time.time())
            media = Media(filename=session_vid_file, file_path=full_vid_path, media_type='stream_capture',
                          user_id=user_id, upload_time=media_ts)
            db.session.add(media);
            db.session.flush();
            media_id_log = media.id
            for det_data in session_dets:
                db.session.add(Detection(media_id=media.id, object_class=det_data["object_class"],
                                         confidence=round(det_data["confidence"], 4),
                                         bbox_x=det_data["bbox_x"], bbox_y=det_data["bbox_y"],
                                         bbox_width=det_data["bbox_width"], bbox_height=det_data["bbox_height"],
                                         detected_at=det_data.get("detected_at", datetime.utcnow()),
                                         result_frame_filename=session_vid_file))
            db.session.commit()
            summary = {k: sum(1 for d in session_dets if d['object_class'] == k) for k in
                       set(d['object_class'] for d in session_dets)}
            obj_str = ", ".join([f"{k}({v})" for k, v in summary.items()])
            log_dets = f"User {user_id}, Media {media.id} (Vid: {session_vid_file}), Dur: {duration:.2f}s, Dets: {len(session_dets)}, Sum: {obj_str}"
            add_log_entry("StreamVideoSaved", log_dets, user_id)
            flash(f"Сессия стрима сохранена: {session_vid_file}. {('Обнаружено: ' + obj_str) if obj_str else ''}",
                  "success")
            resp_msg = 'Stream ended & video saved.';
            saved_vid_url = url_for('main.results_file', filename=session_vid_file)
        except Exception as e:
            db.session.rollback();
            current_app.logger.error(f"User {user_id}: DB Error save stream: {e}", exc_info=True)
            add_log_entry("StreamProcessingError", f"User {user_id}, DB Err: {str(e)}", user_id)
            flash("Ошибка БД при сохранении сессии.", "danger");
            resp_msg = 'Error saving session data.'
    elif duration > 0.5:
        add_log_entry("StreamEndNoVideo",
                      f"User {user_id}, Dur: {duration:.2f}s. No valid video (File: {session_vid_file}).", user_id)
        resp_msg = 'Stream ended. No video saved.';
        flash("Сессия завершена. Видео не сохранено.", "warning")
    else:
        resp_msg = 'Stream too short.'; current_app.logger.info(
            f"User {user_id}: Stream end, no data (dur: {duration:.2f}s).")
    return jsonify(
        {'status': 'success', 'message': resp_msg, 'duration': round(duration, 2), 'result_video_url': saved_vid_url})


# --- Загрузка и ОБРАБОТКА ВИДЕО ПОКАДРОВО (Uploaded Videos) ---
def process_video_frames_generator(video_path, media_id, user_id, original_media_filename_for_naming):
    if model is None: current_app.logger.error(
        "YOLO model N/A for uploaded video."); yield "event: error\ndata: {{\"message\": \"Model not loaded\"}}\n\n"; return
    if video_path is None or not os.path.exists(
        video_path): yield f"event: error\ndata: {json.dumps({'message': 'Invalid video file'})}\n\n"; return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): yield f"event: error\ndata: {json.dumps({'message': 'Cannot open video'})}\n\n"; return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # <<--- ИСПРАВЛЕНО ЗДЕСЬ: Убрана лишняя скобка
    if fps == 0 or fps > 120: fps = DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: current_app.logger.warning(
        f"Video {original_media_filename_for_naming} reported 0 total frames.")

    base_name, _ = os.path.splitext(original_media_filename_for_naming)
    processed_fn = f"processed_{base_name}.mp4"
    processed_path = os.path.join(current_app.config['UPLOAD_FOLDER'], processed_fn)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(processed_path, fourcc, float(fps), (frame_w, frame_h))

    if not out_writer.isOpened():
        cap.release();
        yield f"event: error\ndata: {json.dumps({'message': 'VideoWriter failed'})}\n\n";
        return

    summary = {};
    frame_idx = 0;
    start_time = time.time()
    yield f"data: Starting: {original_media_filename_for_naming} ({total_frames if total_frames > 0 else 'Unk.'} frames @ {fps:.2f} FPS). Out: {processed_fn}\n\n"
    current_app.logger.info(f"Uploaded video proc start: media {media_id}, file {original_media_filename_for_naming}")
    try:
        while True:
            success, frame = cap.read();
            if not success: break
            frame_idx += 1;
            results = model(frame, verbose=False);
            annotated = frame.copy()
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated = results[0].plot()
                for r in results:
                    for box in r.boxes: summary[model.names[int(box.cls[0])]] = summary.get(
                        model.names[int(box.cls[0])], 0) + 1
            out_writer.write(annotated)
            disp_fps = 5.0
            if frame_idx == 1 or (fps > 0 and frame_idx % (int(fps / disp_fps) if fps >= disp_fps else 1) == 0):
                _, buf = cv2.imencode('.jpg', annotated);
                jpg_txt = base64.b64encode(buf).decode('utf-8')
                yield f"event: frame\ndata: {json.dumps({'frame_count': frame_idx, 'image_data': jpg_txt})}\n\n"
            if frame_idx == total_frames or (fps > 0 and frame_idx % int(fps) == 0) or (
                    total_frames == 0 and frame_idx % int(DEFAULT_FPS) == 0):  # Progress if total_frames is 0
                perc = (frame_idx / total_frames) * 100 if total_frames > 0 else (
                    50.0 if frame_idx > 0 else 0.0)  # Placeholder for unknown total
                yield f"event: progress\ndata: {json.dumps({'frame': frame_idx, 'total_frames': total_frames if total_frames > 0 else 'Unknown', 'percent': round(perc, 1)})}\n\n"
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'message': f'Frame proc error: {str(e)}'})}\n\n"; current_app.logger.error(
            f"Error in uploaded video loop {media_id}: {e}", exc_info=True)
    finally:
        if cap.isOpened(): cap.release()
        if out_writer.isOpened(): out_writer.release()
        current_app.logger.debug(f"Uploaded video gen: cap/writer released for {processed_fn}.")
        post_process_video_with_ffmpeg(processed_path, current_app.logger)

    proc_time = time.time() - start_time
    current_app.logger.info(f"Uploaded video proc done for {media_id}. Time: {proc_time:.2f}s. Out: {processed_path}")
    if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
        if summary:
            for obj, cnt in summary.items(): db.session.add(
                Detection(media_id=media_id, object_class=obj, confidence=cnt, bbox_x=-2, bbox_y=-2, bbox_width=-2,
                          bbox_height=-2, detected_at=datetime.utcnow(), result_frame_filename=processed_fn))
            db.session.commit();
            add_log_entry("VideoProcessedSummary", f"Media {media_id}, Proc: {processed_fn}, Sum: {summary}", user_id)
        else:
            db.session.add(
                Detection(media_id=media_id, object_class="No objects detected", confidence=0, bbox_x=-2, bbox_y=-2,
                          bbox_width=-2, bbox_height=-2, detected_at=datetime.utcnow(),
                          result_frame_filename=processed_fn))
            db.session.commit();
            add_log_entry("VideoProcessedNoDetections", f"Media {media_id}, Proc: {processed_fn}", user_id)
        final_data = {"result_video_filename": processed_fn,
                      "result_url": url_for('main.results_file', filename=processed_fn, _external=True),
                      "summary": summary, "processing_time": round(proc_time, 2)}
        yield f"data: Finished. Video saved as {processed_fn}.\n\n";
        yield f"event: complete\ndata: {json.dumps(final_data)}\n\n"
    else:
        yield f"event: error\ndata: {json.dumps({'message': 'Output video error'})}\n\n"; current_app.logger.error(
            f"Processed video {processed_fn} missing/empty for media {media_id}.")


@main_bp.route('/process_uploaded_video/<int:media_id>')
@login_required
def process_uploaded_video_stream(media_id):
    media_item = Media.query.get_or_404(media_id)
    if not (media_item.user_id == session['user_id'] or session.get('is_admin')): abort(403)
    if media_item.media_type != 'video': return jsonify({"error": "Not an uploaded video"}), 400
    if not os.path.exists(media_item.file_path):
        return Response(stream_with_context(
            process_video_frames_generator(None, media_id, session['user_id'], media_item.filename)),
                        mimetype='text/event-stream')
    return Response(stream_with_context(
        process_video_frames_generator(media_item.file_path, media_id, session['user_id'], media_item.filename)),
                    mimetype='text/event-stream')


@main_bp.route('/upload_media', methods=['POST'])
@login_required
def upload_media():
    if model is None: return jsonify({'error': 'YOLO model not loaded.'}), 500
    if 'media' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['media']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    original_fn = secure_filename(file.filename);
    originals_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'originals')
    base, ext = os.path.splitext(original_fn)
    unique_orig_fn = f"{base}_{session.get('user_id', 0)}_{int(time.time())}{ext}"
    original_fp = os.path.join(originals_dir, unique_orig_fn)
    try:
        file.save(original_fp)
    except Exception as e:
        return jsonify({'error': f'Failed to save: {str(e)}'}), 500
    media_type = 'image' if original_fn.lower().endswith(('.png', '.jpg', '.jpeg')) else (
        'video' if original_fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else None)

    if not media_type:
        try:
            os.remove(original_fp)
        except OSError:
            pass  # pass теперь на правильном уровне отступа внутри except
        current_app.logger.warning(f"Unsupported file type uploaded: {original_fn}. Original deleted (if possible).")
        return jsonify({
                           'error': 'Unsupported file type. Only images (PNG, JPG) and videos (MP4, AVI, MOV, MKV) are allowed.'}), 400
    media = Media(filename=unique_orig_fn, file_path=original_fp, media_type=media_type, user_id=session['user_id'])
    db.session.add(media);
    db.session.commit();
    media_id = media.id
    add_log_entry("MediaUpload", f"Media {media_id}, Orig: {unique_orig_fn}", session['user_id'])
    if media_type == 'image':
        s_time = time.time();
        res_img_fn = f"result_img_{unique_orig_fn}";
        res_img_fp = os.path.join(current_app.config['UPLOAD_FOLDER'], res_img_fn);
        summary = {}
        try:
            img = cv2.imread(original_fp);
            if img is None: db.session.delete(media);db.session.commit(); return jsonify({'error': 'Bad image.'}), 500
            results = model(img, verbose=False);
            annotated = img.copy()
            if results and results[0].boxes is not None and len(results[0].boxes) > 0: annotated = results[0].plot()
            cv2.imwrite(res_img_fp, annotated)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0]);
                    obj_cls = model.names[cls_id];
                    conf = float(box.conf[0]);
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    db.session.add(
                        Detection(media_id=media_id, object_class=obj_cls, confidence=round(conf, 4), bbox_x=x1,
                                  bbox_y=y1, bbox_width=x2 - x1, bbox_height=y2 - y1, result_frame_filename=res_img_fn))
                    summary[obj_cls] = summary.get(obj_cls, 0) + 1
            db.session.commit();
            p_time = time.time() - s_time;
            add_log_entry("ImageProcessed", f"Media {media_id}, Res: {res_img_fn}, Time: {p_time:.2f}s",
                          session['user_id'])
            obj_s = ", ".join([f"{k}({v})" for k, v in summary.items()]) if summary else "None"
            return jsonify(
                {'message': f'Изображение {unique_orig_fn} обработано. Найдено: {obj_s}', 'media_id': media_id,
                 'media_type': 'image', 'original_filename': unique_orig_fn,
                 'result_filename': res_img_fn, 'detected_objects': obj_s, 'processing_time': round(p_time, 2),
                 'result_url': url_for('main.results_file', filename=res_img_fn)})
        except Exception as e:
            db.session.rollback();add_log_entry("ProcessingError", f"Image {media_id}, Error: {str(e)}",
                                                session['user_id']); return jsonify(
                {'error': f'Image proc error: {str(e)}'}), 500
    elif media_type == 'video':
        return jsonify(
            {'message': f'Видео {unique_orig_fn} загружено. Обработка...', 'media_id': media_id, 'media_type': 'video',
             'original_filename': unique_orig_fn,
             'processing_stream_url': url_for('main.process_uploaded_video_stream', media_id=media_id)})
    return jsonify({'error': 'Unexpected error'}), 500


# --- НОВАЯ ФУНКЦИЯ ДЛЯ ОТДАЧИ ВИДЕО ЧАСТЯМИ (Range Requests) ---
# ОПРЕДЕЛЕНА РАНЬШЕ, ЧЕМ /results/filename, КОТОРЫЙ ЕЕ ИСПОЛЬЗУЕТ
def ranged_video_response(file_path, content_type='video/mp4'):
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        current_app.logger.error(f"Ranged: File not found {file_path}"); abort(404)
    range_header = request.headers.get('Range', None);
    start = 0;
    end = file_size - 1
    status_code = 200;
    content_length = file_size
    if range_header:
        current_app.logger.debug(f"Ranged: Range header: {range_header} for {file_path}")
        match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if match:
            groups = match.groups();
            start = int(groups[0])
            if groups[1]: end = int(groups[1])
            if start >= file_size or end >= file_size or start > end:
                resp = Response("Range Not Satisfiable", 416);
                resp.headers['Content-Range'] = f'bytes */{file_size}';
                return resp
            content_length = (end - start) + 1;
            status_code = 206
            current_app.logger.debug(
                f"Ranged: Serving bytes {start}-{end} (len {content_length}) of {file_size} for {file_path}")
        else:
            current_app.logger.warning(f"Ranged: Could not parse Range: {range_header}. Serving full.")

    def generate_file_chunks(f_handle, offset, length):
        f_handle.seek(offset);
        remaining = length;
        chunk_s = 8192
        while remaining > 0:
            to_read = min(chunk_s, remaining);
            data = f_handle.read(to_read)
            if not data: break
            yield data;
            remaining -= len(data)

    try:
        file_handle = open(file_path, 'rb')
    except IOError:
        current_app.logger.error(f"Ranged: Could not open {file_path}."); abort(404)
    response = Response(stream_with_context(generate_file_chunks(file_handle, start, content_length)),
                        status=status_code, mimetype=content_type)
    response.headers['Content-Length'] = str(content_length);
    response.headers['Accept-Ranges'] = 'bytes'
    if status_code == 206 and range_header: response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
    try:  # ETag/Last-Modified
        mtime = os.path.getmtime(file_path)
        response.headers['Last-Modified'] = datetime.utcfromtimestamp(mtime).strftime("%a, %d %b %Y %H:%M:%S GMT")
        response.headers['ETag'] = f'"{mtime}-{file_size}"'
    except OSError:
        pass
    current_app.logger.debug(f"Ranged: Sending headers for {file_path}: {dict(response.headers)}")
    return response


# --- Просмотр результатов и история ---
@main_bp.route('/results/<path:filename>')
@login_required
def results_file(filename):
    file_path_on_server = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path_on_server): current_app.logger.warning(
        f"Result file N/A: {file_path_on_server}"); abort(404)
    if not session.get('is_admin'):  # Access Control
        user_id = session['user_id'];
        can_access = False
        q_det = db.session.query(Detection.id).join(Media).filter(Media.user_id == user_id,
                                                                  Detection.result_frame_filename == filename)
        if db.session.query(q_det.exists()).scalar(): can_access = True
        if not can_access:
            q_media = db.session.query(Media.id).filter(Media.user_id == user_id, Media.media_type == 'stream_capture',
                                                        Media.filename == filename)
            if db.session.query(q_media.exists()).scalar(): can_access = True
        if not can_access and filename.startswith('result_img_'):
            orig_cand = filename[len('result_img_'):]
            q_orig = db.session.query(Media.id).filter(Media.user_id == user_id, Media.media_type == 'image',
                                                       Media.filename == orig_cand)
            if db.session.query(q_orig.exists()).scalar(): can_access = True
        if not can_access: current_app.logger.warning(f"User {user_id} access denied for '{filename}'."); abort(403)

    lower_fn = filename.lower();
    content_type = 'application/octet-stream'
    if lower_fn.endswith(".mp4"):
        content_type = 'video/mp4'
    elif lower_fn.endswith(".avi"):
        content_type = 'video/x-msvideo'
    elif lower_fn.endswith((".jpg", ".jpeg")):
        content_type = 'image/jpeg'
    elif lower_fn.endswith(".png"):
        content_type = 'image/png'

    if content_type.startswith('video/'):
        current_app.logger.info(f"Serving video '{filename}' via ranged_video_response.")
        return ranged_video_response(file_path_on_server, content_type)
    else:
        current_app.logger.info(f"Serving non-video '{filename}' via send_from_directory.")
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@main_bp.route('/history')
@login_required
def history_page():
    page = request.args.get('page', 1, type=int);
    user_media_q = Media.query.filter_by(user_id=session['user_id'])
    media_type_f = request.args.get('media_type')
    if media_type_f: user_media_q = user_media_q.filter(Media.media_type == media_type_f)
    pagination = user_media_q.order_by(Media.upload_time.desc()).paginate(page=page, per_page=5, error_out=False)
    items = []
    for m_item in pagination.items:
        summary = {};
        preview_img = None;
        assoc_vid = None
        if m_item.media_type == 'image':
            for d in m_item.detections: summary[d.object_class] = summary.get(d.object_class, 0) + 1
            fn = next((d.result_frame_filename for d in m_item.detections if
                       d.result_frame_filename and d.result_frame_filename.startswith("result_img_")), None)
            if not fn and m_item.filename: fn = f"result_img_{m_item.filename}"
            if fn and os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], fn)): preview_img = fn
        elif m_item.media_type == 'video':
            for d in m_item.detections:
                if d.bbox_x == -2: summary[d.object_class] = summary.get(d.object_class, 0) + int(d.confidence)
            fn = next((d.result_frame_filename for d in m_item.detections if
                       d.bbox_x == -2 and d.result_frame_filename and d.result_frame_filename.startswith("processed_")),
                      None)
            if not fn and m_item.filename: base, _ = os.path.splitext(m_item.filename); fn = f"processed_{base}.mp4"
            if fn and os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], fn)): assoc_vid = fn
        elif m_item.media_type == 'stream_capture':
            assoc_vid = m_item.filename
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], assoc_vid)): assoc_vid = None
            for d in m_item.detections: summary[d.object_class] = summary.get(d.object_class, 0) + 1
        items.append({'media': m_item, 'detections_summary_str': (
            ", ".join([f"{k}({v})" for k, v in summary.items()]) if summary else "Нет"),
                      'preview_image_file': preview_img, 'associated_video_file': assoc_vid})
    return render_template('history_page.html', media_items=items, pagination=pagination,
                           current_media_type_filter=media_type_f)


@main_bp.route('/download_report/media/<int:media_id>')
@login_required
def download_media_report(media_id):
    media_item = Media.query.get_or_404(media_id)
    if not (media_item.user_id == session['user_id'] or session.get('is_admin')): abort(403)
    detections = media_item.detections
    if not detections: flash("Нет детекций.", "info"); return redirect(request.referrer or url_for('main.history_page'))
    si = io.StringIO();
    cw = csv.writer(si);
    header = ['ID Детекции', 'Класс Объекта']
    is_sum = media_item.media_type == 'video' and any(d.bbox_x == -2 for d in detections)
    if is_sum:
        header.append('Кол-во (видео)')
    else:
        header.extend(['Уверенность', 'X', 'Y', 'Ширина', 'Высота'])
    header.extend(['Время Детекции', 'Файл результата']);
    cw.writerow(header)
    for det in detections:
        row = [det.id, det.object_class]
        if is_sum:
            row.append(int(det.confidence))
        else:
            row.extend([f"{det.confidence:.4f}", det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height])
        row.extend([det.detected_at.strftime('%Y-%m-%d %H:%M:%S'), det.result_frame_filename or (
            media_item.filename if media_item.media_type == 'stream_capture' else "N/A")])
        cw.writerow(row)
    output = si.getvalue();
    resp = Response(output, mimetype="text/csv")
    safe_fn = "".join(c if c.isalnum() else "_" for c in media_item.filename);
    resp.headers["Content-Disposition"] = f"attachment; filename=report_media_{safe_fn}_{media_id}.csv"
    return resp


@main_bp.route('/admin')
@admin_required
def admin_dashboard():
    users = User.query.order_by(User.username).all()
    return render_template('admin/dashboard.html', users=users)


@main_bp.route('/admin/user_history/<int:user_id>')
@admin_required
def admin_user_history(user_id):
    target_user = User.query.get_or_404(user_id)
    page = request.args.get('page', 1, type=int);
    user_media_q = Media.query.filter_by(user_id=target_user.id)
    media_type_f = request.args.get('media_type')
    if media_type_f: user_media_q = user_media_q.filter(Media.media_type == media_type_f)
    pagination = user_media_q.order_by(Media.upload_time.desc()).paginate(page=page, per_page=5, error_out=False)
    items = []  # Копируем логику из history_page
    for m_item in pagination.items:
        summary = {};
        preview_img = None;
        assoc_vid = None
        # ... (та же логика определения preview_image_file и associated_video_file, что и в history_page) ...
        if m_item.media_type == 'image':
            for d in m_item.detections: summary[d.object_class] = summary.get(d.object_class, 0) + 1
            fn = next((d.result_frame_filename for d in m_item.detections if
                       d.result_frame_filename and d.result_frame_filename.startswith("result_img_")), None)
            if not fn and m_item.filename: fn = f"result_img_{m_item.filename}"
            if fn and os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], fn)): preview_img = fn
        elif m_item.media_type == 'video':
            for d in m_item.detections:
                if d.bbox_x == -2: summary[d.object_class] = summary.get(d.object_class, 0) + int(d.confidence)
            fn = next((d.result_frame_filename for d in m_item.detections if
                       d.bbox_x == -2 and d.result_frame_filename and d.result_frame_filename.startswith("processed_")),
                      None)
            if not fn and m_item.filename: base, _ = os.path.splitext(m_item.filename); fn = f"processed_{base}.mp4"
            if fn and os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], fn)): assoc_vid = fn
        elif m_item.media_type == 'stream_capture':
            assoc_vid = m_item.filename
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], assoc_vid)): assoc_vid = None
            for d in m_item.detections: summary[d.object_class] = summary.get(d.object_class, 0) + 1
        items.append({'media': m_item, 'detections_summary_str': (
            ", ".join([f"{k}({v})" for k, v in summary.items()]) if summary else "Объекты не найдены"),
                      'preview_image_file': preview_img, 'associated_video_file': assoc_vid})
    return render_template('admin/user_history_page.html', target_user=target_user, media_items=items,
                           pagination=pagination, current_media_type_filter=media_type_f)


@main_bp.route('/admin/download_user_report/<int:user_id>')
@admin_required
def admin_download_user_report(user_id):
    target_user = User.query.get_or_404(user_id)
    media_items = Media.query.filter_by(user_id=user_id).order_by(Media.upload_time.asc()).all()
    if not media_items: flash(f"Нет данных для отчета по {target_user.username}.", "info"); return redirect(
        url_for('main.admin_user_history', user_id=user_id))
    si = io.StringIO();
    cw = csv.writer(si)
    cw.writerow(
        ['ID Медиа', 'Имя Файла', 'Тип', 'Время Загрузки', 'ID Детекции', 'Класс', 'Увер./Кол-во', 'X', 'Y', 'Шир.',
         'Выс.', 'Время Детекции', 'Файл результата'])
    for m in media_items:
        if not m.detections:
            res_f = m.filename if m.media_type == 'stream_capture' else (
                f"processed_{os.path.splitext(m.filename)[0]}.mp4" if m.media_type == 'video' else 'N/A')
            cw.writerow(
                [m.id, m.filename, m.media_type_rus, m.upload_time.strftime('%Y-%m-%d %H:%M:%S'), 'N/A', 'Нет детекций',
                 '', '', '', '', '', '', res_f])
        else:
            for det in m.detections:
                is_sum = (m.media_type == 'video' and det.bbox_x == -2)
                conf = int(det.confidence) if is_sum else f"{det.confidence:.4f}"
                x, y, w, h = ("N/A",) * 4 if is_sum else (det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height)
                res_f = det.result_frame_filename or (m.filename if m.media_type == 'stream_capture' else 'N/A')
                cw.writerow([m.id, m.filename, m.media_type_rus, m.upload_time.strftime('%Y-%m-%d %H:%M:%S'), det.id,
                             det.object_class, conf, x, y, w, h, det.detected_at.strftime('%Y-%m-%d %H:%M:%S'), res_f])
    output = si.getvalue();
    response = Response(output, mimetype="text/csv")
    safe_un = "".join(c if c.isalnum() else "_" for c in target_user.username);
    rep_fn = f"full_report_user_{safe_un}_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-Disposition"] = f"attachment; filename=\"{rep_fn}\"";
    return response