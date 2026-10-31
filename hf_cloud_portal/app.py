# -*- coding: utf-8 -*-
"""
Hugging Face Spaces - Standalone Student Face Registration Cloud Portal
Provides a 24/7 web portal for student self-registration and face image collection.
Runs on port 7860 (Hugging Face default) using Flask and face_recognition.
"""

import base64
import json
import os
import re
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    import face_recognition
    FR_AVAILABLE = True
    FR_ERROR = None
except Exception as _exc:
    face_recognition = None
    FR_AVAILABLE = False
    FR_ERROR = str(_exc)

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('DATA_DIR', '/tmp/data')
CLOUD_FACES_DIR = os.path.join(DATA_DIR, 'cloud_faces')
CLOUD_STUDENTS_JSON = os.path.join(DATA_DIR, 'cloud_students.json')
DEFAULT_TOKEN = os.environ.get('CLOUD_SYNC_TOKEN', 'mahyar123')
PORT = int(os.environ.get('PORT', 7860))

os.makedirs(CLOUD_FACES_DIR, exist_ok=True)

# Digits normalization and input validation regex
_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
_CODE_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def clean_text(value, label, max_len=80, required=True):
    s = ' '.join(str(value or '').split())
    if required and not s:
        raise ValueError(f'{label} الزامی است')
    if len(s) > max_len:
        raise ValueError(f'{label} نباید بیش از {max_len} نویسه باشد')
    if any(ord(c) < 32 for c in s):
        raise ValueError(f'{label} نویسه نامعتبر دارد')
    if s[:1] in ('=', '+', '-', '@'):
        raise ValueError(f'{label} نباید با = + - @ شروع شود')
    return s


def fit(img, max_side=1280):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    k = max_side / max(h, w)
    return cv2.resize(img, (0, 0), fx=k, fy=k)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_jpeg(path, img, quality=95):
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError('JPEG encoding failed')
    with open(path, 'wb') as f:
        f.write(buf.tobytes())


# ==============================================================================
# Persistent Storage & Biometric Cache
# ==============================================================================
class CloudStorage:
    def __init__(self, json_path, faces_dir):
        self.json_path = json_path
        self.faces_dir = faces_dir
        self.lock = threading.Lock()
        self.students = {}
        self._encodings = []  # list of 128-d numpy arrays
        self._ids = []        # corresponding student_ids
        self._load()

    def _load(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.students = json.load(f)
            except Exception as e:
                print(f"[storage] Warning loading {self.json_path}: {e}")
                self.students = {}
        else:
            self._save()

        # Preload face encodings for duplicate checking if face_recognition is available
        if FR_AVAILABLE and os.path.isdir(self.faces_dir):
            for sid in os.listdir(self.faces_dir):
                sdir = os.path.join(self.faces_dir, sid)
                if os.path.isdir(sdir):
                    for fname in sorted(os.listdir(sdir)):
                        if fname.lower().endswith(IMAGE_EXT):
                            fpath = os.path.join(sdir, fname)
                            try:
                                img = cv2.imread(fpath)
                                if img is not None:
                                    rgb = to_rgb(fit(img, 1280))
                                    locs = face_recognition.face_locations(rgb)
                                    if locs:
                                        encs = face_recognition.face_encodings(rgb, locs)
                                        if encs:
                                            self._encodings.append(encs[0])
                                            self._ids.append(sid)
                            except Exception:
                                pass
            print(f"[storage] Loaded {len(self.students)} students, {len(self._encodings)} face encodings.")

    def _save(self):
        tmp = self.json_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.students, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.json_path)

    def has_student(self, student_id):
        with self.lock:
            if student_id in self.students:
                return True
            pdir = os.path.join(self.faces_dir, student_id)
            if os.path.isdir(pdir) and any(f.lower().endswith(IMAGE_EXT) for f in os.listdir(pdir)):
                return True
            return False

    def add_student(self, student_id, name, grade, valid_images, encodings):
        with self.lock:
            person_dir = os.path.join(self.faces_dir, student_id)
            os.makedirs(person_dir, exist_ok=True)
            for i, img in enumerate(valid_images, 1):
                write_jpeg(os.path.join(person_dir, f'{i:02d}.jpg'), img)

            self.students[student_id] = {
                'student_id': student_id,
                'name': name,
                'grade': grade,
                'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'image_count': len(valid_images)
            }
            self._save()

            for enc in encodings:
                self._encodings.append(enc)
                self._ids.append(student_id)

    def check_duplicate_face(self, encodings, tolerance=0.50):
        if not FR_AVAILABLE or not self._encodings:
            return None
        existing_encs = np.array(self._encodings)
        for enc in encodings:
            dists = face_recognition.face_distance(existing_encs, enc)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist <= tolerance:
                matched_id = self._ids[best_idx]
                matched_info = self.students.get(matched_id, {})
                matched_name = matched_info.get('name', matched_id)
                return {
                    'student_id': matched_id,
                    'name': matched_name,
                    'distance': best_dist
                }
        return None

    def get_all(self):
        with self.lock:
            return dict(self.students)


storage = CloudStorage(CLOUD_STUDENTS_JSON, CLOUD_FACES_DIR)

# ==============================================================================
# Flask Application
# ==============================================================================
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.json.ensure_ascii = False


@app.route('/')
@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Persian RTL Student Self-Registration portal.
    GET  -> Renders mobile-first responsive Persian registration form.
    POST -> Validates student information and face photos with face_recognition.
    """
    if request.method == 'GET':
        return render_template('register.html')

    is_ajax = (
        request.headers.get('Accept') == 'application/json' or
        request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
        (request.content_type and 'application/json' in request.content_type)
    )

    full_name = request.form.get('full_name', '').strip()
    student_id = request.form.get('student_id', '').strip()

    def respond_error(message, status_code=400):
        if is_ajax:
            return jsonify({'success': False, 'error': message}), status_code
        return render_template('register.html', error=message, full_name=full_name, student_id=student_id), status_code

    def respond_success(message):
        if is_ajax:
            return jsonify({'success': True, 'message': message})
        return render_template('register.html', success=message)

    # 1. Validate full_name
    if not full_name:
        return respond_error('لطفاً نام و نام خانوادگی را وارد کنید.')
    try:
        full_name = clean_text(full_name, 'نام و نام خانوادگی', 80, True)
    except ValueError as exc:
        return respond_error(str(exc))

    # 2. Validate student_id & duplicate check
    if not student_id:
        return respond_error('لطفاً شماره دانشجویی را وارد کنید.')
    student_id = student_id.translate(_DIGITS).strip()
    if not _CODE_RE.match(student_id):
        return respond_error('شماره دانشجویی فقط می‌تواند شامل اعداد و حروف انگلیسی باشد (حداکثر ۳۲ نویسه).')

    if storage.has_student(student_id):
        return respond_error(f'شماره دانشجویی {student_id} قبلاً در سامانه ثبت شده است!')

    # 3. Collect uploaded image files
    uploaded_files = []
    for key in ('image_1', 'image_2', 'image_3'):
        if key in request.files and request.files[key].filename:
            uploaded_files.append(request.files[key])

    if not uploaded_files:
        if 'images' in request.files:
            uploaded_files = [f for f in request.files.getlist('images') if f.filename]
        elif 'image' in request.files:
            uploaded_files = [f for f in request.files.getlist('image') if f.filename]

    if len(uploaded_files) < 3:
        if len(uploaded_files) == 0:
            return respond_error('لطفاً هر ۳ تصویر چهره (روبرو، متمایل به راست، متمایل به چپ) را بارگذاری کنید.')
        return respond_error(
            f'برای دستیابی به حداکثر دقت شناسایی، بارگذاری هر ۳ تصویر الزامی است (تعداد تصاویر انتخابی فعلی: {len(uploaded_files)}).'
        )

    uploaded_files = uploaded_files[:3]

    if not FR_AVAILABLE:
        return respond_error(f'موتور پردازش چهره در سرور ابری در دسترس نیست: {FR_ERROR}', 503)

    # 4. Decode and validate each face image
    valid_images = []
    extracted_encodings = []

    for idx, file_item in enumerate(uploaded_files, 1):
        idx_fa = str(idx).translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹'))
        try:
            file_bytes = file_item.read()
        except Exception as exc:
            return respond_error(f'خطا در خواندن فایل تصویر شماره {idx_fa}: {exc}')

        if not file_bytes:
            return respond_error(f'فایل تصویر شماره {idx_fa} خالی است.')

        if len(file_bytes) > 12 * 1024 * 1024:
            return respond_error(f'حجم فایل تصویر شماره {idx_fa} بیش از حد مجاز است (حداکثر ۱۲ مگابایت).')

        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return respond_error(f'فایل ارسالی شماره {idx_fa} یک تصویر معتبر نیست. لطفاً عکس با فرمت JPG یا PNG ارسال کنید.')

        h, w = img.shape[:2]
        if min(h, w) < 64 or max(h, w) > 6000:
            return respond_error(f'ابعاد تصویر شماره {idx_fa} نامعتبر است.')

        img_fitted = fit(img, 1280)
        rgb_img = to_rgb(img_fitted)

        face_locations = face_recognition.face_locations(rgb_img)
        num_faces = len(face_locations)
        if num_faces == 0:
            return respond_error(f'در تصویر شماره {idx_fa} چهره‌ای شناسایی نشد! لطفاً یک عکس واضح و تمام‌رخ انتخاب کنید.')
        elif num_faces > 1:
            return respond_error(f'در تصویر شماره {idx_fa} بیش از یک چهره دیده می‌شود! لطفاً یک عکس تکی ارسال کنید.')

        encs = face_recognition.face_encodings(rgb_img, face_locations)
        if not encs:
            return respond_error(f'امکان استخراج ویژگی‌های چهره در تصویر شماره {idx_fa} وجود نداشت.')

        valid_images.append(img_fitted)
        extracted_encodings.append(encs[0])

    # 5. Check internal consistency between images
    if len(extracted_encodings) >= 2:
        d12 = float(face_recognition.face_distance([extracted_encodings[0]], extracted_encodings[1])[0])
        if d12 > 0.55:
            return respond_error('تصاویر ارسالی متعلق به یک فرد یکسان تشخیص داده نشدند! لطفاً هر ۳ تصویر را از چهره خودتان انتخاب کنید.')
    if len(extracted_encodings) >= 3:
        d13 = float(face_recognition.face_distance([extracted_encodings[0]], extracted_encodings[2])[0])
        d23 = float(face_recognition.face_distance([extracted_encodings[1]], extracted_encodings[2])[0])
        if max(d13, d23) > 0.55:
            return respond_error('تصاویر ارسالی متعلق به یک فرد یکسان تشخیص داده نشدند! لطفاً هر ۳ تصویر را از چهره خودتان انتخاب کنید.')

    # 6. Biometric duplicate check against already registered faces in cloud
    dup = storage.check_duplicate_face(extracted_encodings, tolerance=0.50)
    if dup and dup['student_id'] != student_id:
        return respond_error(
            f'تشخیص چهره تکراری: این چهره قبلاً در سامانه برای «{dup["name"]}» با شماره دانشجویی {dup["student_id"]} '
            f'ثبت شده است! امکان ثبت مجدد این چهره با کد دانشجویی جدید وجود ندارد.'
        )

    # 7. Store student record and high-quality images
    try:
        storage.add_student(student_id, full_name, 'دانشجو', valid_images, extracted_encodings)
    except Exception as exc:
        return respond_error(f'خطا در ذخیره‌سازی اطلاعات: {exc}', 500)

    return respond_success('ثبت‌نام شما با موفقیت انجام شد! ۳ تصویر چهره با زوایای مختلف در سرور ابری ذخیره گردید.')


@app.route('/api/cloud/export', methods=['GET'])
def cloud_export():
    """
    JSON API endpoint for local system sync.
    Accepts an optional secret token query param (default: mahyar123).
    Returns all registered students with high-quality JPEG images encoded in base64.
    """
    req_token = request.args.get('token', DEFAULT_TOKEN).strip()
    if req_token != DEFAULT_TOKEN:
        return jsonify({
            'ok': False,
            'success': False,
            'error': 'توکن امنیتی نامعتبر است (Invalid security token)'
        }), 401

    all_students = storage.get_all()
    export_list = []

    for sid, info in all_students.items():
        person_dir = os.path.join(CLOUD_FACES_DIR, sid)
        b64_images = []
        primary_b64 = None

        if os.path.isdir(person_dir):
            for fname in sorted(os.listdir(person_dir)):
                if fname.lower().endswith(IMAGE_EXT):
                    fpath = os.path.join(person_dir, fname)
                    try:
                        with open(fpath, 'rb') as img_f:
                            raw_b64 = base64.b64encode(img_f.read()).decode('utf-8')
                            b64_images.append(raw_b64)
                            if primary_b64 is None or fname.startswith('01'):
                                primary_b64 = raw_b64
                    except OSError:
                        pass

        if not primary_b64 and b64_images:
            primary_b64 = b64_images[0]

        export_list.append({
            'student_id': sid,
            'code': sid,
            'name': info.get('name') or sid,
            'grade': info.get('grade') or 'دانشجو',
            'registered_at': info.get('registered_at', ''),
            'image': primary_b64 or '',
            'images': b64_images,
        })

    return jsonify({
        'ok': True,
        'success': True,
        'total': len(export_list),
        'count': len(export_list),
        'students': export_list
    })


@app.route('/healthz', methods=['GET'])
@app.route('/api/status', methods=['GET'])
def health():
    return jsonify({
        'ok': True,
        'status': 'online',
        'service': 'hf_cloud_portal',
        'registered_students': len(storage.students),
        'face_recognition_ready': FR_AVAILABLE
    })


if __name__ == '__main__':
    print('=' * 65)
    print(f'Hugging Face Cloud Portal running on port {PORT}')
    print(f'Registration Page: http://0.0.0.0:{PORT}/')
    print(f'Cloud Export API : http://0.0.0.0:{PORT}/api/cloud/export?token={DEFAULT_TOKEN}')
    print('=' * 65)
    app.run(host='0.0.0.0', port=PORT, debug=False)
