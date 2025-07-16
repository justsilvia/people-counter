from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from database import init_db, save_request, get_history, delete_request, export_csv, get_summary
import sqlite3
import csv
import io

app = Flask(__name__)
model = YOLO('yolov8l.pt')  # Улучшенная модель

# Конфигурация
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Инициализация БД
init_db()

def detect_table_area(img):
    """Определяет область стола на изображении с улучшенной логикой"""
    # Уменьшаем изображение для ускорения обработки
    scale_percent = 40
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized = cv2.resize(img, (width, height))
    
    # Преобразуем в HSV для лучшего выделения цветов
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Определяем диапазон цветов для дерева (стол)
    lower_brown1 = np.array([0, 30, 30])
    upper_brown1 = np.array([20, 200, 200])
    lower_brown2 = np.array([160, 30, 30])
    upper_brown2 = np.array([180, 200, 200])
    
    # Создаем маску для коричневых цветов
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Применяем морфологические операции для улучшения маски
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Если не нашли стол, используем центральную область
        return [
            int(img.shape[1] * 0.2), 
            int(img.shape[0] * 0.4), 
            int(img.shape[1] * 0.8), 
            int(img.shape[0] * 0.9)
        ]
    
    # Выбираем самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Масштабируем координаты обратно к оригинальному размеру
    scale_x = img.shape[1] / width
    scale_y = img.shape[0] / height
    
    # Расширяем область для покрытия всего стола
    return [
        max(0, int((x - 30) * scale_x)),
        max(0, int((y - 30) * scale_y)),
        min(img.shape[1], int((x + w + 30) * scale_x)),
        min(img.shape[0], int((y + h + 30) * scale_y))
    ]

def is_point_in_rect(point, rect):
    """Проверяет, находится ли точка внутри прямоугольника"""
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Генерация уникальных имен файлов
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    upload_filename = f'upload_{timestamp}.jpg'
    result_filename = f'result_{timestamp}.jpg'
    
    # Сохранение загруженного изображения
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    file.save(upload_path)
    
    # Обработка изображения
    img = cv2.imread(upload_path)
    original_img = img.copy()
    
    # Определяем область стола
    table_area = detect_table_area(img)
    
    # Детекция людей с фильтрацией по классу (0 - человек)
    results = model(img, classes=[0], conf=0.5)
    
    # Подсчёт людей за столом
    person_count = 0
    table_people = 0
    
    # Обрабатываем результаты
    for box in results[0].boxes:
        # Координаты центра нижней части bbox (примерное положение ног)
        x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
        y_bottom = int(box.xyxy[0][3])
        
        # Проверяем, находится ли точка в области стола
        if is_point_in_rect((x_center, y_bottom), table_area):
            # Человек за столом - зеленый
            color = (0, 255, 0)
            table_people += 1
        else:
            # Человек не за столом - красный
            color = (0, 0, 255)
        
        # Рисуем bounding box
        cv2.rectangle(img, 
                      (int(box.xyxy[0][0]), int(box.xyxy[0][1])), 
                      (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                      color, 2)
        
        person_count += 1
    
    # Рисуем область стола
    cv2.rectangle(img, 
                 (table_area[0], table_area[1]), 
                 (table_area[2], table_area[3]), 
                 (0, 255, 255), 2)
    
    # Сохраняем результат
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)
    
    # Сохранение в историю
    save_request(
        timestamp=datetime.now(),
        upload_path=upload_path,
        result_path=result_path,
        person_count=person_count,
        table_people=table_people
    )
    
    return jsonify({
        'total_count': person_count,
        'table_count': table_people,
        'image_url': result_path
    })

@app.route('/history')
def history():
    # Параметры запроса
    page = request.args.get('page', 1, type=int)
    per_page = 10
    sort_by = request.args.get('sort_by', 'timestamp')
    sort_order = request.args.get('sort_order', 'desc')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Получаем историю с пагинацией
    requests, total_requests = get_history(
        page=page, 
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
        start_date=start_date,
        end_date=end_date
    )
    
    # Рассчитываем общее количество страниц
    total_pages = (total_requests + per_page - 1) // per_page
    
    # Получаем сводную статистику
    summary = get_summary()
    
    return render_template(
        'history.html',
        requests=requests,
        current_page=page,
        total_pages=total_pages,
        summary=summary
    )

@app.route('/delete_request/<int:request_id>', methods=['DELETE'])
def delete_single_request(request_id):
    try:
        delete_request(request_id)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    # Очистка истории и файлов
    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        for file in files:
            os.remove(os.path.join(root, file))
    
    for root, dirs, files in os.walk(app.config['RESULT_FOLDER']):
        for file in files:
            os.remove(os.path.join(root, file))
    
    # Очистка БД
    init_db(clear=True)
    
    return redirect(url_for('history'))

@app.route('/export_csv')
def export_history_csv():
    # Создаем CSV в памяти
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Заголовки CSV
    writer.writerow(['ID', 'Timestamp', 'Total People', 'Table People', 'Image Path', 'Result Path'])
    
    # Получаем все записи
    conn = sqlite3.connect('requests.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM requests ORDER BY timestamp DESC')
    requests = cursor.fetchall()
    conn.close()
    
    # Записываем данные
    for req in requests:
        writer.writerow([
            req[0],
            req[1],
            req[4],
            req[5],
            req[2],
            req[3]
        ])
    
    # Возвращаем CSV как файл
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='history_export.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)