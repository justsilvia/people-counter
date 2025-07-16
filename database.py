import sqlite3
import os
from datetime import datetime

DB_NAME = 'requests.db'

def init_db(clear=False):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    if clear:
        c.execute('DROP TABLE IF EXISTS requests')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            upload_path TEXT NOT NULL,
            result_path TEXT NOT NULL,
            person_count INTEGER NOT NULL,
            table_people INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_request(timestamp, upload_path, result_path, person_count, table_people):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO requests (timestamp, upload_path, result_path, person_count, table_people)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp.isoformat(), upload_path, result_path, person_count, table_people))
    conn.commit()
    conn.close()

def get_history(page=1, per_page=10, sort_by='timestamp', sort_order='desc', start_date=None, end_date=None):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    
    # Основной запрос
    query = 'SELECT * FROM requests'
    params = []
    
    # Фильтрация по дате
    if start_date and end_date:
        query += ' WHERE date(timestamp) BETWEEN ? AND ?'
        params.extend([start_date, end_date])
    
    # Сортировка
    valid_columns = ['id', 'timestamp', 'person_count', 'table_people']
    sort_by = sort_by if sort_by in valid_columns else 'timestamp'
    sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
    query += f' ORDER BY {sort_by} {sort_order}'
    
    # Пагинация
    offset = (page - 1) * per_page
    query += ' LIMIT ? OFFSET ?'
    params.extend([per_page, offset])
    
    # Выполняем запрос
    c = conn.cursor()
    c.execute(query, params)
    requests = c.fetchall()
    
    # Получаем общее количество записей
    count_query = 'SELECT COUNT(*) FROM requests'
    if start_date and end_date:
        count_query += ' WHERE date(timestamp) BETWEEN ? AND ?'
        c.execute(count_query, [start_date, end_date])
    else:
        c.execute(count_query)
    
    total_requests = c.fetchone()[0]
    conn.close()
    
    return requests, total_requests

def delete_request(request_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM requests WHERE id = ?', (request_id,))
    conn.commit()
    conn.close()

def get_summary():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Общее количество запросов
    c.execute('SELECT COUNT(*) FROM requests')
    total_requests = c.fetchone()[0]
    
    # Общее количество людей
    c.execute('SELECT SUM(person_count) FROM requests')
    total_people = c.fetchone()[0] or 0
    
    # Общее количество людей за столом
    c.execute('SELECT SUM(table_people) FROM requests')
    total_table_people = c.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total_requests': total_requests,
        'total_people': total_people,
        'total_table_people': total_table_people
    }

def export_csv():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM requests ORDER BY timestamp DESC')
    requests = c.fetchall()
    conn.close()
    return requests