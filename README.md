# 🪑 Система подсчета гостей за столом с использованием YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.0-red)](https://ultralytics.com/yolov8)

Проект представляет собой веб-приложение на Flask, которое использует модель YOLOv8 для обнаружения людей на изображении и определения, сколько из них находится за столом.

## ✨ Ключевые особенности

- 🕵️‍♂️ **Обнаружение людей** с помощью модели YOLOv8
- 🪑 **Определение области стола** с использованием алгоритмов компьютерного зрения
- 📊 **Статистика по положению людей** (за столом/вне стола)
- 📁 **История запросов** с фильтрацией и сортировкой
- 📈 **Сводная статистика** по всем обработкам
- 📤 **Экспорт данных** в формате CSV
- 🗑️ **Управление историей** (удаление отдельных записей или полная очистка)

## 🛠️ Технологический стек

- **Backend**: Python, Flask
- **Computer Vision**: Ultralytics YOLOv8, OpenCV
- **База данных**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript

## 🚀 Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Запустите приложение:
```bash
python app.py
```

5. Откройте в браузере: [http://localhost:5000](http://localhost:5000)

## 📂 Структура проекта

```
├── app.py                # Основное Flask-приложение
├── database.py           # Работа с базой данных
├── requirements.txt      # Зависимости Python
├── static/               # Статические файлы (CSS, JS)
│   ├── uploads/          # Загруженные изображения
│   └── results/          # Результаты обработки
└── templates/            # HTML шаблоны
    ├── index.html        # Главная страница
    └── history.html      # Страница истории
```

## 💻 Как использовать

1. На главной странице загрузите изображение с людьми и столом
2. Нажмите "Запустить анализ"
3. Система покажет:
   - Общее количество людей
   - Количество людей за столом
   - Оригинальное изображение
   - Результат обработки с выделенными областями
4. Для просмотра истории перейдите на соответствующую вкладку

## 📜 Лицензия

Этот проект распространяется под лицензией MIT. Подробнее см. в файле [LICENSE](https://github.com/justsilvia/people-counter/blob/main/LICENSE).
