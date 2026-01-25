import sqlite3
import json

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Таблица с информацией об изображениях
cursor.execute('''
CREATE TABLE IF NOT EXISTS monkey_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    species TEXT NOT NULL,
    faiss_index INTEGER,      -- Индекс в FAISS
    embedding BLOB,           -- Вектор (опционально)
    metadata JSON,            -- Доп. информация
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Таблица для истории поисков
cursor.execute('''
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_image TEXT,
    results JSON,            -- Список ID найденных изображений
    search_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Индексы для быстрого поиска
cursor.execute('CREATE INDEX idx_species ON monkey_images(species)')
cursor.execute('CREATE INDEX idx_faiss ON monkey_images(faiss_index)')

conn.commit()
conn.close()
print("Database initialized successfully.")