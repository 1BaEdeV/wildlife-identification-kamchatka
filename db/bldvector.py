import tensorflow as tf
import numpy as np
import sqlite3
import os
import faiss
import json

# Загрузка модели для эмбеддингов
model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

# Подключение к БД
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Очищаем старые данные
cursor.execute('DELETE FROM monkey_images')
conn.commit()

# Инициализация FAISS индекса
embedding_dim = 1280  # Для EfficientNetB0
index = faiss.IndexFlatL2(embedding_dim)  # L2 расстояние

# Проход по всем изображениям
dataset_path = r"E:\ULTIMATE_PROJECT\wildlife-identification-kamchatka\full_dataset"
all_images = []
all_embeddings = []

print("Начало обработки изображений...")

for species in os.listdir(dataset_path):
    species_path = os.path.join(dataset_path, species)
    if not os.path.isdir(species_path):
        continue
    
    # Проверяем наличие папки all_photos
    all_photos_path = os.path.join(species_path, "all_photos")
    if not os.path.exists(all_photos_path):
        print(f"⚠️ Пропускаем {species}: нет папки all_photos")
        continue
    
    print(f"Обработка {species}...")
    
    for img_file in os.listdir(all_photos_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # ИСПРАВЛЕННЫЙ ПУТЬ
        img_path = os.path.join(all_photos_path, img_file)
        
        try:
            # Загрузка и предобработка
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array[np.newaxis, ...])
            
            # Извлечение эмбеддинга
            embedding = model.predict(img_array, verbose=0)[0]
            
            # Сохраняем
            all_embeddings.append(embedding)
            all_images.append({
                'path': img_path,
                'filename': img_file,
                'species': species
            })
            
        except Exception as e:
            print(f"Ошибка обработки {img_path}: {e}")

# Проверяем, есть ли данные
if len(all_embeddings) == 0:
    print("❌ Нет данных для обработки!")
    conn.close()
    exit()

print(f"Обработано {len(all_embeddings)} изображений")

# Добавление векторов в FAISS
if all_embeddings:
    embeddings_array = np.array(all_embeddings).astype('float32')
    index.add(embeddings_array)  # Добавление в индекс
    
    # Сохраняем FAISS индекс
    faiss.write_index(index, 'faiss_index.bin')
    print("✅ FAISS индекс сохранен")

# Сохранение информации в SQLite
for i, img_info in enumerate(all_images):
    cursor.execute('''
    INSERT INTO monkey_images 
    (filename, filepath, species, faiss_index, metadata) 
    VALUES (?, ?, ?, ?, ?)
    ''', (
        img_info['filename'],
        img_info['path'],
        img_info['species'],
        i,  # Индекс в FAISS
        json.dumps({
            'processed': True,
            'embedding_shape': embeddings_array[i].shape if 'embeddings_array' in locals() else []
        })
    ))

conn.commit()
conn.close()

print(f"✅ Векторная база данных создана!")
print(f"   • Изображений: {len(all_images)}")
print(f"   • FAISS индекс: faiss_index.bin")
print(f"   • SQLite база: monkey_database.db")