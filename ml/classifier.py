import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import os
import yaml, json
import shutil

# ДЕБАГ: Проверка директории
print(f"Текущая директория: {os.getcwd()}")
print(f"Существует ли models: {os.path.exists('models')}")

# Загружаем параметры
with open('./params.yaml') as f:
    params = yaml.safe_load(f)

# Параметры
EPOCHS = params['train']['epochs']
BATCH_SIZE = params['train']['batch_size']
DATA_DIR = "training_dataset"

# 1. Подготовка данных
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. Создание модели EfficientNetB0
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Замораживаем сначала
base_model.trainable = False

# Простая модель поверх EfficientNet
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# 3. Компиляция
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Обучение
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)

# 5. СОЗДАЕМ ПАПКУ ПЕРЕД СОХРАНЕНИЕМ
model_dir = "models"
if not os.path.exists(model_dir):
    print(f"Создаем папку {model_dir}")
    os.makedirs(model_dir)

# Полный путь для сохранения
model_path = os.path.join(model_dir, "monkey_classifier.h5")
print(f"Сохраняем модель по пути: {model_path}")

# Сохраняем модель
model.save(model_path)

# 6. Проверяем, что файл создан
if os.path.exists(model_path):
    print(f"✓ Файл создан успешно, размер: {os.path.getsize(model_path)} байт")
else:
    print(f"✗ Файл не создан!")

# 7. Сохранение метрик
val_acc = history.history['val_accuracy'][-1]
val_loss = history.history['val_loss'][-1]

metrics = {
    "val_accuracy": float(val_acc),
    "val_loss": float(val_loss),
    "train_accuracy": float(history.history['accuracy'][-1]),
    "classes": train_gen.class_indices
}

metrics_path = "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
    print(f"✓ Метрики сохранены в {metrics_path}")

print(f"\n✅ Обучение завершено!")
print(f"📊 Точность на валидации: {val_acc:.2%}")
print(f"📁 Модель сохранена: {model_path}")