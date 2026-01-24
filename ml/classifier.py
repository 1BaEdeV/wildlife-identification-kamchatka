import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import yaml, json

# Загружаем параметры
with open('./params.yaml') as f:
    params = yaml.safe_load(f)

# Параметры
EPOCHS = params['train']['epochs']
BATCH_SIZE = params['train']['batch_size']
DATA_DIR = "training_dataset"

IMG_SIZE = (224,224)

# 1. Подготовка данных
datagen = ImageDataGenerator(
    rescale=1./255,
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

# 2. Создание модели
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Замораживаем сначала
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

# 3. Компиляция и обучение
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Начало обучения...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)

# 4. Сохранение модели
os.makedirs("models", exist_ok=True)
model.save("models/monkey_classifier.h5")

# 5. Сохранение метрик
val_acc = history.history['val_accuracy'][-1]
val_loss = history.history['val_loss'][-1]

metrics = {
    "val_accuracy": float(val_acc),
    "val_loss": float(val_loss),
    "train_accuracy": float(history.history['accuracy'][-1]),
    "classes": train_gen.class_indices
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Обучение завершено!")
print(f"📊 Точность на валидации: {val_acc:.2%}")
print(f"📁 Модель сохранена: models/monkey_classifier.h5")