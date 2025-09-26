import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

def train_face_model(dataset_path='data', model_save_path='models/face_model.h5'):
    # 1. Подготовка данных
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # 2. Создание модели на основе VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Замораживаем слои VGG16

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    # 3. Компиляция
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Обучение
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator
    )

    # 5. Сохранение модели и меток классов
    os.makedirs('models', exist_ok=True)
    model.save(model_save_path)
    np.save('models/class_indices.npy', train_generator.class_indices)

    print(f"Модель сохранена в {model_save_path}")
    print("Классы:", train_generator.class_indices)

if __name__ == "__main__":
    train_face_model()