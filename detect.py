import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def detect_faces(model_path='models/face_model.h5', class_indices_path='models/class_indices.npy'):
    # 1. Загрузка модели и меток классов
    model = load_model(model_path)
    class_indices = np.load(class_indices_path, allow_pickle=True).item()
    class_names = list(class_indices.keys())

    # 2. Загрузка каскада для обнаружения лиц (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Захват видео с камеры
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Обнаружение лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # 5. Подготовка области лица для модели
            face_roi = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (150, 150))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            # 6. Предсказание
            predictions = model.predict(face_input)
            class_id = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            name = class_names[class_id]

            # 7. Отрисовка рамки и подписи
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{name} ({confidence:.2f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 8. Вывод результата
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()