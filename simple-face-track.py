import cv2
import numpy as np

# Inisialisasi Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buka kamera
cap = cv2.VideoCapture(0)

# Periksa jika kamera berhasil dibuka
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

# Variabel untuk posisi wajah sebelumnya (untuk stabilitas)
prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
alpha = 0.2  # Koefisien smoothing (semakin kecil, semakin halus)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah warna menjadi grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        # Ambil wajah pertama yang terdeteksi
        x, y, w, h = faces[0]

        # Terapkan low-pass filter untuk stabilisasi
        x = int(alpha * x + (1 - alpha) * prev_x)
        y = int(alpha * y + (1 - alpha) * prev_y)
        w = int(alpha * w + (1 - alpha) * prev_w)
        h = int(alpha * h + (1 - alpha) * prev_h)

        # Simpan posisi wajah saat ini untuk iterasi berikutnya
        prev_x, prev_y, prev_w, prev_h = x, y, w, h

        # Perbesar area wajah
        zoom_factor = 2  # Skala zoom
        x1 = max(0, x - int(w * (zoom_factor - 1) / 2))
        y1 = max(0, y - int(h * (zoom_factor - 1) / 2))
        x2 = min(frame.shape[1], x + w + int(w * (zoom_factor - 1) / 2))
        y2 = min(frame.shape[0], y + h + int(h * (zoom_factor - 1) / 2))

        # Potong area wajah dari frame
        zoomed_face = frame[y1:y2, x1:x2]

        # Resize area wajah agar memenuhi seluruh jendela
        zoomed_face = cv2.resize(zoomed_face, (frame.shape[1], frame.shape[0]))

        # Tampilkan area zoom
        cv2.imshow("Face Zoom", zoomed_face)
    else:
        # Tampilkan frame asli jika wajah tidak terdeteksi
        cv2.imshow("Face Zoom", frame)

    # Perintah keluar: Tekan 'q' untuk keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Keluar program
        break
    elif key == ord('r'):  # Reset zoom (kembali ke frame asli)
        prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
