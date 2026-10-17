# Project Computer Vision: Deteksi Path Landing (RET503)

**Nama:** Zahkiatul Yasmi
**NIM:** 4222401040
**Mata Kuliah:** Computer Vision and Deep Learning (RET503)

## 1. Deskripsi Proyek (Tahap 1-2)
Proyek ini bertujuan untuk mendeteksi objek "path landing" (landasan pendaratan) menggunakan algoritma YOLO. Dataset terdiri dari foto lapangan yang memiliki path landing dan tanpa path landing (background). Dataset telah dianotasi menggunakan Roboflow dan diekspor dalam format YOLO.
*   **Jumlah Kelas:** 1 (`landing_path`)
*   **Link Dataset:** [Masukkan link Roboflow atau Google Drive di sini]

## 2. Metodologi Eksperimen (Tahap 3)
Eksperimen dilakukan menggunakan model `yolo26n` dengan dua skenario:
*   **E1 (Pretrained):** Baseline menggunakan bobot awal (Transfer Learning) dengan resolusi `imgsz=416`.
*   **E3 (Tuning):** Eksperimen tuning dengan mengubah resolusi gambar menjadi `imgsz=640`.

## 3. Hasil Eksperimen
Tabel perbandingan hasil evaluasi pada data validasi:
*(Salin tabel dari terminal output E1 vs E3 ke sini)*

**Grafik E1:**
![Hasil E1](hasil_praktik/E1_pretrained/results.png)

**Grafik E3:**
![Hasil E3](hasil_praktik/E3_tuning/results.png)

## 4. Analisis dan Hasil Test (Tahap 4-6)
*(Copy-paste analisis yang sudah kita buat sebelumnya ke sini)*

**A. Pemilihan Model Terbaik**
Meskipun eksperimen E1 dan E3 sama-sama mencapai performa luar biasa pada data validasi dengan `mAP50-95` sebesar 0.995 dan `Recall` 100%, **E1 terpilih sebagai model terbaik**. Hal ini karena E1 jauh lebih efisien dari segi waktu (15.4 menit vs 103.9 menit pada E3) dan kecepatan inferensi (42.3 ms vs 78.8 ms). 

**B. Evaluasi Model pada Data Test**
Pada pengujian akhir menggunakan data test (data yang belum pernah dilihat model), model E1 berhasil mempertahankan performa tinggi dengan:
*   **mAP50:** 0.995
*   **mAP50-95:** 0.948
Hal ini membuktikan model tidak mengalami *overfitting* dan sangat reliabel untuk mendeteksi objek *path landing* pada gambar baru.

**C. Catatan Tuning (E3)**
Pada eksperimen E3, grafik `mAP50-95` terlihat fluktuatif dan sempat *drop* di awal. Ini bukan indikasi *overfitting*, melainkan efek dari penyesuaian gradien (learning rate) saat resolusi gambar dinaikkan. Model berhasil pulih dengan cepat dan mencapai titik optimal, dibuktikan dengan nilai `val_loss` yang terus menurun hingga akhir epoch.

## 5. Cara Menjalankan Program
1. Install library: `pip install ultralytics opencv-python`
2. Jalankan pengecekan anotasi: `python program/01_cek_anotasi.py dataset_yolo`
3. Jalankan pembagian dataset: `python program/05_split_dataset_yolo.py`
4. Jalankan pelatihan: `python program/06_latih_bandingkan.py --nim 4222401040 --nama "Zahkiatul Yasmi" --model yolo26n`