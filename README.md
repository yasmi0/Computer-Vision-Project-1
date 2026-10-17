# Project Computer Vision: Deteksi Path Landing (RET503)

**Nama:** Zahkiatul Yasmi
**NIM:** 4222401040
**Mata Kuliah:** Computer Vision and Deep Learning (RET503)

## 1. Deskripsi Proyek
Proyek ini bertujuan untuk mendeteksi objek landing path (landasan pendaratan) yang digunakan untuk PBL drone fixed-wing dengan menggunakan algoritma YOLO. Dataset terdiri dari foto lapangan yang memiliki path landing dan tanpa path landing (background). Dataset telah dianotasi menggunakan Roboflow dan diekspor dalam format YOLO.

*   **Jumlah Kelas:** 1 (`landing_path`)
*   **Link Dataset:** [Google Drive Dataset](https://drive.google.com/drive/folders/1Yt304TzYQOO1anPIL8j070CVePjdHtKd?usp=sharing)

## 2. Metodologi Eksperimen
Eksperimen dilakukan menggunakan model `yolo26n` dengan dua skenario:
*   **E1 (Pretrained):** Baseline menggunakan bobot awal (Transfer Learning) dengan resolusi `imgsz=416`.
*   **E3 (Tuning):** Eksperimen tuning dengan mengubah resolusi gambar menjadi `imgsz=640`.

## 3. Hasil Eksperimen
Berikut adalah grafik hasil evaluasi dari kedua eksperimen:

**Grafik E1:**
![Hasil E1](hasil_praktik/E1_pretrained/results.png)

**Grafik E3:**
![Hasil E3](hasil_praktik/E3_tuning/results.png)

## 4. Analisis dan Hasil Test

### A. Perbandingan E1 dan E3 pada Data Validasi
Eksperimen dilakukan dengan dua konfigurasi model `yolo26n`, yaitu E1 (pretrained, imgsz=416) dan E3 (tuning, imgsz=640). Perbandingan keduanya dirangkum pada tabel berikut.

| Aspek | E1 (Pretrained, imgsz=416) | E3 (Tuning, imgsz=640) | Interpretasi |
| :--- | :--- | :--- | :--- |
| Waktu pelatihan | 15,4 menit | 103,9 menit | E1 sekitar 6,7 kali lebih cepat |
| Waktu inferensi | 42,3 ms | 78,8 ms | E1 sekitar 1,9 kali lebih cepat |
| mAP50-95 validasi | 0,995 | 0,995 | Performa validasi setara |
| Recall validasi | 1,000 / 100% | 1,000 / 100% | Keduanya mampu menemukan seluruh objek validasi |
| Stabilitas kurva | Relatif stabil | mAP50-95 lebih fluktuatif, sempat drop di awal epoch | E3 lebih sensitif terhadap perubahan resolusi |
| Keputusan | **Model terpilih** | Tidak dipilih | E1 lebih efisien dan stabil |

Berdasarkan tabel tersebut, E1 dan E3 sama-sama mencapai performa validasi yang sangat tinggi, dengan mAP50-95 sebesar 0,995 dan Recall 100%. Artinya, dari sisi akurasi validasi, peningkatan resolusi pada E3 dari 416 menjadi 640 tidak memberikan peningkatan metrik yang signifikan. Sebaliknya, E3 membutuhkan waktu pelatihan jauh lebih lama dan inferensi lebih lambat. Oleh karena itu, E1 dipilih sebagai model terbaik karena memberikan performa yang setara namun jauh lebih efisien secara komputasi. Dengan kata lain, E3 bukanlah model yang gagal, tetapi tidak cukup kompetitif dari sisi efisiensi dibandingkan E1 untuk dataset dan tugas deteksi ini.

### B. Evaluasi Model Terpilih (E1) pada Data Test
Setelah E1 terpilih, model diuji pada data test yang belum pernah dilihat selama pelatihan maupun validasi. Hasil pengujian E1 adalah sebagai berikut.

| Metrik | Nilai pada Data Test |
| :--- | :--- |
| mAP50 | 0,995 |
| mAP50-95 | 0,948 |

Hasil ini menunjukkan bahwa E1 mampu mempertahankan performa tinggi pada data baru. Nilai mAP50-95 pada data test sebesar 0,948, sedikit lebih rendah dibandingkan validasi sebesar 0,995, tetapi masih tergolong sangat baik. Penurunan kecil ini wajar karena data test benar-benar belum pernah dilihat model. Dengan demikian, model E1 menunjukkan kemampuan generalisasi yang baik dan tidak mengalami overfitting. Pada laporan ini, evaluasi data test difokuskan pada E1 sebagai model terpilih, karena E1 unggul dari sisi efisiensi dan stabilitas dibandingkan E3.

### C. Analisis Fluktuasi mAP50-95 dan Perbandingan E1 vs E3
Meskipun mAP50 dan Recall terlihat sangat stabil dan langsung mencapai nilai maksimal, metrik mAP50-95 terlihat lebih fluktuatif. Hal ini terjadi karena mAP50-95 menggunakan ambang IoU yang jauh lebih ketat, yaitu dari 0,50 hingga 0,95. Pergeseran *bounding box* hanya sebesar 1–2 piksel dapat menurunkan nilai mAP50-95 secara signifikan. Karena ukuran dataset validasi relatif kecil, perubahan pada satu atau dua gambar saja dapat memengaruhi rata-rata nilai secara keseluruhan.

Perbedaan stabilitas antara E1 dan E3 juga terlihat pada grafik. E1 cenderung lebih stabil karena menggunakan konfigurasi pretrained pada resolusi 416. Sementara itu, E3 sempat mengalami penurunan sementara pada mAP50-95 di awal epoch, sekitar epoch ke-6. Penurunan ini bukan berarti E3 gagal, melainkan efek dari penyesuaian bobot dan *learning rate* ketika resolusi gambar dinaikkan menjadi 640. Model sempat "kaget" dan mengambil langkah gradien yang terlalu jauh, tetapi YOLO mampu melakukan koreksi dan pulih pada epoch berikutnya. Fenomena ini menunjukkan bahwa E3 lebih sensitif terhadap perubahan konfigurasi resolusi dibandingkan E1.

Meskipun demikian, fluktuasi tersebut bukan indikasi overfitting. Berdasarkan kurva *loss* pada `results.png`, nilai *validation loss* terus menurun secara konsisten hingga epoch ke-75, misalnya `val/box_loss` mencapai 0,207, sementara mAP50-95 meningkat hingga mencapai puncak 0,995. Jika model mengalami overfitting, kurva *validation loss* biasanya membentuk pola "U", yaitu turun lalu naik tajam di akhir pelatihan. Pola tersebut tidak terjadi pada model ini. 

Ultralytics secara otomatis menyimpan bobot terbaik pada `best.pt`, yaitu epoch dengan mAP50-95 tertinggi. Oleh karena itu, model E1 yang diuji pada data test merupakan versi paling optimal. Dengan hasil test mAP50 sebesar 0,995 dan mAP50-95 sebesar 0,948, dapat disimpulkan bahwa E1 lebih layak dipilih dibandingkan E3 karena memberikan akurasi validasi yang setara, stabilitas yang lebih baik, serta biaya pelatihan dan inferensi yang jauh lebih rendah.

### D. Kesimpulan Perbandingan E1 dan E3
Secara ringkas, E1 dan E3 sama-sama mencapai performa validasi yang sangat tinggi. Namun, E3 dengan `imgsz=640` tidak memberikan peningkatan akurasi yang sepadan dengan biaya komputasi yang dikeluarkan. E3 membutuhkan waktu pelatihan sekitar 6,7 kali lebih lama dan inferensi sekitar 1,9 kali lebih lambat dibandingkan E1. Selain itu, kurva mAP50-95 E3 lebih fluktuatif dan sempat mengalami *drop* di awal epoch akibat penyesuaian resolusi. Oleh karena itu, E1 dipilih sebagai model terbaik karena lebih efisien, lebih stabil, dan tetap mampu melakukan generalisasi dengan sangat baik pada data test, dibuktikan dengan mAP50 sebesar 0,995 dan mAP50-95 sebesar 0,948.

## 5. Cara Menjalankan Program
1. Install library: `pip install ultralytics opencv-python`
2. Jalankan pengecekan anotasi: `python program/01_cek_anotasi.py dataset_yolo`
3. Jalankan pembagian dataset: `python program/03_split_dataset_yolo.py`
4. Jalankan pelatihan: `python program/04_latih_bandingkan.py --nim 4222401040 --nama "Zahkiatul Yasmi" --model yolo26n`
