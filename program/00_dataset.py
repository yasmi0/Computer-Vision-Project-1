import shutil
from pathlib import Path

FOLDER_ROBOFLOW = Path("dataset_roboflow") 
OUT = Path("dataset_yolo")

if not FOLDER_ROBOFLOW.exists():
    print(f"ERROR: Folder '{FOLDER_ROBOFLOW}' tidak ditemukan!")
    exit()

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

# Tulis classes.txt (sesuaikan jika nama kelasmu bukan landing_path)
(OUT / "classes.txt").write_text("landing_path\n")

jumlah_gambar = 0

# Pencarian otomatis untuk semua folder bernama 'images'
for folder_gambar in FOLDER_ROBOFLOW.rglob("images"):
    # Cari folder 'labels' yang sejajar dengan folder 'images'
    folder_label = folder_gambar.parent / "labels"
    
    if folder_gambar.is_dir() and folder_label.exists():
        print(f"Memproses gambar dari: {folder_gambar}")
        
        # Copy gambar
        for img in folder_gambar.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy(img, OUT / "images" / img.name)
                jumlah_gambar += 1
                
        # Copy label
        for lbl in folder_label.glob("*.txt"):
            shutil.copy(lbl, OUT / "labels" / lbl.name)

print(f"\nSelesai! Berhasil menggabungkan {jumlah_gambar} gambar ke dalam folder '{OUT}'")

if jumlah_gambar == 0:
    print("PERINGATAN: 0 gambar ditemukan. Pastikan file ZIP dari Roboflow sudah di-EKSTRAK (di-unzip) ke dalam folder 'dataset', bukan masih berbentuk file .zip.")