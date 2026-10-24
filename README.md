# KPP Week 2 - OpenCV: Multi-Object 3D Tracker & HUD

- **Nama Lengkap** : _(isi)_
- **NRP** : _(isi)_

## Deskripsi

Program C++ OpenCV yang melacak 3 barang fisik berwarna kontras lewat webcam dan
mengestimasi posisi 3D (X, Y, Z) tiap objek dengan Pinhole Camera Model.

| Program | Fungsi |
|---|---|
| `src/calib.cpp` | Kalibrasi webcam dengan chessboard -> `yaml/kalibrasi.yaml` |
| `src/main.cpp` | Tracker multi-objek + HUD (membaca `yaml/kalibrasi.yaml` dengan `cv::FileStorage`) |
| `include/tracker.hpp` | Kelas `CameraParams`, `ObjectTracker`, `Detection`, `SharedParams` |

## Objek yang Dilacak

> Isi dengan hasil pengukuran dan pencatatan sendiri, lalu samakan dengan nilai di `src/main.cpp`.

| Objek | W_real (cm) | H low-high | L low-high | S low-high |
|---|---|---|---|---|
| Botol Biru | _..._ | _..._ | _..._ | _..._ |
| Kotak Spidol Merah | _..._ | _..._ | _..._ | _..._ |
| Bola Kuning | _..._ | _..._ | _..._ | _..._ |

Foto objek bersama penggaris: `docs/objek_fisik.jpg`.
Screenshot deteksi ketiga objek: `docs/objek_detect.png`.

## Cara Penggunaan

Semua perintah dijalankan dari **folder utama proyek** (supaya path `yaml/` dan `docs/` benar).

```bash
# Build (butuh OpenCV 4 dan CMake)
mkdir build && cd build
cmake .. && make
cd ..

# 1. Kalibrasi webcam (chessboard bawaan soal: 9x6 kotak = 8x5 sudut dalam)
./build/calib                      # atau: ./build/calib <kolom> <baris> <ukuran_kotak_mm>
#   s = simpan frame (chessboard harus terdeteksi), c = kalibrasi (min 15 frame), q = keluar
#   Hasil: yaml/kalibrasi.yaml

# 2. Jalankan tracker
./build/main
#   s = simpan screenshot + log koordinat 3D ke docs/, q / ESC = keluar
```

Tanpa CMake:

```bash
g++ -std=c++17 -Iinclude src/calib.cpp -o calib `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -Iinclude src/main.cpp  -o main  `pkg-config --cflags --libs opencv4`
```

### Trackbar
- Satu jendela **Threshold - <nama objek>** per objek: H/L/S low-high.
  Jika `H low > H high`, rentang Hue dianggap melewati 179 -> 0 (diperlukan untuk warna merah).
- Jendela **Parameter**: luas kontur minimum, iterasi erode, iterasi dilate, dan jarak
  *proximity warning* (cm).

### Output tombol `s`
- `docs/screenshot_<waktu>.png` - frame HUD saat ditekan.
- `docs/log_koordinat.csv` - kolom: `waktu, objek, x_pixel, y_pixel, w_pixel, X_cm, Y_cm, Z_cm`
  (ditambahkan di akhir file setiap kali `s` ditekan).

## Rumus yang Dipakai

```
Z = (fx * W_real) / w_pixel
X = ((x_pixel - cx) * Z) / fx
Y = ((y_pixel - cy) * Z) / fy
```

fx, fy, cx, cy dibaca dari `yaml/kalibrasi.yaml`; tidak ada nilai yang di-hardcode.

## Alur Pemrosesan

1. BGR -> HLS (`cv::cvtColor`)
2. Mask per objek (`cv::inRange`)
3. Morfologi: erode -> dilate -> erode
4. `cv::findContours` -> ambil kontur terbesar yang lolos filter `cv::contourArea`
5. `cv::boundingRect` + centroid dari momen kontur
6. Estimasi Z, X, Y -> HUD (nama, Z dalam cm, (X, Y, Z), FPS, proximity warning)

## Analisis Singkat & Kendala

_(isi dengan analisis dan kendala yang kalian temui sendiri, misalnya: akurasi Z pada jarak
dekat vs jauh, pengaruh pencahayaan terhadap threshold HLS, nilai reprojection error
kalibrasi, dsb.)_