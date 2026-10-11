# Advanced CNNs and Transfer Learning

Tugas Sesi 05: *Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas*

- **Nama:** Alexandria Natasya Beslar
- **Mata kuliah:** Deep Learning, BINUS University
- **Bahan rujukan:** Sesi 05, Advanced CNNs and Transfer Learning (Slide 5–30)

## Ringkasan

Repositori ini berisi implementasi dan hasil eksperimen untuk membandingkan strategi pelatihan CNN pada data terbatas. Dataset yang dipakai adalah subset 10 kelas dari Oxford-IIIT Pet (5 ras kucing dan 5 ras anjing, sekitar 2.000 citra) dengan split stratified 70% train, 15% validation, dan 15% test (seed 42).

| Kode | Kondisi | Keterangan |
|---|---|---|
| B1 | From scratch | ResNet-18 tanpa bobot pretrained |
| B2 | Feature extraction | Backbone ResNet-18 pretrained dibekukan, hanya `fc` dilatih |
| B3 | Fine-tuning | `layer4` dan `fc` dilatih dengan differential learning rate |
| B4 | B3 + augmentasi | Augmentasi hanya pada train set |
| B5 (bonus) | EfficientNet-B0 | Konfigurasi B3 pada arsitektur pembanding |

Seluruh kondisi dijalankan dengan seed 42, 43, dan 44 (split data tetap), lalu dilaporkan sebagai mean ± standar deviasi. Satu varian B1 tambahan (patience 15, satu seed) dijalankan di luar protokol.

---

Isi `logs/`:

- `<run_id>_epochs.csv`: metrik per epoch tiap run (train/val loss dan accuracy).
- `<run_id>_summary.json`: ringkasan tiap run (epoch konvergen, waktu, akurasi, Macro F1, jumlah parameter).
- `<run_id>_test_preds.npz`: label sebenarnya dan probabilitas prediksi pada test set.
- `summary_all.csv` dan `summary_agg.csv`: tabel konsolidasi (sumber Tabel 1 pada laporan).
- `split_seed42.json` dan `class_distribution.csv`: bukti pemilihan 10 kelas dan pembagian data.
- `failure_cases.csv` dan `classification_report.csv`: bahan failure analysis.
- `env.json`: versi library dan lingkungan eksekusi.

## Hasil Utama

Mean ± standar deviasi dari tiga seed (sumber: `logs/summary_agg.csv`). Baris B1 patience 15 hanya satu seed.

| Kondisi | Trainable params | Val acc | Gap train-val | Macro F1 (test) |
|---|---|---|---|---|
| B1: from scratch (patience 5) | 11.181.642 | 0,3919 ± 0,0531 | 0,0767 ± 0,0185 | 0,3663 ± 0,0618 |
| B1: from scratch (patience 15, n=1) | 11.181.642 | 0,4696 | 0,4733 | 0,5137 |
| B2: feature extraction | 5.130 | 0,8863 ± 0,0052 | 0,0925 ± 0,0056 | 0,8754 ± 0,0029 |
| B3: fine-tuning layer4 + fc | 8.398.858 | 0,9077 ± 0,0078 | 0,0909 ± 0,0091 | 0,9015 ± 0,0097 |
| B4: B3 + augmentasi | 8.398.858 | 0,9009 ± 0,0039 | 0,0601 ± 0,0277 | 0,8726 ± 0,0167 |
| B5: EfficientNet-B0 (konfigurasi B3) | 1.142.202 | 0,9257 ± 0,0089 | 0,0731 ± 0,0077 | 0,9279 ± 0,0090 |

Analisis lengkap dan batasan studi ada di laporan PDF. Test set hanya sekitar 297 citra (standard error akurasi sekitar 1,7 poin), sehingga selisih kecil antar kondisi sebaiknya dibaca sebagai tren, bukan temuan pasti.

## Cara Menjalankan

Notebook dirancang untuk Google Colab dengan runtime GPU T4.

1. Unggah `GSLC_DEEP_LEARNING_SESI_5.ipynb` ke Google Colab dan pilih runtime **GPU (T4)**.
2. Instal dependensi bila perlu:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan sel dari atas ke bawah (**Run all**). Sel pertama me-mount Google Drive dan membuat folder kerja `MyDrive/Tugas05` (variabel `BASE`) berisi `logs/`, `figures/`, dan `checkpoints/`.
4. Bila sesi Colab terputus, set `RESUME = True` pada bagian eksperimen lalu jalankan ulang. Run yang sudah selesai akan dilewati.
5. Setelah seluruh eksperimen selesai, jalankan bagian konsolidasi, analisis B4, B5, failure analysis, dan Grad-CAM untuk menghasilkan tabel dan gambar.
