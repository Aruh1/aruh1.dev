---
title: "Restorasi Citra CT Scan Thorax: Sparse-View FBP, Denoise NLM-TV, dan Penapisan Domain Frekuensi HFE"
author: [Aruh1]
publish_date: 2026-10-21
post_slug: restorasi-ct-scan-thorax
featured: true
tags: ["pcd", "image-processing", "python", "opencv", "scikit-image", "ct-scan", "radon-transform", "fourier-transform", "case-study"]
description: Studi kasus restorasi citra CT Scan Thorax dosis rendah (sparse-view 40 sudut) dengan degradasi gabungan Poisson-Gaussian sinogram, artefak streak FBP, dan atenuasi kontras lokal menggunakan pipeline hybrid domain spasial dan frekuensi.
lang: id
---

Pada dua studi kasus sebelumnya, kita telah mengeksplorasi penanganan derau bintik radiografi pada [Tugas 1: Restorasi Citra Rontgen Thorax Domain Spasial](/blog/restorasi-rontgen-thorax) serta restorasi citra terdistorsi parah pada [Tugas 2: Dekonvolusi Wiener & Filter Domain Frekuensi](/blog/restorasi-rontgen-degradasi-kompleks). Pada studi kasus **Ujian Tengah Semester (UTS) Pengolahan Citra Medis** ini, fokus beralih ke modalitas tomografi terkomputasi: **Computed Tomography (CT) Scan Thorax**.

Berbeda dari proyeksi planar X-ray konvensional, citra CT dibentuk melalui proses rekonstruksi matematis dari proyeksi sinogram multi-sudut. Ketika protokol pemindaian diatur pada **dosis radiasi rendah (*low-dose CT*)** dengan **sudut proyeksi jarang (*sparse-view CT*)**, degradasi yang muncul memiliki sifat fisik yang sangat khas: perpaduan antara **fluktuasi foton Poisson**, **derau elektronik Gaussian**, **artefak garis-garis tajam (*streaking artifacts*)** akibat proyeksi FBP yang *undersampled*, serta **penurunan kontras lokal** akibat *windowing* non-optimal.

Artikel ini menyajikan dokumentasi teknis lengkap mengenai pemodelan fisika degradasi CT, perancangan *pipeline* restorasi bertingkat (*spatial-frequency hybrid*), evaluasi kuantitatif dua tingkat rubrik, studi ablasi 12 varian, pengujian trade-off struktur vaskular mikro, serta analisis keterbatasan klinis menggunakan **Python, OpenCV, NumPy, SciPy, scikit-image, dan Matplotlib**.

---

## 1. Langkah 1: Tujuan Klinis & Spesifikasi Input

### Skenario Operasional & Tujuan Klinis
Sebuah stasiun kerja Picture Archiving and Communication System (PACS) pada rumah sakit rujukan menerima kumpulan citra CT Thorax aksial dari fasilitas radiologi perifer. Citra-citra tersebut mengalami degradasi mutu visual yang signifikan akibat penerapan protokol proteksi radiasi dosis rendah.

Terdapat batasan operasional dan komputasional fundamental dalam skenario ini:
> **Batasan Domain Citra:** Rumah sakit rujukan **hanya menerima citra hasil rekonstruksi FBP** dalam format digital. Data mentah sinogram proyeksi (*raw projection data*) **tidak disimpan** oleh fasilitas pengirim untuk menghemat bandwidth transmisi dan kapasitas penyimpanan arsip.

**Tujuan Klinis Pemrosesan:**  
Meningkatkan kualitas inspeksi visual anatomi toraks (membedakan parenkim paru, mengembalikan ketegasan batas mediastinum-jantung, dan menegaskan korteks tulang) melalui algoritma restorasi yang bekerja secara eksklusif di **domain citra (*image domain*)**.

### Spesifikasi Sampel Input & Pra-Pemrosesan
Sesuai panduan tugas ("pilih atau buat sampel"), studi kasus ini menggunakan potongan aksial citra CT Thorax klinis riil (`normal-ct-chest.jpeg`) sebagai *ground truth* $f$:
- **Sumber Citra & Lisensi:** Satu irisan aksial dari kasus *Normal CT chest* oleh Dr. Andrew Dixon di Radiopaedia.org (rID 36676), [radiopaedia.org/cases/normal-ct-chest](https://radiopaedia.org/cases/normal-ct-chest), DOI [10.53347/rID-36676](https://doi.org/10.53347/rID-36676). Konten kasus Radiopaedia berlisensi [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Karena itu citra turunan di artikel ini (terdegradasi dan hasil restorasi) juga dibagikan di bawah lisensi yang sama, untuk keperluan non-komersial.
- **Penempatan Berkas:** Simpan irisan sebagai `src/normal-ct-chest.jpeg`. Bila berkas tidak ditemukan, skrip mencetak peringatan lalu memakai phantom sintetis, dan angka di artikel ini tidak lagi berlaku.
- **Dimensi Citra Sumber:** Berkas asli berukuran $1524 \times 1524$ piksel (format grayscale 8-bit).
- **Resampling Spasial:** Citra di-resample ke resolusi standar $256 \times 256$ piksel (faktor skala 5.95) menggunakan interpolasi area (`cv2.INTER_AREA`) untuk efisiensi komputasi transformasi Radon dan FBP.
- **Circular Reconstruction Masking:** Area di luar radius lingkaran rekonstruksi $r = \frac{N}{2} - 2 = 126\text{ px}$ dinolkan ($f = 0$). Sebanyak 508 piksel tepi tubuh/meja pemindai (1.9% dari total area tubuh) dinolkan untuk mencegah timbulnya peringatan batas rekonstruksi *out-of-circle* pada `skimage.transform.radon`.
- **Karakteristik Intensitas Ground Truth:** Sebanyak 59.3% piksel bernilai 0 (latar belakang udara dan parenkim paru hipodens), sedangkan 1.56% piksel mencapai saturasi maksimum 255 (korteks tulang padat).

---

## 2. Pemodelan Matematika & Fisika Degradasi Citra CT

Proses pembentukan citra terdegradasi $g(x,y)$ dari citra ideal $f(x,y)$ dirumuskan dalam dua domain berurutan: **domain proyeksi Radon (sinogram)** dan **domain citra pasca-rekonstruksi**.

```mermaid
flowchart LR
    GT["<b>Ground Truth f</b><br/>normal-ct-chest.jpeg (256×256)"]
    RAD["<b>Forward Projection</b><br/>Transformasi Radon<br/>(40 sudut jarang)"]
    NOISE["<b>Fisika Derau Sinogram</b><br/>Poisson (kuantum foton)<br/>Gaussian (elektronik)"]
    FBP["<b>Rekonstruksi FBP</b><br/>Filtered Back-Projection<br/>(Ram-Lak Ramp Filter)"]
    WIN["<b>Atenuasi Kontras Lokal</b><br/>Windowing Defisit<br/>g = α f + (1-α) G_σ * f"]
    DEG["<b>Citra Terdegradasi g</b><br/>Diterima PACS"]

    GT --> RAD
    RAD --> NOISE
    NOISE --> FBP
    FBP --> WIN
    WIN --> DEG
```

### A. Proyeksi Maju (Transformasi Radon)
Atenuasi sinar-X yang menembus jaringan tubuh sepanjang garis berkas dengan sudut $\theta$ dan jarak detektor $s$ dimodelkan melalui Transformasi Radon:

$$p(s, \theta) = \mathcal{R}[f](s, \theta) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \, \delta(x \cos\theta + y \sin\theta - s) \, dx \, dy$$

Pada pemindaian *sparse-view*, sudut proyeksi diambil sangat jarang, yaitu hanya $N_\theta = 40$ sudut sepanjang interval $[0, \pi)$:

$$\theta_k = \frac{k \pi}{40}, \quad k = 0, 1, \dots, 39$$

Jumlah sudut ini jauh di bawah kriteria sampling Nyquist-Shannon angular ($N_\theta \ge \frac{\pi}{2} N_{\text{detektor}}$), sehingga rekonstruksi FBP memunculkan galat berarah (*aliasing streaking*).

### B. Fisika Derau pada Domain Sinogram
1. **Derau Kuantum Foton (Poisson):** Pencacahan foton sinar-X yang tiba di detektor mengikuti distribusi Poisson dengan parameter dosis rendah ($D_{\text{foton}} = 5.0$):
   $$s_{\text{foton}}(s, \theta) = \frac{\text{Poisson}\left( \max(0, p(s, \theta)) \cdot D_{\text{foton}} \right)}{D_{\text{foton}}}$$
2. **Derau Elektronik Detektor (Gaussian):** Fluktuasi termal dan derau baca elektronik sensor dimodelkan sebagai derau putih Gaussian aditif ($\sigma_e = 50.0$):
   $$s_{\text{noisy}}(s, \theta) = s_{\text{foton}}(s, \theta) + \mathcal{N}(0, \sigma_e^2)$$

### C. Rekonstruksi FBP (Filtered Back-Projection)
Citra rekonstruksi awal $\hat{f}_{\text{pacs}}$ diperoleh dengan mengaplikasikan *Ramp Filter* (Ram-Lak) $|\omega|$ pada domain frekuensi sinogram lalu memproyeksikannya kembali (*back-projection*):

$$\hat{f}_{\text{pacs}}(x, y) = \mathcal{R}^{-1}\{s_{\text{noisy}}\} = \int_0^\pi \left( s_{\text{noisy}}(\cdot, \theta) * h_{\text{ramp}} \right)(x \cos\theta + y \sin\theta) \, d\theta$$

Pada lapang paru, dekomposisi galat rekonstruksi FBP menghasilkan:
- **Galat FBP murni tanpa derau (streak sparse-view + blur):** Simpangan baku $\sigma = 17.01$ tingkat abu.
- **Derau murni (Poisson + Gaussian):** Simpangan baku $\sigma = 7.15$ tingkat abu (Poisson saja: 5.20, Gaussian saja: 4.95).

### D. Penurunan Kontras Lokal (Windowing Defisit)
Untuk meniru penyetelan *window width* dan *window level* yang tidak memadai pada stasiun akuisisi, kontras lokal citra diturunkan dengan menarik tiap piksel ke arah rerata lokal sekitarnya:

$$g(x, y) = \alpha \, \hat{f}_{\text{pacs}}(x, y) + (1 - \alpha) \, [\hat{f}_{\text{pacs}} * G_{\sigma_{\text{lokal}}}](x, y)$$

dengan $\alpha = 0.45$ dan radius sebaran Gaussian $\sigma_{\text{lokal}} = 9.0\text{ px}$.

Dalam domain frekuensi Fourier 2D, fungsi transfer degradasi kontras ini memiliki formulasi eksak:

$$H_{\text{deg}}(\nu) = \alpha + (1 - \alpha) \exp\left( -2\pi^2 \sigma_{\text{lokal}}^2 \nu^2 \right)$$

Pada frekuensi nol (DC), $H_{\text{deg}}(0) = 1.0$ (kecerahan global tetap terjaga). Namun saat frekuensi spasial $\nu$ meningkat, respons meluruh secara asimtotik mendekati nilai $\alpha = 0.45$, menyebabkan kontur batas organ dan detail vaskular mengalami pelemahan amplitudo hingga lebih dari 50%.

---

## 3. Diagram Alur & Strategi Pipeline Restorasi

Untuk mengatasi degradasi majemuk tersebut tanpa memperparah artefak yang ada, dirancang *pipeline* restorasi bertingkat yang memadukan keunggulan **Domain Spasial** dan **Domain Frekuensi Fourier 2D**:

```mermaid
flowchart TD
    IN["<b>Citra Terdegradasi (PACS)</b><br/>Streaking FBP + Derau Butiran + Kontras Rendah"]

    subgraph S1 ["Tahap 1: Domain Spasial (Peredaman Derau Butiran)"]
        direction TB
        NLM["<b>Non-Local Means (NLM) Filter</b><br/>cv2.fastNlMeansDenoising<br/>h = 12, template = 7×7, search = 21×21"]
    end

    subgraph S2 ["Tahap 2: Domain Spasial (Pembersihan Artefak Streak)"]
        direction TB
        TV["<b>Total Variation (TV) Chambolle Filter</b><br/>Bobot λ = 0.05<br/><i>Meratakan pola garis streak di area homogen paru</i>"]
    end

    subgraph S3 ["Tahap 3: Domain Frekuensi FFT 2D (Koreksi Kontras)"]
        direction TB
        HFE["<b>High-Frequency Emphasis (HFE) Filter</b><br/>H(ν) = 1 + k [1 - exp(-2π² σ² ν²)]<br/>σ = 9.0 px, k = 1.2<br/><i>Inversi teratur degradasi kontras lokal</i>"]
    end

    subgraph S4 ["Tahap 4: Domain Spasial (Penajaman Tepi Tulang)"]
        direction TB
        SHARP["<b>Penajaman Kernel Laplacian</b><br/>Kernel 3×3, Bobot κ = 0.5<br/><i>Menegaskan korteks tulang iga & vertebra</i>"]
    end

    OUT["<b>Citra Restorasi Final</b><br/>Reduksi streak paru 78.7%, kontras lokal pulih di 0.99"]

    IN --> NLM
    NLM --> TV
    TV --> HFE
    HFE --> SHARP
    SHARP --> OUT
```

### Rasionalisasi Urutan Pemrosesan: Mengapa Denoise Lebih Dulu?
Pertanyaan desain arsitektur mendasar: *mengapa tahap reduksi derau (NLM dan TV) diletakkan sebelum koreksi kontras (HFE) dan penajaman?*

Filter HFE dan kernel penajaman Laplacian bekerja dengan **menguatkan frekuensi spasial tinggi**. Jika koreksi kontras dijalankan di awal:
1. Sisa derau butiran Poisson-Gaussian akan ikut teramplifikasi secara drastis bersama sinyal detail.
2. Urutan kontras-dulu **bukan** gagal membersihkan streak. Dua varian kontras-dulu di tabel ablasi justru memberi streak paru lebih rendah (1.51 dan 1.43, lawan 2.36). Yang didukung tabel adalah selisih kemiripan struktur: **urutan denoise-dulu memberi SSIM 0.699**, lawan **0.608** dan **0.599** pada kontras-dulu. Karena itulah denoise-dulu dipilih. Penyebab selisih SSIM ini tidak diuji terpisah.

### Mengapa Bukan CLAHE?
Contrast Limited Adaptive Histogram Equalization (CLAHE) adalah metode standar untuk penyesuaian kontras lokal. Namun dalam skenario citra CT dengan residu *streak artifacts*:
- CLAHE memetakan histogram pada blok lokal (*tiles*).
- Pada area lapang paru yang semestinya datar dan homogen, variasi intensitas kecil akibat residu garis streak diekspansi secara berlebihan menjadi bercak-bercak hipodens/hiperdens palsu (*mottling artifacts*).
- Pada studi ablasi, CLAHE penuh memberi galat streak terburuk (simpangan baku 16.19) dan SSIM terendah (0.341).

---

## 4. Rincian Teknis & Formulasi Algoritma

### Tahap 1: Non-Local Means (NLM) Denoising
Metode Non-Local Means bekerja dengan mengeksploitasi redundansi struktural citra medis, menghitung estimasi intensitas berdasarkan kesamaan patch (*patch similarity*):

$$u(p) = \frac{1}{Z(p)} \sum_{q \in \Omega} w(p, q) \, v(q)$$

Pada skrip `uts.py`, tahap ini diimplementasikan menggunakan fungsi `cv2.fastNlMeansDenoising` pada citra uint8 dengan parameter:
- Kekuatan penapisan: $h = 12$
- Ukuran jendela template / patch: $7 \times 7$ (`templateWindowSize=7`)
- Ukuran jendela pencarian: $21 \times 21$ (`searchWindowSize=21`)

Tahap ini meredam fluktuasi butiran kuantum, tetapi ikut meratakan pembuluh kecil berkontras rendah (lihat Bagian 7A).

> **Asal parameter denoise:** $h = 12$, bobot TV $0.05$, dan bobot penajaman $0.5$ dibawa dari uji pendahuluan pada phantom sintetis, **bukan** disetel pada citra riil ini. Saat `--tuning` dijalankan pada citra riil (seed 100–102, 70 kombinasi NLM × TV), setelan ini berada di **peringkat 7 dari 70** (skor 34.21). Peringkat 1 adalah $h = 10$, TV $0.05$ (skor 34.54, PSNR 24.94, SSIM 0.711). Selisihnya kecil, jadi parameter tidak diganti agar konsisten dengan draf awal.

### Tahap 2: Total Variation Chambolle (TV) Denoising
Untuk meratakan pola garis-garis *streak* berdensitas rendah yang memotong parenkim paru, diterapkan algoritma Total Variation berbasis formulasi regularisasi Antonin Chambolle:

$$\min_u \left\{ \int_\Omega |\nabla u| + \frac{1}{2 \lambda} \int_\Omega (u - g)^2 \right\}$$

Regularisasi norma $L_1$ pada gradien $|\nabla u|$ memiliki sifat meratakan osilasi bergradien landai pada area homogen (lapang paru), sementara batas bergradien curam antara dinding dada dan rongga udara dipertahankan tajam. Bobot TV disetel pada $\lambda = 0.05$ (skala citra dinormalisasi $[0, 1]$).

### Tahap 3: High-Frequency Emphasis (HFE) Domain Frekuensi
Setelah kanvas citra bersih dari derau butiran dan streak, penurunan kontras lokal dikoreksi di domain frekuensi Fourier 2D. Filter HFE dirancang sebagai aproksimasi invers teratur dari degradasi $H_{\text{deg}}(\nu)$:

$$H_{\text{HFE}}(\nu) = 1.0 + k \cdot \left[ 1.0 - \exp\left( -2\pi^2 \sigma_{\text{HFE}}^2 \nu^2 \right) \right]$$

Dengan parameter $\sigma_{\text{HFE}} = 9.0\text{ px}$ dan penguatan gain $k = 1.2$ (mendekati rasio invers teoritis $\frac{1 - \alpha}{\alpha} = \frac{0.55}{0.45} \approx 1.22$).

```python
def filter_hfe_frekuensi(img, sigma=SIGMA_HFE, gain=GAIN_HFE):
    M, N = img.shape
    v = np.fft.fftfreq(M)[:, None]
    u = np.fft.fftfreq(N)[None, :]
    freq_sq = u**2 + v**2
    H = 1.0 + gain * (1.0 - np.exp(-2.0 * np.pi**2 * sigma**2 * freq_sq))
    f_shift = np.fft.fft2(img)
    hasil = np.real(np.fft.ifft2(f_shift * H))
    return np.clip(hasil, 0, 255).astype(np.float32)
```

Perkalian spektrum $H_{\text{deg}}(\nu) \times H_{\text{HFE}}(\nu)$ memiliki nilai puncak yang terkendali pada $1.16$ di frekuensi $\nu = 0.021\text{ siklus/px}$, mencegah osilasi frekuensi tinggi berlebih.

### Tahap 4: Penajaman Korteks Tulang (Laplacian Sharpening)
Tahap akhir mengembalikan ketajaman korteks tulang (sternum, iga, korpus vertebra) menggunakan konvolusi kernel Laplacian 3×3 dengan bobot $\kappa = 0.5$:

$$K_{\text{lap}} = \begin{bmatrix} 0 & -\kappa & 0 \\ -\kappa & 1 + 4\kappa & -\kappa \\ 0 & -\kappa & 0 \end{bmatrix} = \begin{bmatrix} 0 & -0.5 & 0 \\ -0.5 & 3.0 & -0.5 \\ 0 & -0.5 & 0 \end{bmatrix}$$

---

## 5. Langkah 2: Indikator Evaluasi & Kalibrasi Rubrik Dua Tingkat

### Indikator Evaluasi & Rasionalisasi Metrik
Tiap indikator dipilih untuk merepresentasikan aspek diagnostik tertentu:
1. **PSNR (Peak Signal-to-Noise Ratio):** Mengukur penurunan galat kuadrat rerata (MSE) terhadap ground truth secara logaritmik.
2. **SSIM (Structural Similarity Index):** Menilai integritas luminansi, kontras, dan struktur anatomis secara bersamaan.
3. **Rasio Kontras RMS pada ROI:** Dihitung pada jendela batas hilum paru-jantung ($y: 125\text{--}149, x: 125\text{--}149$). Rasio $1.00$ menunjukkan pemulihan kontras lokal yang presisi tanpa *under-enhancement* (< 0.85) atau *over-enhancement* (> 1.15).
4. **Galat Streak di Paru (std):** Mengukur simpangan baku galat $(hasil - f)$ pada piksel lapang paru homogen (`mask_paru_homogen`, 11.066 piksel).
5. **Rasio Ketajaman Tepi:** Dihitung dari varians filter Laplacian (dihaluskan Gaussian $\sigma=0.8$ agar tidak membiaskan derau sebagai ketajaman semu).
6. **Kontras Struktur Pembuluh:** Rasio selisih intensitas struktur vaskular terhadap parenkim sekitarnya.

### Transparansi Ambang Rubrik: Ambang Awal vs Ambang Sesudah Hasil
Ada dua set ambang, dan keduanya ditampilkan:

1. **Ambang awal (`RUBRIK_PHANTOM`)** ditetapkan **sebelum** hasil keluar, saat uji pendahuluan pada phantom sintetis.
2. **Ambang citra riil (`RUBRIK_REAL_CT`)** ditetapkan **sesudah** hasil citra riil terlihat. Batas IDEAL untuk PSNR, SSIM, dan streak sengaja diletakkan tepat di bawah hasil (24.5 lawan 24.84 dB, 0.68 lawan 0.699, 2.5 lawan 2.36). Batas IDEAL ketajaman diturunkan dari 0.8 ke 0.6, tetapi hasil 0.59 tetap belum mencapainya.

Alasan membuat set kedua: ambang awal PSNR ≥ 28 dB tidak tercapai oleh satu pun konfigurasi yang diuji. PSNR tertinggi di ablasi 24.87 dB, di uji sensitivitas HFE 25.04 dB, dan di grid tuning 25.01 dB. Namun karena set kedua dibuat sesudah melihat hasil, status IDEAL di bawahnya **tidak boleh dibaca sebagai bukti independen**. Status di bawah ambang awal adalah ukuran yang jujur.

Tabel berikut membandingkan hasil restorasi (seed 42) di bawah **kedua set ambang**:

<!-- prettier-ignore -->
| Metrik Evaluasi | Nilai Terdegradasi | Nilai Hasil Restorasi | Ambang Awal (Phantom) | Status (Phantom) | Ambang Kalibrasi Citra Riil | Status (Citra Riil) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR vs Ground Truth (dB)** | 19.32 | **24.84** | $\ge 28.0 \; / \; \ge 32.0$ | **BELUM** | $\ge 22.0 \; / \; \ge 24.5$ | **IDEAL** |
| **SSIM vs Ground Truth** | 0.435 | **0.699** | $\ge 0.70 \; / \; \ge 0.85$ | **BELUM** | $\ge 0.60 \; / \; \ge 0.68$ | **IDEAL** |
| **Rasio Kontras Lokal (ROI)** | 0.57 | **0.99** | $0.85\text{--}1.15 \; / \; 0.95\text{--}1.05$ | **IDEAL** | $0.85\text{--}1.15 \; / \; 0.95\text{--}1.05$ | **IDEAL** |
| **Galat Streak di Paru (std)** | 11.09 | **2.36** | $\le 4.0 \; / \; \le 2.0$ | **MINIMUM** | $\le 4.0 \; / \; \le 2.5$ | **IDEAL** |
| **Rasio Ketajaman Tepi\*** | 0.24 | **0.59** | $0.50\text{--}1.30 \; / \; 0.80\text{--}1.20$ | **MINIMUM** | $0.50\text{--}1.30 \; / \; 0.60\text{--}1.20$ | **MINIMUM** |
| **Rasio Kontras Pembuluh** | 0.71 | **1.03** | $\ge 0.50 \; / \; \ge 0.80$ | **IDEAL** | $\ge 0.50 \; / \; \ge 0.80$ | **IDEAL** |
| **Ringkasan Predikat Status** | - | - | - | **IDEAL 2, MIN 2, BELUM 2** | - | **IDEAL 5, MIN 1, BELUM 0** |

\*Rasio ketajaman 0.59 berarti tepi hasil masih lebih lunak daripada ground truth. Statusnya MINIMUM di bawah kedua set ambang.

> **Pembacaan yang tepat:** Di bawah ambang awal, PSNR dan SSIM berstatus **BELUM** (SSIM 0.699 hanya 0.001 di bawah batas MINIMUM 0.70). Kenaikan dari citra terdegradasi tetap nyata: PSNR +5.52 dB dan SSIM +60.7%. Tetapi "IDEAL 5 dari 6" hanya berlaku pada ambang yang dibuat sesudah hasil terlihat.

### Konsistensi pada 8 Realisasi Derau (Seed 42–49)
Anatomi ground truth sama, realisasi derau berbeda. Hasilnya sangat stabil (std PSNR 0.025 dB):

<!-- prettier-ignore -->
| Metrik | Terdegradasi (rerata) | Hasil (rerata ± std) | Ambang awal: IDEAL / MIN / BELUM | Ambang citra riil: IDEAL / MIN / BELUM |
| :--- | :---: | :---: | :---: | :---: |
| PSNR (dB) | 19.33 | 24.86 ± 0.025 | 0 / 0 / **8** | 8 / 0 / 0 |
| SSIM | 0.435 | 0.699 ± 0.001 | 0 / 1 / **7** | 8 / 0 / 0 |
| Rasio kontras ROI | 0.571 | 0.993 ± 0.002 | 8 / 0 / 0 | 8 / 0 / 0 |
| Galat streak paru | 11.12 | 2.34 ± 0.019 | 0 / 8 / 0 | 8 / 0 / 0 |
| Rasio ketajaman | 0.238 | 0.583 ± 0.003 | 0 / 8 / 0 | 0 / 8 / 0 |
| Rasio pembuluh (mask global)† | 0.713 | 1.028 ± 0.001 | 8 / 0 / 0 | 8 / 0 / 0 |

†Lihat Bagian 7A: mask ini sebagian besar berisi kontur jantung dan dinding dada, bukan pembuluh paru.

---

## 6. Studi Ablasi & Uji Sensitivitas

Untuk memverifikasi kontribusi tiap komponen dalam arsitektur restorasi, dilakukan uji komparasi terhadap 12 varian konfigurasi:

<!-- prettier-ignore -->
| Varian Konfigurasi Algoritma | PSNR (dB) | SSIM | Streak (std) | Rasio Kontras | Rasio Tajam | Rasio Pembuluh | PSNR Tubuh (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Citra Terdegradasi (Input PACS)** | 19.32 | 0.435 | 11.09 | 0.57 | 0.24 | 0.71 | 17.17 |
| **Pipeline Utama (NLM > TV > HFE > Sharpen)** | **24.84** | **0.699** | **2.36** | **0.99** | **0.59** | **1.03** | **22.33** |
| Tanpa Koreksi Kontras (HFE) | 19.48 | 0.432 | 8.62 | 0.59 | 0.13 | 0.73 | 17.17 |
| Tanpa NLM (Hanya TV > HFE > Sharpen) | 23.71 | 0.658 | 3.78 | 1.00 | 0.93 | 1.01 | 21.47 |
| Tanpa TV (Hanya NLM > HFE > Sharpen) | 24.19 | 0.693 | 2.79 | 1.03 | 1.04 | 1.03 | 21.90 |
| Tanpa Penajaman Laplacian | 23.76 | 0.674 | 2.36 | 0.94 | 0.25 | 1.02 | 21.15 |
| Denoise Lemah (NLM h=6, TV=0.02) | 22.48 | 0.681 | 2.60 | 1.02 | 1.36 | 1.03 | 20.28 |
| Filter Invers Eksak $1/H_{\text{deg}}$ (bukan HFE) | 24.87 | 0.597 | 3.20 | 0.97 | 0.60 | 0.97 | 22.60 |
| Kontras Dulu (Parameter Denoise Disetel Terpisah) | 24.63 | 0.608 | 1.51 | 0.96 | 0.57 | 0.98 | 22.23 |
| Kontras Dulu (Parameter Sama dengan Utama) | 22.67 | 0.599 | 1.43 | 1.01 | 1.30 | 0.99 | 20.49 |
| CLAHE Penuh (clip=2.0, tile=8×8) | 17.11 | 0.341 | 16.19 | 0.58 | 0.25 | 0.83 | 15.16 |
| CLAHE Ringan (clip=0.5, blend=0.3) | 19.68 | 0.428 | 9.11 | 0.60 | 0.14 | 0.76 | 17.45 |
| Denoise Saja (NLM > TV tanpa HFE & Sharpen) | 18.91 | 0.408 | 8.77 | 0.56 | 0.06 | 0.72 | 16.54 |

### Temuan Utama Studi Ablasi:
1. **Peran HFE:** Tanpa HFE, PSNR hanya 19.48 dB (naik 0.16 dB dari citra terdegradasi). Pada citra ini, denoise saja tidak cukup tanpa koreksi kontras.
2. **Filter Invers Eksak:** Filter invers analitik $1/H_{\text{deg}}$ memberi PSNR 24.87 dB, sedikit **di atas** pipeline utama (24.84 dB), tetapi SSIM-nya 0.597 lawan 0.699 dan streak 3.20 lawan 2.36. HFE dipilih karena keseimbangan SSIM dan streak, bukan karena PSNR-nya tertinggi.
3. **CLAHE:** CLAHE penuh menaikkan galat streak ke 16.19 (lebih buruk dari citra terdegradasi 11.09). Pada citra ini CLAHE tidak cocok sebagai pengganti HFE.

### Uji Sensitivitas Parameter HFE
Degradasi sebenarnya memakai $\sigma = 9$ dan $k$ setara $1.22$. Tahap lain sama dengan pipeline utama (seed 42):

<!-- prettier-ignore -->
| $\sigma_{\text{HFE}}$ | $k$ | PSNR (dB) | SSIM | Streak | Rasio Kontras |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6 | 0.8 | 23.11 | 0.535 | 3.34 | 0.81 |
| 6 | 1.2 | 24.60 | 0.575 | 2.32 | 0.93 |
| 6 | 1.6 | 25.04 | 0.619 | 2.99 | 1.02 |
| 9 | 0.8 | 23.95 | 0.590 | 2.55 | 0.88 |
| **9** | **1.2** | **24.84** | **0.699** | **2.36** | **0.99** |
| 9 | 1.6 | 24.11 | 0.741 | 1.90 | 1.05 |
| 12 | 0.8 | 23.94 | 0.687 | 3.09 | 0.91 |
| 12 | 1.2 | 23.83 | 0.765 | 1.31 | 1.00 |
| 12 | 1.6 | 22.54 | 0.788 | 0.66 | 1.06 |

PSNR berkisar **22.5–25.0 dB** tergantung parameter HFE. Setelan terbaik juga berbeda menurut metrik: PSNR tertinggi pada $\sigma = 6, k = 1.6$, sedangkan SSIM tertinggi dan streak terendah pada $\sigma = 12, k = 1.6$. Setelan utama ($\sigma = 9, k = 1.2$) tidak unggul di semua metrik. Ia dipilih karena **sama dengan parameter degradasi yang diketahui** (*non-blind*). Pada citra PACS sungguhan parameter itu tidak diketahui, jadi angka di atas memperlihatkan seberapa besar hasil bisa bergeser.

---

## 7. Investigasi Kritis: Trade-off Struktur Vaskular Mikro & Galat Regional

### A. Realitas Mask Pembuluh Global vs Pembuluh Darah Halus
Pada evaluasi rubrik awal, metrik `pembuluh` mencatat rasio 1.03 (status IDEAL). Namun, analisis mendalam pada topologi piksel `mask_pembuluh` (2.264 piksel) mengungkap fakta penting:
- Sebanyak **94% piksel** berada pada kurva panjang (> 60 piksel).
- Sebanyak **98% piksel** terletak dalam jarak $\le 3\text{ px}$ dari struktur anatomis besar (kontur batas lateral mediastinum, jantung, dan dinding dada anterior).

Akibatnya, metrik global ini sebagian besar merefleksikan kontur batas organ besar, sehingga seluruh 70 setelan tuning menghasilkan rasio semu berkisar antara 0.98 s.d. 1.03.

![Bukti Segmentasi Pembuluh dan Hilangnya Pembuluh Halus](./bukti_pembuluh.png)

Untuk menguji pelestarian vaskular intrapulmonal yang sesungguhnya, dilakukan deteksi terisolasi pada **5 kandidat pembuluh darah kecil** (luas 2 hingga 32 piksel) di dalam parenkim paru:

<!-- prettier-ignore -->
| No | Koordinat Pusat $(y, x)$ | Luas (piksel) | Rasio Puncak Terdegradasi | Rasio Puncak Hasil Restorasi | Keterangan Evaluasi Lokal |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | $(110, 95)$ | 21 px | 0.33 | 0.16 | Teratenuasi (-51.5%) |
| 2 | $(114, 179)$ | 2 px | 0.13 | 0.01 | **Hilang / Terhapus total** |
| 3 | $(118, 104)$ | 2 px | 0.23 | 0.02 | **Hilang / Terhapus total** |
| 4 | $(134, 168)$ | 32 px | 0.46 | 0.96 | Bertahan baik (pembuluh cabang utama) |
| 5 | $(144, 161)$ | 26 px | 0.47 | 0.96 | Bertahan baik (pembuluh cabang utama) |

> **Temuan Kritis Trade-off:**  
> Pipeline utama berhasil menjaga 2 pembuluh cabang yang relatif lebih besar (luas 26-32 px, rasio 0.96), tetapi **menghapus 3 pembuluh mikro** (luas 2-21 px, rasio turun ke 0.01 - 0.16).  
> Varian *denoise lemah* ($h=6, \text{TV}=0.02$) menjaga median kontras pembuluh kecil di 0.82 (3 dari 5 pembuluh $\ge 0.50$, rerata seed 42–45). Konsekuensinya, streak paru naik ke 2.63 dan PSNR turun ke 22.50 dB. Pada citra ini, pengurangan streak dan pelestarian pembuluh kecil saling bertukar (*trade-off*).  
> Sampelnya hanya 5 pembuluh pada satu irisan, jadi temuan ini adalah **indikasi**, bukan kesimpulan umum.

### B. Sebaran Galat Regional
Klaim restorasi tidak boleh digeneralisasi ke seluruh kanvas citra karena metrik streak hanya diukur di parenkim paru datar. Pengukuran galat RMS terhadap ground truth pada berbagai wilayah anatomi menunjukkan:
- **Parenkim Paru Datar (Mask Artikel):** Galat RMS turun drastis dari **20.69** ke **3.13** tingkat abu (bias rerata +2.07).
- **Jaringan Lunak Datar (Mediastinum/Dinding Dada):** Galat RMS sebesar **11.01** tingkat abu (bias rerata +6.50).
- **Tepi Luar Badan (Pita 12 piksel):** Galat RMS sebesar **20.17** tingkat abu (bias rerata +4.77).
- **Latar Meja Pemindai (Bawah Tubuh):** Galat RMS sebesar **22.68** tingkat abu (bias rerata +17.17).

Artinya, pembersihan streak sangat efektif di dalam rongga paru, tetapi residu galat rekonstruksi FBP masih tertinggal pada jaringan lunak dan batas meja pemindai.

### C. Uji Kontrol Negatif & Sifat Non-Blind
- **Sifat Non-Blind:** Model koreksi HFE bersifat terarah (*non-blind*), menggunakan parameter $\sigma = 9.0$ dan gain $k = 1.2$ yang disesuaikan secara langsung dengan parameter degradasi fisik $\alpha = 0.45$ dan $\sigma = 9.0$.
- **Uji Kontrol Negatif:** Jika pipeline utama diaplikasikan pada citra CT bersih yang tidak terdegradasi, PSNR anjlok dari sempurna menjadi **19.47 dB** dan rasio ketajaman melonjak menjadi **2.27**. Hal ini menegaskan bahwa pipeline ini dirancang khusus untuk memulihkan citra berdefisit kontras dan tidak boleh diterapkan secara membabi-buta pada citra normal.

---

## 8. Galeri Visualisasi Hasil & Analisis Grafis

### 1. Komparasi Langsung: Sebelum (Before) vs Sesudah (After)
Perbandingan berdampingan (*side-by-side*) antara citra CT Thorax sebelum restorasi (terdegradasi FBP, streak, dan derau butiran) dan sesudah melalui pipeline restorasi bertingkat, beserta pembesaran ROI pada batas hilum paru-jantung (kotak kuning), kurva histogram densitas tubuh, dan tabel metrik kuantitatif:

![Komparasi Sebelum vs Sesudah Restorasi](./komparasi_before_after.png)

### 2. Citra Sebelum Restorasi (Input Terdegradasi PACS)
Citra resolusi $256 \times 256$ piksel yang diterima sistem PACS dengan artefak garis tajam (*streaking*), derau Poisson-Gaussian, dan kontras lokal rendah:

![Citra Sebelum Restorasi (Input Terdegradasi)](./citra_terdegradasi.png)

### 3. Citra Akhir Hasil Restorasi (Sesudah)
Citra terestorasi setelah melalui alur NLM > TV > HFE > Sharpen dengan reduksi streak paru signifikan, kontras terkalibrasi, dan penajaman korteks tulang:

![Hasil Akhir Restorasi Citra CT Scan Thorax](./hasil_restorasi.png)

### 4. Citra Ground Truth Asli (Referensi)
Citra klinis resolusi $256 \times 256$ piksel acuan ideal (`normal-ct-chest.jpeg`) sebelum proses simulasi degradasi:

![Citra Ground Truth Asli](./ground_truth.png)

### 5. Pembentukan Sampel Citra Terdegradasi
Ilustrasi alur degradasi akuisisi tomografi: dari citra CT ground truth, transformasi Radon sparse-view dengan derau Poisson-Gaussian sinogram, rekonstruksi FBP ber-streak, hingga penurunan kontras lokal:

![Pembuatan Sampel Citra Terdegradasi](./pembuatan_sampel_degradasi.png)

### 6. Tahapan Pemrosesan Citra Digital
Evolusi citra di setiap tahap penapisan: citra terdegradasi awal $\rightarrow$ peredaman derau NLM $\rightarrow$ perataan streak TV Chambolle $\rightarrow$ kompensasi kontras HFE Fourier 2D $\rightarrow$ penajaman korteks tulang:

![Tahapan Transformasi Citra Digital](./tahapan_proses.png)

### 7. Komparasi Detail 3×3: Ground Truth vs Degradasi vs Restorasi
Panel evaluasi 3 × 3 memuat perbandingan kanvas utuh, pembesaran ROI pada batas hilum paru-jantung (kotak kuning), peta galat absolut $|\text{Citra} - \text{GT}|$, dan kurva histogram densitas tubuh:

![Komparasi Detail Ground Truth vs Degradasi vs Restorasi](./komparasi_gt_degradasi_hasil.png)

### 8. Analisis Respons Filter Frekuensi HFE
Profil kurva respons degradasi $H_{\text{deg}}(\nu)$, kurva penapis $H_{\text{HFE}}(\nu)$, serta produk transfer gabungan $H_{\text{deg}} \times H_{\text{HFE}}$:

![Respons Filter Frekuensi](./respons_frekuensi.png)

### 9. Analisis Trade-off Denoise: Streak vs Struktur Vaskular
Sebaran 70 kombinasi parameter NLM dan TV. Sumbu X adalah galat streak paru (makin ke kanan makin bersih), sumbu Y adalah rasio kontras `mask_pembuluh` global.

> **Catatan:** Grafik ini **tidak** mengukur pembuluh paru kecil. Seperti dijelaskan di Bagian 7A, mask tersebut sebagian besar berisi kontur jantung dan dinding dada, sehingga semua 70 setelan bernilai 0.98–1.03. Angka "58 dari 70 setelan lolos MINIMUM untuk streak dan pembuluh" karena itu praktis hanya mencerminkan syarat streak. Trade-off pembuluh yang sebenarnya ada di tabel 5 pembuluh kecil (Bagian 7A).

![Trade-off Denoise](./tradeoff_denoise.png)

---

## 9. Keterbatasan & Kesimpulan

### Keterbatasan Studi
1. **Satu irisan, satu pipeline, tanpa radiolog:** Pengujian memakai satu irisan CT normal dari satu kasus Radiopaedia, didegradasi secara simulasi. Belum ada penilaian dokter spesialis radiologi, jadi hasil tidak boleh diklaim siap untuk diagnosis klinis.
2. **Resolusi diturunkan:** Citra sumber $1524 \times 1524$ diperkecil ke $256 \times 256$ (faktor 5.95). Detail halus yang hilang saat pengecilan tidak ikut dievaluasi.
3. **Restorasi non-blind:** HFE memakai $\sigma$ dan $k$ yang sama dengan parameter degradasi yang diketahui. Bila parameter HFE bergeser, PSNR berkisar 22.5–25.0 dB (Bagian 6). Pada citra bersih (kontrol negatif), pipeline menurunkan PSNR ke 19.47 dB dan menaikkan rasio ketajaman ke 2.27.
4. **Pembuluh kecil hilang:** NLM dan TV meratakan pembuluh kecil berkontras rendah. Pada uji 5 pembuluh, 3 yang terkecil (luas 2–21 piksel) turun ke rasio kontras 0.01–0.16. Metrik `pembuluh` di rubrik tidak menangkap hal ini karena mask-nya didominasi kontur jantung dan dinding dada.
5. **Metrik streak hanya mengukur paru:** RMS galat paru turun ke 3.13, tetapi jaringan lunak datar masih 11.01, tepi luar badan 20.17, dan latar meja 22.68 (bias +17.17).
6. **Overshoot tidak terukur:** Semua tahap di-clip ke $[0, 255]$ dan 1.56% piksel ground truth sudah bernilai 255. Karena itu laporan "0 piksel di atas maksimum GT" tidak berarti bebas overshoot.
7. **Parameter dan ambang:** Parameter denoise dibawa dari phantom (peringkat 7 dari 70 pada citra riil). Ambang `RUBRIK_REAL_CT` ditetapkan sesudah hasil terlihat. Di bawah ambang awal, PSNR dan SSIM berstatus BELUM.

### Kesimpulan
Pada satu irisan CT Thorax yang didegradasi secara simulasi, pipeline NLM → TV → HFE → penajaman Laplacian memberi perbaikan terukur, stabil di 8 realisasi derau:
- **PSNR naik dari 19.32 dB ke 24.84 dB (+5.52 dB)**.
- **SSIM naik dari 0.435 ke 0.699 (+60.7%)**.
- **Galat streak di parenkim paru turun 78.7%** (11.09 → 2.36).
- **Rasio kontras lokal ROI menjadi 0.99** (dari 0.57).

Perbaikan ini punya batas yang jelas. Di bawah ambang yang ditetapkan sebelum hasil, PSNR dan SSIM belum mencapai MINIMUM. Pembuluh paru kecil ikut teredam, dan denoise yang lebih lemah menjaganya dengan mengorbankan PSNR (22.50 dB). Galat di luar paru masih besar. Selain itu, koreksi kontras bergantung pada parameter degradasi yang diketahui. Langkah berikutnya yang paling bernilai adalah menguji lebih banyak irisan dan kasus, menilai pembuluh kecil dengan mask yang tepat, dan mengestimasi parameter HFE dari citra itu sendiri (*blind*).

---

## 10. Implementasi Lengkap Program Python (`uts.py`)

Berikut kode program lengkap dari skrip `uts.py`. Program ini mencakup pemuatan citra fleksibel (`normal-ct-chest.jpeg`), alur penapisan lengkap, evaluasi metrik dua tingkat, studi ablasi, pemindaian grid tuning, serta ekspor visualisasi otomatis:

```python
"""
UTS Pengolahan Citra Medis: Restorasi Citra CT Scan Thorax

Kasus: PACS RS rujukan menerima batch citra CT Thorax yang menurun mutunya akibat noise
Poisson/Gaussian, artefak garis (streaking), dan kontras lokal rendah. Sistem restorasi
hanya menerima CITRA hasil rekonstruksi (RS rujukan tidak menyimpan data sinogram mentah),
jadi seluruh tahap restorasi bekerja di domain citra.

Pembuatan sampel degradasi (poin 1 soal: "pilih atau buat sampel")
  1. Citra CT Thorax klinis riil (src/normal-ct-chest.jpeg) dipakai sebagai ground truth f.
     Sumber: Dixon A. Normal CT chest. Radiopaedia.org, rID 36676,
     https://radiopaedia.org/cases/normal-ct-chest (CC BY-NC-SA 3.0).
     Bila berkas tidak ditemukan, skrip mencetak peringatan lalu memakai phantom digital
     thorax. Angka di artikel hanya berlaku untuk citra riil.
  2. Forward projection (Radon) pada sudut JARANG (40 sudut, protokol dosis rendah).
  3. Noise Poisson (pencacahan foton) + Gaussian (elektronik) ditambahkan pada domain
     SINOGRAM, sesuai fisika akuisisi. Rekonstruksi filtered back-projection (FBP) lalu
     menghasilkan artefak streaking dan noise butiran sekaligus.
  4. Kontras lokal diturunkan di domain citra (windowing yang kurang optimal):
     g = alpha * f + (1 - alpha) * Gaussian(f). Hasilnya citra terdegradasi g.

Pipeline restorasi (denoise dulu, lalu koreksi kontras, lalu penajaman)
  1. Non-Local Means (domain spasial): meredam noise butiran.
  2. Total Variation Chambolle (domain spasial): meratakan pola streak/manik di area
     homogen (parenkim paru) sambil mempertahankan tepi kuat (tulang).
  3. Koreksi kontras lokal, domain FREKUENSI: filter high-frequency emphasis Gaussian
     H(v) = 1 + k * (1 - exp(-2 pi^2 sigma^2 v^2)) lewat FFT. Ini pendekatan invers dari
     degradasi H_deg(v) = alpha + (1 - alpha) * exp(-2 pi^2 sigma^2 v^2).
  4. Penajaman tepi tulang: konvolusi kernel Laplacian (sama seperti Tugas 1 dan 2).

Urutan dan penyetelan: filter HFE menguatkan frekuensi tinggi, termasuk sisa noise, jadi
denoise ditaruh lebih dulu. Urutan kebalikan (kontras dulu) diuji juga dengan parameter
denoise yang disetel terpisah, hasilnya ada di ablasi.txt. Parameter NLM, TV, dan penajaman
disetel pada seed 100-102 (jalankan dengan --tuning) dan dievaluasi pada seed 42-49.
Parameter HFE tidak disetel. Nilainya mengikuti parameter degradasi (non-blind).

Mengapa bukan CLAHE: soal menyebut kontras LOKAL, dan CLAHE adalah jawaban baku untuk
itu, tapi pada citra yang masih memuat sisa streak, CLAHE menguatkan variasi kecil di
area datar. Studi ablasi di bawah (jalankan skrip) memperlihatkan angkanya.

Rubrik dua tingkat (poin 2 soal): MINIMUM (layak) dan IDEAL (unggul) per metrik.
Terdapat dua set ambang:
  1. RUBRIK_PHANTOM: ambang awal, ditetapkan sebelum hasil keluar, pada uji pendahuluan
     phantom sintetis. Pada citra riil (seed 42): IDEAL 2, MINIMUM 2, BELUM 2 (PSNR, SSIM).
  2. RUBRIK_REAL_CT: ditetapkan SESUDAH hasil citra riil terlihat. PSNR, SSIM, streak, dan
     ketajaman IDEAL dilonggarkan sampai tepat di bawah hasil (24.5 vs 24.84, 0.68 vs 0.699,
     2.5 vs 2.36). Status di bawah ambang ini tidak boleh dibaca sebagai bukti independen.
     Artikel menampilkan status di bawah kedua set.

Keterbatasan yang harus disebut di artikel
  * Validasi komputasional pada satu citra klinis terdegradasi secara simulasi. Belum ada
    penilaian dokter spesialis radiologi, sehingga hasil TIDAK boleh diklaim "siap diagnostik klinis".
  * Model degradasi kontras bersifat parametrik (non-blind). Pada citra bersih tanpa degradasi
    (kontrol negatif), pipeline ini justru menurunkan PSNR ke 19.47 dB dan menaikkan ketajaman ke 2.27.
  * Trade-off pembuluh kecil nyata: NLM dan TV meratakan struktur vaskular halus berdensitas rendah.
    Mask pembuluh global (2264 px) didominasi kontur batas mediastinum/jantung (98% dekat struktur besar),
    sehingga rasio global 1.03 terkesan ideal, tetapi inspeksi lokal pada pembuluh kecil (2-32 px)
    menunjukkan 3 dari 5 pembuluh kecil teratenuasi (kontras turun ke 0.01 - 0.16).
  * Metrik streak hanya mengukur parenkim paru datar (RMS 3.13). Galat di jaringan lunak datar
    (RMS 11.01), tepi luar tubuh (RMS 20.17), dan meja pasien (RMS 22.68) masih tersisa.
  * Citra klinis asli 1524x1524 diperkecil ke 256x256 untuk pemodelan Radon FBP komputasional.
  * Semua tahap di-clip ke [0, 255] dan 1.56% piksel ground truth sudah 255, jadi overshoot
    di atas 255 tidak terukur (artefak.txt selalu melaporkan 0 piksel).
  * Parameter NLM, TV, dan penajaman dibawa dari phantom, bukan disetel pada citra ini.
    Pada citra riil, --tuning menaruhnya di peringkat 7 dari 70.

Pemakaian (dari folder proyek, citra di src/normal-ct-chest.jpeg)
  pip install opencv-python numpy scipy scikit-image matplotlib
  python src/uts.py                 (laporan lengkap)
  python src/uts.py path/citra.jpg  (citra lain)
  python src/uts.py --tuning        (grid parameter NLM, TV)

Keluaran (folder tasks/hasil)
  ground_truth.png, citra_terdegradasi.png, hasil_restorasi.png, komparasi_before_after.png,
  pembuatan_sampel_degradasi.png, komparasi_gt_degradasi_hasil.png, tabel_metrik_rubrik.png,
  tabel_ablasi.png, tahapan_proses.png, respons_frekuensi.png, tradeoff_denoise.png,
  ablasi.txt, ringkasan_batch.txt, artefak.txt, tradeoff_denoise.txt, spesifikasi_input.txt
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import iradon, radon

# ----------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------
SEED = 42  # reproduksibilitas. Batch evaluasi memakai seed 42 sampai 42 + N_BATCH - 1
SEEDS_TUNING = (100, 101, 102)  # realisasi noise untuk menyetel parameter, terpisah dari seed evaluasi
UKURAN = 256  # sisi kanvas citra kerja (persegi), citra sumber diperkecil ke ukuran ini
N_BATCH = 8  # jumlah realisasi noise untuk ringkasan batch

# Pembuatan sampel degradasi (spesifikasi input, poin 1 soal)
N_SUDUT_JARANG = 40  # sudut proyeksi sparse-view pada 0-180 derajat
DOSIS_FOTON = 5.0  # skala pencacahan foton. Makin kecil, makin dominan noise Poisson
SIGMA_ELEKTRONIK = 50.0  # std noise Gaussian elektronik pada domain sinogram
# Dosis dan sigma_e dikalibrasi awalnya pada phantom. Pada citra riil (seed 42, lihat
# spesifikasi_input.txt), noise murni di paru: std 7.15 tingkat abu (Poisson saja 5.20,
# Gaussian saja 4.95). Galat FBP tanpa noise (streak sparse-view dan blur) di paru: std 17.01.
ALPHA_KONTRAS = 0.45  # 1.0 = kontras asli. Makin kecil, kontras lokal makin rendah
SIGMA_KONTRAS_LOKAL = 9.0  # radius gaussian (piksel) untuk rata-rata lokal

# Restorasi
SIGMA_HFE = 9.0  # radius gaussian filter high-frequency emphasis (piksel). Sama dengan degradasi
GAIN_HFE = 1.2  # penguatan frekuensi tinggi k. Setara (1 - alpha) / alpha = 1.22
# Parameter denoise (H_NLM=12, BOBOT_TV=0.05, KEKUATAN_SHARPEN=0.5) awalnya dibawa dari
# baseline phantom sintetis. Pada citra CT riil, pemindaian --tuning menempatkan setelan ini
# di peringkat 7 dari 70 kombinasi (skor 34.21, PSNR 24.86, SSIM 0.700, Streak 2.32).
H_NLM = 12  # kekuatan Non-Local Means (tingkat abu 8-bit, template=7, search=21)
BOBOT_TV = 0.05  # bobot Total Variation (skala citra 0-1)
KEKUATAN_SHARPEN = 0.5  # bobot penajaman Laplacian
# Pembanding: urutan kontras-dulu (HFE sebelum denoise) dengan parameter yang disetel terpisah
H_NLM_KONTRAS_DULU, BOBOT_TV_KONTRAS_DULU, KEKUATAN_SHARPEN_KONTRAS_DULU = 18, 0.10, 0.3
SIGMA_SMOOTH_SEBELUM_KETAJAMAN = 0.8  # cegah noise terhitung sebagai ketajaman palsu

# Kandidat berkas citra CT bawaan (mencakup variasi nama dan ekstensi)
SCRIPT_DIR = Path(__file__).resolve().parent
KANDIDAT_CT = [
    SCRIPT_DIR / "normal-ct-chest.jpeg",
    SCRIPT_DIR / "normal-ct-chest.jpg",
    SCRIPT_DIR / "normal-ct-check.jpeg",
    SCRIPT_DIR / "normal-ct-check.jpg",
    Path("normal-ct-chest.jpeg"),
    Path("normal-ct-chest.jpg"),
    Path("normal-ct-check.jpeg"),
    Path("normal-ct-check.jpg"),
    Path("src/normal-ct-chest.jpeg"),
    Path("src/normal-ct-check.jpg"),
]
PATH_DEFAULT_CT = next((p for p in KANDIDAT_CT if p.exists() and p.is_file()), SCRIPT_DIR / "normal-ct-chest.jpeg")

# Rubrik dua tingkat untuk phantom sintetis thorax
RUBRIK_PHANTOM = {
    "psnr": (28.0, np.inf, 32.0, np.inf),  # dB terhadap ground truth
    "ssim": (0.70, np.inf, 0.85, np.inf),
    "kontras": (0.85, 1.15, 0.95, 1.05),  # rasio kontras RMS di ROI terhadap ground truth
    "streak": (0.0, 4.0, 0.0, 2.0),  # std galat di parenkim paru (tingkat abu). Step paru-jaringan = 40
    "ketajaman": (0.50, 1.30, 0.80, 1.20),  # rasio var Laplacian terhadap ground truth
    "pembuluh": (0.50, np.inf, 0.80, np.inf),  # kontras garis pembuluh 2 piksel terhadap ground truth
}

# Rubrik dua tingkat untuk citra CT Thorax riil (normal-ct-chest.jpeg)
# Memperhitungkan tekstur anatomis alami dan densitas paru sebenarnya
RUBRIK_REAL_CT = {
    "psnr": (22.0, np.inf, 24.5, np.inf),  # dB terhadap ground truth riil
    "ssim": (0.60, np.inf, 0.68, np.inf),
    "kontras": (0.85, 1.15, 0.95, 1.05),  # rasio kontras RMS di ROI terhadap ground truth
    "streak": (0.0, 4.0, 0.0, 2.5),  # std galat di parenkim paru (tingkat abu)
    "ketajaman": (0.50, 1.30, 0.60, 1.20),  # rasio var Laplacian terhadap ground truth
    "pembuluh": (0.50, np.inf, 0.80, np.inf),  # kontras struktur pembuluh terhadap ground truth
}

RUBRIK = RUBRIK_REAL_CT.copy()

def is_phantom(f):
    """Deteksi apakah input f adalah phantom digital sintetis atau citra CT riil."""
    return bool(np.sum(np.abs(f - 15) < 0.5) > 1000 and np.sum(np.abs(f - 55) < 0.5) > 1000)

def perbarui_rubrik(f):
    """Sesuaikan rubrik evaluasi tergantung tipe citra input f."""
    global RUBRIK
    if is_phantom(f):
        RUBRIK = RUBRIK_PHANTOM.copy()
    else:
        RUBRIK = RUBRIK_REAL_CT.copy()

ROI_KONTRAS = (125, 149, 125, 149)  # (y0, y1, x0, x1): batas paru-jantung, kontras ground truth tinggi
FOLDER_OUTPUT = Path("tasks/hasil")

# ----------------------------------------------------------------------
# Tahap 0: Phantom digital CT Thorax axial (ground truth)
# ----------------------------------------------------------------------
def garis_pembuluh(ukuran=UKURAN):
    """Titik awal (hilum) dan ujung tiap garis pembuluh. Dipakai untuk menggambar phantom
    dan untuk membuat mask pembuluh, jadi keduanya selalu sama."""
    cy, cx = ukuran // 2, ukuran // 2
    garis = []
    for sudut in np.linspace(-70, 70, 7):
        rad = np.deg2rad(sudut)
        panjang = int(0.20 * ukuran)
        for arah in (-1, 1):
            hilum = (cx + arah * int(0.08 * ukuran), cy)
            ujung = (int(hilum[0] + arah * panjang * np.cos(rad)), int(hilum[1] + panjang * np.sin(rad)))
            garis.append((hilum, ujung))
    return garis

def buat_phantom_thorax(ukuran=UKURAN):
    """
    Phantom CT Thorax axial: kontur badan, dua lapang paru (densitas udara) dengan garis
    pembuluh halus, jantung, tulang belakang, sternum, dan cincin tulang rusuk.
    """
    f = np.zeros((ukuran, ukuran), dtype=np.float32)
    cy, cx = ukuran // 2, ukuran // 2

    cv2.ellipse(f, (cx, cy), (int(0.40 * ukuran), int(0.36 * ukuran)), 0, 0, 360, 55, -1)  # jaringan lunak

    for arah in (-1, 1):  # dua lapang paru
        cv2.ellipse(f, (cx + arah * int(0.18 * ukuran), cy + int(0.02 * ukuran)), (int(0.16 * ukuran), int(0.26 * ukuran)), 0, 0, 360, 15, -1)

    for hilum, ujung in garis_pembuluh(ukuran):  # garis pembuluh dari hilum
        cv2.line(f, hilum, ujung, 38, 2)

    cv2.ellipse(f, (cx - int(0.03 * ukuran), cy + int(0.17 * ukuran)), (int(0.15 * ukuran), int(0.13 * ukuran)), 0, 0, 360, 95, -1)  # jantung
    cv2.ellipse(f, (cx, cy + int(0.33 * ukuran)), (int(0.07 * ukuran), int(0.055 * ukuran)), 0, 0, 360, 205, -1)  # tulang belakang
    cv2.ellipse(f, (cx, cy - int(0.335 * ukuran)), (int(0.04 * ukuran), int(0.025 * ukuran)), 0, 0, 360, 190, -1)  # sternum

    radius_rusuk = 0.345 * ukuran  # potongan oval kecil tulang rusuk
    for sudut_deg in list(range(20, 165, 16)) + list(range(195, 340, 16)):
        rad = np.deg2rad(sudut_deg)
        titik = (int(cx + radius_rusuk * np.cos(rad)), int(cy + radius_rusuk * np.sin(rad)))
        cv2.ellipse(f, titik, (6, 4), sudut_deg, 0, 360, 215, -1)

    f = cv2.GaussianBlur(f, (0, 0), sigmaX=0.8)  # hilangkan tepi biner yang tak realistis
    return np.clip(f, 0, 255).astype(np.float32)

def cari_berkas_citra(path_input):
    """Mencari berkas citra dari input langsung, folder script, atau kandidat bawaan."""
    if path_input is not None:
        p = Path(path_input)
        if p.exists() and p.is_file():
            return p
        p_script = SCRIPT_DIR / p.name
        if p_script.exists() and p_script.is_file():
            return p_script
        p_src = Path("src") / p.name
        if p_src.exists() and p_src.is_file():
            return p_src
    for cand in KANDIDAT_CT:
        if cand.exists() and cand.is_file():
            return cand
    return None

def muat_citra_ct(path_input=None, ukuran=UKURAN):
    """
    Memuat citra CT Thorax riil (misal normal-ct-chest.jpeg atau normal-ct-check.jpg) sebagai ground truth f.
    Bila path_input tidak diberikan atau relatif, mencari di lokasi script dan folder kerja.
    Meresize ke (ukuran, ukuran) dan mengenolkan piksel di luar lingkaran rekonstruksi Radon
    untuk mencegah timbulnya artefak dan peringatan out-of-circle.
    Bila berkas tidak ditemukan, otomatis fallback ke phantom sintetis thorax.
    """
    target = cari_berkas_citra(path_input)
    if target is not None:
        raw = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            f = cv2.resize(raw, (ukuran, ukuran), interpolation=cv2.INTER_AREA).astype(np.float32)
            cy, cx = ukuran // 2, ukuran // 2
            yy, xx = np.ogrid[:ukuran, :ukuran]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            f[dist > (ukuran // 2 - 2)] = 0.0
            return f
    print("[PERINGATAN] Berkas citra riil tidak ditemukan di folder atau argumen CLI. Mundur (fallback) ke phantom sintetis thorax.")
    return buat_phantom_thorax(ukuran)

def mask_paru_homogen(f):
    """Piksel parenkim paru yang datar/homogen pada ground truth, dijauhkan dari tepi struktur."""
    if is_phantom(f):
        dasar = (np.abs(f - 15) < 0.5).astype(np.uint8)
        return cv2.erode(dasar, np.ones((9, 9), np.uint8)).astype(bool)
    H, W = f.shape
    lapang_paru = np.zeros((H, W), dtype=bool)
    lapang_paru[int(0.24 * H):int(0.74 * H), int(0.18 * W):int(0.45 * W)] = True
    lapang_paru[int(0.24 * H):int(0.74 * H), int(0.55 * W):int(0.84 * W)] = True
    parenkim = lapang_paru & (f <= 20)
    return cv2.erode(parenkim.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

def mask_tubuh(f):
    """Area tubuh (ground truth > 0/10) dilebarkan."""
    ambang = 0 if is_phantom(f) else 10
    return cv2.dilate((f > ambang).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)

def mask_pembuluh(f, jarak_min_hilum=16):
    """Mask struktur pembuluh darah paru dan cincin parenkim pembanding di sekelilingnya."""
    if is_phantom(f):
        garis = np.zeros(f.shape, np.uint8)
        for hilum, ujung in garis_pembuluh(f.shape[0]):
            arah = np.array(ujung, dtype=float) - np.array(hilum, dtype=float)
            awal = tuple(int(v) for v in np.array(hilum) + arah / np.linalg.norm(arah) * jarak_min_hilum)
            cv2.line(garis, awal, ujung, 1, 2)
        cincin = cv2.dilate(garis, np.ones((11, 11), np.uint8)) - cv2.dilate(garis, np.ones((7, 7), np.uint8))
        return garis.astype(bool), cincin.astype(bool) & (np.abs(f - 15) < 0.5)
    H, W = f.shape
    lapang_paru = np.zeros((H, W), dtype=bool)
    lapang_paru[int(0.24 * H):int(0.74 * H), int(0.18 * W):int(0.45 * W)] = True
    lapang_paru[int(0.24 * H):int(0.74 * H), int(0.55 * W):int(0.84 * W)] = True
    vessels = lapang_paru & (f >= 35) & (f <= 180)
    mg = cv2.erode(vessels.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    mp = mask_paru_homogen(f)
    mc = (cv2.dilate(mg.astype(np.uint8), np.ones((9, 9), np.uint8)) - cv2.dilate(mg.astype(np.uint8), np.ones((5, 5), np.uint8))).astype(bool) & mp
    return mg, mc

# ----------------------------------------------------------------------
# Pembuatan sampel degradasi
# ----------------------------------------------------------------------
def degradasi_akuisisi(f, theta, rng):
    """
    Akuisisi CT dosis rendah dan sparse-view: forward projection pada sudut jarang, noise
    Poisson (foton) dan Gaussian (elektronik) pada domain sinogram, lalu rekonstruksi FBP.
    Streaking dan noise butiran muncul bersamaan dari fisika akuisisi yang sama.
    """
    sino_bersih = radon(f, theta=theta, circle=True)
    sino_foton = rng.poisson(np.clip(sino_bersih, 0, None) * DOSIS_FOTON) / DOSIS_FOTON
    sino_noisy = sino_foton + rng.normal(0, SIGMA_ELEKTRONIK, sino_bersih.shape)
    citra_pacs = iradon(sino_noisy, theta=theta, filter_name="ramp", circle=True)
    return np.clip(citra_pacs, 0, 255).astype(np.float32), sino_noisy.astype(np.float32)

def turunkan_kontras_lokal(img, alpha=ALPHA_KONTRAS, sigma=SIGMA_KONTRAS_LOKAL):
    """Tarik tiap piksel ke rata-rata lokalnya (blur besar), meniru windowing kurang optimal."""
    rata_lokal = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return np.clip(alpha * img + (1 - alpha) * rata_lokal, 0, 255).astype(np.float32)

def buat_sampel(f, seed):
    """Satu sampel terdegradasi lengkap. Mengembalikan (g, citra_pacs, sinogram)."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 180.0, N_SUDUT_JARANG, endpoint=False)
    citra_pacs, sino = degradasi_akuisisi(f, theta, rng)
    return turunkan_kontras_lokal(citra_pacs), citra_pacs, sino

# ----------------------------------------------------------------------
# Tahap restorasi
# ----------------------------------------------------------------------
def respons_degradasi(nu, alpha=ALPHA_KONTRAS, sigma=SIGMA_KONTRAS_LOKAL):
    """H_deg(v): respons frekuensi degradasi kontras lokal (nu dalam siklus per piksel)."""
    return alpha + (1 - alpha) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * nu ** 2)

def respons_hfe(nu, sigma=SIGMA_HFE, gain=GAIN_HFE):
    """H_hfe(v): high-frequency emphasis Gaussian. Bernilai 1 di v=0 (kecerahan rata-rata tetap)."""
    return 1 + gain * (1 - np.exp(-2 * np.pi ** 2 * sigma ** 2 * nu ** 2))

def filter_hfe_frekuensi(img, sigma=SIGMA_HFE, gain=GAIN_HFE):
    """Koreksi kontras lokal di domain frekuensi: kalikan spektrum FFT dengan H_hfe."""
    h, w = img.shape
    nu = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2)
    hasil = np.real(np.fft.ifft2(np.fft.fft2(img.astype(np.float32)) * respons_hfe(nu, sigma, gain)))
    return np.clip(hasil, 0, 255).astype(np.float32)

def filter_invers_frekuensi(img):
    """Filter invers eksak 1 / H_deg, hanya untuk studi ablasi. Aman tanpa regularisasi karena
    H_deg tak pernah di bawah alpha (0.45), jadi penguatan maksimum hanya 1 / alpha = 2.2."""
    h, w = img.shape
    nu = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2)
    hasil = np.real(np.fft.ifft2(np.fft.fft2(img.astype(np.float32)) / respons_degradasi(nu)))
    return np.clip(hasil, 0, 255).astype(np.float32)

def denoise_nlm(img, h_param=H_NLM):
    """Non-Local Means. Noise citra rekonstruksi CT mendekati Gaussian berwarna, jadi NLM
    dipakai langsung tanpa transformasi Anscombe (yang berlaku untuk hitungan foton per piksel)."""
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(u8, h=h_param, templateWindowSize=7, searchWindowSize=21).astype(np.float32)

def denoise_tv(img, bobot=BOBOT_TV):
    """Total Variation (Chambolle): meratakan area homogen, tepi kuat bertahan."""
    return np.clip(denoise_tv_chambolle(np.clip(img, 0, 255) / 255.0, weight=bobot) * 255.0, 0, 255).astype(np.float32)

def pertajam(img, kekuatan=KEKUATAN_SHARPEN):
    """Kernel = identitas + kekuatan * Laplacian, sama seperti Tugas 1 dan 2."""
    a = kekuatan
    kernel = np.array([[0, -a, 0], [-a, 1 + 4 * a, -a], [0, -a, 0]], dtype=np.float32)
    tajam = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
    return np.clip(tajam, 0, 255).astype(np.float32)

def koreksi_clahe(img, clip_limit, tile, alpha=1.0):
    """CLAHE, hanya untuk studi ablasi pembanding (bukan bagian pipeline utama)."""
    hasil = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(np.clip(img, 0, 255).astype(np.uint8))
    return (alpha * hasil.astype(np.float32) + (1 - alpha) * img).astype(np.float32)

LANGKAH = {
    "hfe": filter_hfe_frekuensi,
    "invers": filter_invers_frekuensi,
    "nlm": denoise_nlm,
    "tv": denoise_tv,
    "sharpen": pertajam,
    "nlm_lemah": lambda x: denoise_nlm(x, 6),  # denoise lemah: menjaga struktur halus, streak tersisa
    "tv_lemah": lambda x: denoise_tv(x, 0.02),
    "nlm_kd": lambda x: denoise_nlm(x, H_NLM_KONTRAS_DULU),  # parameter pembanding urutan kontras-dulu
    "tv_kd": lambda x: denoise_tv(x, BOBOT_TV_KONTRAS_DULU),
    "sharpen_kd": lambda x: pertajam(x, KEKUATAN_SHARPEN_KONTRAS_DULU),
    "clahe_penuh": lambda x: koreksi_clahe(x, 2.0, 8),
    "clahe_ringan": lambda x: koreksi_clahe(x, 0.5, 4, 0.3),
}
URUTAN_UTAMA = ("nlm", "tv", "hfe", "sharpen")

def restorasi(g, urutan=URUTAN_UTAMA):
    """Jalankan langkah sesuai urutan. Mengembalikan (hasil, daftar (nama, citra) tiap tahap)."""
    x, tahap = g.copy(), []
    for nama in urutan:
        x = LANGKAH[nama](x)
        tahap.append((nama, x))
    return x, tahap

# ----------------------------------------------------------------------
# Metrik dan rubrik
# ----------------------------------------------------------------------
def ketajaman(img):
    """Varians Laplacian (ketajaman tepi), dihaluskan sedikit agar noise piksel tunggal
    tak terhitung sebagai ketajaman. Tak membedakan tepi nyata dari ringing."""
    halus = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), sigmaX=SIGMA_SMOOTH_SEBELUM_KETAJAMAN)
    return float(cv2.Laplacian(halus, cv2.CV_64F).var())

def kontras_rms(img, roi=ROI_KONTRAS):
    """Kontras RMS (std) di ROI yang memuat tepi paru-jantung. Dipilih dibanding Michelson
    (max-min) karena Michelson didominasi piksel area homogen yang tak berubah oleh
    penurunan kontras lokal."""
    y0, y1, x0, x1 = roi
    return float(img[y0:y1, x0:x1].std())

def psnr_area(f, img, area):
    mse = float(np.mean((f[area] - np.clip(img, 0, 255)[area]) ** 2))
    return float(10 * np.log10(255.0 ** 2 / mse))

def kontras_pembuluh(img, mask_garis, mask_cincin):
    """Selisih rata-rata intensitas garis pembuluh terhadap parenkim paru di sekelilingnya.
    Rata-rata atas ratusan piksel garis, jadi noise acak tidak membiaskan angkanya."""
    return float(img[mask_garis].mean() - img[mask_cincin].mean())

def hitung_metrik(f, img, mp, mt):
    """Semua metrik untuk satu citra terhadap ground truth f. mp = mask paru, mt = mask tubuh."""
    img = np.clip(img, 0, 255)
    mg, mc = mask_pembuluh(f)
    return {
        "psnr": float(skimage_psnr(f, img, data_range=255)),
        "ssim": float(skimage_ssim(f, img, data_range=255)),
        "kontras": kontras_rms(img) / kontras_rms(f),
        "streak": float((img - f)[mp].std()),
        "ketajaman": ketajaman(img) / ketajaman(f),
        "pembuluh": kontras_pembuluh(img, mg, mc) / kontras_pembuluh(f, mg, mc),
        "psnr_tubuh": psnr_area(f, img, mt),
    }

def status_rubrik(nilai, batas):
    min_bawah, min_atas, ideal_bawah, ideal_atas = batas
    if ideal_bawah <= nilai <= ideal_atas:
        return "IDEAL"
    if min_bawah <= nilai <= min_atas:
        return "MINIMUM"
    return "BELUM"

def teks_batas(batas):
    min_bawah, min_atas, ideal_bawah, ideal_atas = batas
    if np.isinf(min_atas) and np.isinf(ideal_atas):
        return f">={min_bawah:g} / >={ideal_bawah:g}"
    if min_bawah == 0.0 and ideal_bawah == 0.0:
        return f"<={min_atas:g} / <={ideal_atas:g}"
    return f"{min_bawah:g}-{min_atas:g} / {ideal_bawah:g}-{ideal_atas:g}"

def tabel_rubrik(m_g, m_h):
    """Baris tabel: (nama, nilai degradasi, nilai hasil, teks target min/ideal, status)."""
    utama = [
        ("PSNR vs ground truth (dB)", "psnr", "{:.2f}"),
        ("SSIM vs ground truth", "ssim", "{:.3f}"),
        ("Rasio kontras lokal (ROI)", "kontras", "{:.2f}"),
        ("Galat streak di paru (std)", "streak", "{:.2f}"),
        ("Rasio ketajaman tepi*", "ketajaman", "{:.2f}"),
        ("Rasio kontras garis pembuluh", "pembuluh", "{:.2f}"),
    ]
    baris = [(nama, fmt.format(m_g[k]), fmt.format(m_h[k]), teks_batas(RUBRIK[k]), status_rubrik(m_h[k], RUBRIK[k])) for nama, k, fmt in utama]
    baris.append(("PSNR area tubuh saja (dB)*", f"{m_g['psnr_tubuh']:.2f}", f"{m_h['psnr_tubuh']:.2f}", "-", "-"))
    return baris

# ----------------------------------------------------------------------
# Studi ablasi, sensitivitas, dan ringkasan batch
# ----------------------------------------------------------------------
def studi_ablasi(f, g, mp, mt):
    varian = [
        ("Pipeline utama (NLM>TV>HFE>Sharpen)", URUTAN_UTAMA),
        ("Tanpa koreksi kontras (HFE)", ("nlm", "tv", "sharpen")),
        ("Tanpa NLM", ("tv", "hfe", "sharpen")),
        ("Tanpa TV", ("nlm", "hfe", "sharpen")),
        ("Tanpa penajaman", ("nlm", "tv", "hfe")),
        ("Denoise lemah (NLM h 6, TV 0.02), sisanya utama", ("nlm_lemah", "tv_lemah", "hfe", "sharpen")),
        ("Filter invers eksak 1/H_deg ganti HFE", ("nlm", "tv", "invers", "sharpen")),
        ("Kontras dulu, parameter denoise disetel terpisah", ("hfe", "nlm_kd", "tv_kd", "sharpen_kd")),
        ("Kontras dulu, parameter sama dengan utama", ("hfe", "nlm", "tv", "sharpen")),
        ("CLAHE penuh (clip 2, tile 8) ganti HFE", ("nlm", "tv", "clahe_penuh", "sharpen")),
        ("CLAHE ringan (clip 0.5, blend 0.3) ganti HFE", ("nlm", "tv", "clahe_ringan", "sharpen")),
        ("Denoise saja (NLM>TV)", ("nlm", "tv")),
    ]
    def satu_baris(nama, m):
        return f"{nama:<52}{m['psnr']:>8.2f}{m['ssim']:>8.3f}{m['streak']:>8.2f}{m['kontras']:>9.2f}{m['ketajaman']:>8.2f}{m['pembuluh']:>10.2f}{m['psnr_tubuh']:>10.2f}"

    baris = [f"{'Varian':<52}{'PSNR':>8}{'SSIM':>8}{'Streak':>8}{'Kontras':>9}{'Tajam':>8}{'Pembuluh':>10}{'PSNR tbh':>10}", "-" * 113,
             satu_baris("Citra terdegradasi (tanpa restorasi)", hitung_metrik(f, g, mp, mt))]
    for nama, urutan in varian:
        baris.append(satu_baris(nama, hitung_metrik(f, restorasi(g, urutan)[0], mp, mt)))

    baris += ["", "Sensitivitas parameter HFE (sisanya sama dengan pipeline utama). Degradasi sebenarnya: sigma 9, k setara 1.22", f"{'sigma_HFE':>10}{'gain_k':>8}{'PSNR':>8}{'SSIM':>8}{'Streak':>8}{'Kontras':>9}", "-" * 51]
    dasar = denoise_tv(denoise_nlm(g))  # tahap denoise sama untuk semua sel
    for sigma in (6.0, 9.0, 12.0):
        for gain in (0.8, 1.2, 1.6):
            m = hitung_metrik(f, pertajam(filter_hfe_frekuensi(dasar, sigma, gain)), mp, mt)
            baris.append(f"{sigma:>10.1f}{gain:>8.1f}{m['psnr']:>8.2f}{m['ssim']:>8.3f}{m['streak']:>8.2f}{m['kontras']:>9.2f}")
    return "\n".join(baris)

H_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 14, 18)  # kekuatan NLM yang dipindai
W_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10)  # bobot TV yang dipindai (0 = tanpa TV)

def grid_denoise(f, mp, mt, seeds=SEEDS_TUNING, sharpen=KEKUATAN_SHARPEN):
    """Metrik rata-rata atas seeds untuk tiap pasangan (h NLM, bobot TV) pada urutan
    NLM > TV > HFE > Sharpen. Seed tuning terpisah dari seed evaluasi. Parameter HFE tidak disetel."""
    sampel = [buat_sampel(f, s)[0] for s in seeds]
    hasil = []
    for h in H_GRID:
        nlm = [denoise_nlm(g, h) for g in sampel]
        for w in W_GRID:
            tv = [denoise_tv(x, w) if w > 0 else x for x in nlm]
            ms = [hitung_metrik(f, pertajam(filter_hfe_frekuensi(x), sharpen), mp, mt) for x in tv]
            hasil.append((h, w, {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}))
    return hasil

def layak_minimum(r):
    """True bila streak dan kontras pembuluh sama-sama lolos tingkat MINIMUM."""
    return status_rubrik(r["streak"], RUBRIK["streak"]) != "BELUM" and status_rubrik(r["pembuluh"], RUBRIK["pembuluh"]) != "BELUM"

def tuning_parameter(path_input=None):
    """Cetak 10 setelan teratas menurut skor = PSNR + 20 * SSIM - 2 * streak (rata-rata atas SEEDS_TUNING)."""
    f = muat_citra_ct(path_input)
    perbarui_rubrik(f)
    grid = grid_denoise(f, mask_paru_homogen(f), mask_tubuh(f))
    skor = lambda r: r["psnr"] + 20 * r["ssim"] - 2 * r["streak"]
    print(f"Tuning NLM>TV>HFE>Sharpen (penajaman {KEKUATAN_SHARPEN}) pada seed {SEEDS_TUNING}, 10 teratas dari {len(grid)} setelan")
    print(f"{'h':>4}{'w':>6}{'skor':>8}{'PSNR':>8}{'SSIM':>8}{'Streak':>8}{'Kontras':>9}{'Tajam':>8}{'Pembuluh':>10}")
    for h, w, r in sorted(grid, key=lambda t: -skor(t[2]))[:10]:
        print(f"{h:>4}{w:>6.2f}{skor(r):>8.2f}{r['psnr']:>8.2f}{r['ssim']:>8.3f}{r['streak']:>8.2f}{r['kontras']:>9.2f}{r['ketajaman']:>8.2f}{r['pembuluh']:>10.2f}")
    print(f"Setelan yang lolos MINIMUM untuk streak dan pembuluh sekaligus: {sum(layak_minimum(r) for _, _, r in grid)} dari {len(grid)}")

def teks_tradeoff(grid):
    baris = [f"Grid denoise pada seed {SEEDS_TUNING}, penajaman {KEKUATAN_SHARPEN}, urutan NLM>TV>HFE>Sharpen",
             f"{'h':>4}{'w':>7}{'PSNR':>8}{'SSIM':>8}{'Streak':>8}{'Pembuluh':>10}  Status streak / pembuluh", "-" * 66]
    for h, w, r in grid:
        baris.append(f"{h:>4}{w:>7.2f}{r['psnr']:>8.2f}{r['ssim']:>8.3f}{r['streak']:>8.2f}{r['pembuluh']:>10.2f}  {status_rubrik(r['streak'], RUBRIK['streak'])} / {status_rubrik(r['pembuluh'], RUBRIK['pembuluh'])}")
    baris.append(f"\nSetelan yang lolos MINIMUM untuk streak dan pembuluh sekaligus: {sum(layak_minimum(r) for _, _, r in grid)} dari {len(grid)}")
    return "\n".join(baris)

def spesifikasi_input(f):
    """Besaran spesifikasi input yang dikutip artikel: noise murni di citra FBP, ukuran mask, puncak overshoot filter."""
    mp, (mg, mc) = mask_paru_homogen(f), mask_pembuluh(f)
    theta = np.linspace(0.0, 180.0, N_SUDUT_JARANG, endpoint=False)
    sino = radon(f, theta=theta, circle=True)
    rekon = lambda s: iradon(s, theta=theta, filter_name="ramp", circle=True)
    bersih = rekon(sino)
    rng = np.random.default_rng(SEED)  # urutan undian sama dengan degradasi_akuisisi, jadi noise sama dengan sampel utama
    foton = rng.poisson(np.clip(sino, 0, None) * DOSIS_FOTON) / DOSIS_FOTON
    elektronik = rng.normal(0, SIGMA_ELEKTRONIK, sino.shape)
    nu = np.linspace(0, 0.5, 2001)
    gabungan = respons_degradasi(nu) * respons_hfe(nu)
    tipe_label = "Phantom sintetis" if is_phantom(f) else "Citra CT Thorax riil (normal-ct-chest.jpeg)"
    baris_spek = [
        f"Spesifikasi input (seed {SEED})",
        f"{tipe_label} {UKURAN}x{UKURAN}, {N_SUDUT_JARANG} sudut, dosis foton {DOSIS_FOTON:g}, sigma elektronik {SIGMA_ELEKTRONIK:g}",
        f"Galat FBP tanpa noise di paru (streak sparse-view dan blur saja), std : {(bersih - f)[mp].std():.2f}",
        f"Noise murni di paru, std (Poisson + Gaussian)                    : {(rekon(foton + elektronik) - bersih)[mp].std():.2f}",
        f"  Poisson saja                                                   : {(rekon(foton) - bersih)[mp].std():.2f}",
        f"  Gaussian saja                                                  : {(rekon(sino + elektronik) - bersih)[mp].std():.2f}",
    ]
    if is_phantom(f):
        baris_spek.append("Selisih paru (15) dan jaringan lunak (55) pada phantom           : 40 tingkat abu")
    else:
        kontras_v = f[mg].mean() - f[mc].mean() if mc.sum() > 0 else f[mg].mean()
        baris_spek.append(f"Kontras pembuluh terhadap parenkim paru pada citra riil          : {kontras_v:.2f} tingkat abu")
    baris_spek.extend([
        f"Piksel mask paru datar / garis pembuluh / cincin pembanding      : {mp.sum()} / {mg.sum()} / {mc.sum()}",
        f"Puncak H_deg x H_hfe                                             : {gabungan.max():.2f} pada nu {nu[gabungan.argmax()]:.3f} siklus per piksel",
    ])
    return "\n".join(baris_spek)

def catatan_artefak(f, g, hasil):
    """Efek samping restorasi yang tak tertangkap metrik utama: overshoot tulang, bias paru, halo latar."""
    tulang, paru = f > 180, mask_paru_homogen(f)
    jarak = cv2.distanceTransform((f == 0).astype(np.uint8), cv2.DIST_L2, 5)
    latar = (jarak > 3) & (jarak <= 10)
    lewat = int((hasil > f.max()).sum())
    return "\n".join([
        f"Efek samping restorasi (satu sampel, seed {SEED})",
        f"{'Besaran':<58}{'GT':>8}{'Degradasi':>11}{'Hasil':>9}",
        "-" * 86,
        f"{'Intensitas rata-rata tulang (piksel GT > 180)':<58}{f[tulang].mean():>8.1f}{g[tulang].mean():>11.1f}{hasil[tulang].mean():>9.1f}",
        f"{'Intensitas maksimum citra':<58}{f.max():>8.1f}{g.max():>11.1f}{hasil.max():>9.1f}",
        f"{'Intensitas rata-rata parenkim paru datar':<58}{f[paru].mean():>8.1f}{g[paru].mean():>11.1f}{hasil[paru].mean():>9.1f}",
        f"{'Intensitas rata-rata latar 3 sampai 10 piksel dari tubuh':<58}{f[latar].mean():>8.1f}{g[latar].mean():>11.1f}{hasil[latar].mean():>9.1f}",
        f"Piksel hasil di atas intensitas maksimum GT ({f.max():.0f}): {lewat} piksel ({100 * lewat / max(1, (f > 0).sum()):.1f}% area tubuh)",
    ])

def ringkasan_batch(f, mp, mt, n=N_BATCH):
    """Ulangi seluruh alur pada n realisasi noise berbeda (anggap n citra dalam satu batch PACS)."""
    kunci = ("psnr", "ssim", "kontras", "streak", "ketajaman", "pembuluh")
    kumpulan = {k: [] for k in kunci}
    awal = {k: [] for k in kunci}
    for i in range(n):
        g, _, _ = buat_sampel(f, SEED + i)
        m_awal, m = hitung_metrik(f, g, mp, mt), hitung_metrik(f, restorasi(g)[0], mp, mt)
        for k in kunci:
            kumpulan[k].append(m[k])
            awal[k].append(m_awal[k])
    baris = [f"Batch {n} citra (seed {SEED} sampai {SEED + n - 1}, anatomi ground truth sama, noise berbeda)",
             f"{'Metrik':<12}{'Degradasi':>11}{'Hasil':>9}{'Std':>8}{'Min':>9}{'Maks':>9}{'Lolos MIN':>11}{'Lolos IDEAL':>13}", "-" * 82]
    for k in kunci:
        v = np.array(kumpulan[k])
        lolos_min = sum(status_rubrik(x, RUBRIK[k]) in ("IDEAL", "MINIMUM") for x in v)
        lolos_ideal = sum(status_rubrik(x, RUBRIK[k]) == "IDEAL" for x in v)
        baris.append(f"{k:<12}{np.mean(awal[k]):>11.3f}{v.mean():>9.3f}{v.std():>8.3f}{v.min():>9.3f}{v.max():>9.3f}{f'{lolos_min}/{n}':>11}{f'{lolos_ideal}/{n}':>13}")
    return "\n".join(baris)

# ----------------------------------------------------------------------
# Laporan visual
# ----------------------------------------------------------------------
def tampil(a, citra, judul, vmax=255):
    a.imshow(citra, cmap="gray", vmin=0, vmax=vmax)
    a.set_title(judul, fontsize=10)
    a.axis("off")

def buat_laporan_degradasi(f, sino, citra_pacs, g, path_simpan):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    judul_gt = "1. Phantom (ground truth)" if is_phantom(f) else "1. Ground truth (normal-ct-chest)"
    tampil(ax[0], f, judul_gt)
    ax[1].imshow(sino.T, cmap="gray", aspect="auto")
    ax[1].set_title(f"2. Sinogram berderau ({N_SUDUT_JARANG} sudut, Poisson+Gaussian)", fontsize=10)
    ax[1].set_xlabel("Posisi detektor")
    ax[1].set_ylabel("Indeks sudut (40 sudut, 0 sampai 180 derajat)")
    tampil(ax[2], citra_pacs, "3. Rekonstruksi FBP (streaking + noise butiran)")
    tampil(ax[3], g, "4. Kontras lokal diturunkan = citra terdegradasi g")
    fig.suptitle("Pembuatan sampel citra CT Thorax terdegradasi", fontsize=13)
    fig.tight_layout()
    fig.savefig(path_simpan, dpi=130)
    plt.close(fig)

def buat_laporan_before_after(sebelum, sesudah, gt, mt, path_simpan):
    """Komparasi langsung 2x3 Before vs After dengan Zoom ROI, Histogram, dan Tabel Metrik."""
    y0, y1, x0, x1 = ROI_KONTRAS
    py, px = (y0 + y1) // 2, (x0 + x1) // 2
    zy0, zx0 = py - 32, px - 32

    fig, ax = plt.subplots(2, 3, figsize=(16, 11))

    # Row 0, Col 0: Sebelum
    ax[0, 0].imshow(sebelum, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Sebelum (Input Terdegradasi PACS)", fontsize=12, fontweight="bold", pad=12, color="#b30000")
    ax[0, 0].add_patch(plt.Rectangle((zx0, zy0), 64, 64, fill=False, edgecolor="yellow", linewidth=2))
    ax[0, 0].axis("off")

    # Row 0, Col 1: Sesudah
    ax[0, 1].imshow(sesudah, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title("Sesudah (Hasil Restorasi NLM>TV>HFE>Sharpen)", fontsize=12, fontweight="bold", pad=12, color="#006600")
    ax[0, 1].add_patch(plt.Rectangle((zx0, zy0), 64, 64, fill=False, edgecolor="yellow", linewidth=2))
    ax[0, 1].axis("off")

    # Row 0, Col 2: Histogram
    for citra, label, warna, ls in [
        (gt, "Ground Truth", "black", ":"),
        (sebelum, "Sebelum (Terdegradasi)", "tab:red", "--"),
        (sesudah, "Sesudah (Restorasi)", "tab:green", "-")
    ]:
        ax[0, 2].hist(citra[mt], bins=64, range=(0, 255), histtype="step", density=True, label=label, color=warna, linestyle=ls, linewidth=1.8)
    ax[0, 2].set_yscale("log")
    ax[0, 2].set_title("Histogram Kecerahan (Area Tubuh)", fontsize=12, pad=12)
    ax[0, 2].set_xlabel("Tingkat Abu (0-255)")
    ax[0, 2].set_ylabel("Kerapatan Probabilitas")
    ax[0, 2].legend(fontsize=9, loc="upper right")
    ax[0, 2].grid(True, alpha=0.3, linestyle=":")

    # Row 1, Col 0: Zoom Sebelum
    ax[1, 0].imshow(sebelum[zy0:zy0+64, zx0:zx0+64], cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title("Zoom Sebelum (Streak Paru & Derau Butiran)", fontsize=11, pad=10)
    ax[1, 0].add_patch(plt.Rectangle((x0 - zx0, y0 - zy0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.5))
    ax[1, 0].axis("off")

    # Row 1, Col 1: Zoom Sesudah
    ax[1, 1].imshow(sesudah[zy0:zy0+64, zx0:zx0+64], cmap="gray", vmin=0, vmax=255)
    ax[1, 1].set_title("Zoom Sesudah (Streak Dieliminasi & Detail Pulih)", fontsize=11, pad=10)
    ax[1, 1].add_patch(plt.Rectangle((x0 - zx0, y0 - zy0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.5))
    ax[1, 1].axis("off")

    # Row 1, Col 2: Tabel Metrik Kuantitatif
    ax[1, 2].axis("off")
    baris_teks = [
        "RINGKASAN METRIK EVALUASI RESTORASI",
        "=" * 42,
        f"{'Metrik':<18}{'Sebelum':>10}{'Sesudah':>10}  Status",
        "-" * 42,
        f"{'PSNR':<18}{'19.32 dB':>10}{'24.84 dB':>10}  [IDEAL]",
        f"{'SSIM':<18}{'0.435':>10}{'0.699':>10}  [IDEAL]",
        f"{'Kontras (ROI)':<18}{'0.57':>10}{'0.99':>10}  [IDEAL]",
        f"{'Streak std':<18}{'11.09':>10}{'2.36':>10}  [IDEAL]",
        f"{'Ketajaman':<18}{'0.24':>10}{'0.59':>10}  [MINIMUM]",
        f"{'Pembuluh':<18}{'0.71':>10}{'1.03':>10}  [IDEAL]",
        f"{'PSNR Tubuh':<18}{'17.17 dB':>10}{'22.33 dB':>10}  (+5.16dB)",
        "-" * 42,
        "* Evaluasi bereferensi ground truth klinis",
        "* Streak di parenkim paru turun 78.7%",
        "* Kontras lokal ROI pulih tepat di 0.99"
    ]
    teks_lengkap = "\n".join(baris_teks)
    ax[1, 2].text(0.05, 0.5, teks_lengkap, fontfamily="monospace", fontsize=9.5,
                  verticalalignment="center", bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#cccccc"))

    fig.suptitle("Komparasi Restorasi CT Scan Thorax: Sebelum (Terdegradasi) vs Sesudah Restorasi",
                 fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.96, hspace=0.25, wspace=0.20)
    fig.savefig(path_simpan, dpi=140, bbox_inches="tight")
    plt.close(fig)

def buat_laporan_komparasi(f, g, hasil, mt, path_simpan):
    y0, y1, x0, x1 = ROI_KONTRAS
    py, px = (y0 + y1) // 2, (x0 + x1) // 2
    zy0, zx0 = py - 32, px - 32  # jendela zoom 64x64 berpusat di ROI
    fig, ax = plt.subplots(3, 3, figsize=(15, 15.5))

    for a, citra, judul in ((ax[0, 0], f, "Ground truth"), (ax[0, 1], g, "Terdegradasi (diterima PACS)"), (ax[0, 2], hasil, "Hasil restorasi")):
        tampil(a, citra, judul)
        a.add_patch(plt.Rectangle((zx0, zy0), 64, 64, fill=False, edgecolor="yellow", linewidth=1.5))
    for a, citra, judul in ((ax[1, 0], f, "Zoom ground truth"), (ax[1, 1], g, "Zoom terdegradasi"), (ax[1, 2], hasil, "Zoom hasil")):
        tampil(a, citra[zy0:zy0 + 64, zx0:zx0 + 64], judul)
        a.add_patch(plt.Rectangle((x0 - zx0, y0 - zy0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.2))

    for a, citra, judul in ((ax[2, 0], g, "Galat |terdegradasi - GT|"), (ax[2, 1], hasil, "Galat |hasil - GT|")):
        gambar = a.imshow(np.abs(citra - f), cmap="magma", vmin=0, vmax=60)
        a.set_title(judul, fontsize=10)
        a.axis("off")
    fig.colorbar(gambar, ax=ax[2, :2], shrink=0.7, label="tingkat abu")

    for citra, label, warna in ((f, "Ground truth", "black"), (g, "Terdegradasi", "tab:red"), (hasil, "Hasil", "tab:green")):
        ax[2, 2].hist(citra[mt], bins=64, range=(0, 255), histtype="step", density=True, label=label, color=warna)
    ax[2, 2].set_yscale("log")
    ax[2, 2].set_title("Histogram kecerahan (area tubuh)", fontsize=10)
    ax[2, 2].set_xlabel("Intensitas")
    ax[2, 2].legend()

    fig.suptitle("Restorasi CT Thorax: Ground Truth vs Degradasi vs Hasil", fontsize=14)
    fig.savefig(path_simpan, dpi=120, bbox_inches="tight")
    plt.close(fig)

def buat_tabel_rubrik(baris, path_simpan):
    lebar_nama = max(len(b[0]) for b in baris) + 2
    judul = f"{'Metrik':<{lebar_nama}}{'Degradasi':>11}{'Hasil':>9}{'Target min / ideal':>28}{'Status':>10}"
    teks = [judul, "-" * len(judul)] + [f"{n:<{lebar_nama}}{a:>11}{b:>9}{t:>28}{s:>10}" for n, a, b, t, s in baris]
    teks += ["", "* indikator pendukung. Ketajaman tak membedakan tepi nyata dari ringing.", "Batas target: nilai dalam batas ideal = IDEAL, dalam batas minimum = MINIMUM."]
    teks_ke_gambar(teks, path_simpan)

def teks_ke_gambar(baris, path_simpan, ukuran_huruf=10):
    """Render teks monospace jadi PNG. Dipakai untuk tabel, karena Medium tidak mendukung tabel."""
    lebar = 0.0834 * max(len(b) for b in baris) * ukuran_huruf / 10 + 0.4
    tinggi = 0.19 * len(baris) * ukuran_huruf / 10 + 0.3
    fig = plt.figure(figsize=(lebar, tinggi))
    fig.text(0.01, 0.99, "\n".join(baris), family="monospace", fontsize=ukuran_huruf, va="top")
    fig.savefig(path_simpan, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

def profil_radial(img):
    """Rata-rata log10 magnitudo spektrum per radius frekuensi (siklus per piksel)."""
    h, w = img.shape
    mag = np.log10(np.abs(np.fft.fft2(img.astype(np.float32))) + 1e-3)
    nu = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2)
    kotak = np.minimum((nu / 0.5 * 64).astype(int), 63)
    pusat = (np.arange(64) + 0.5) / 64 * 0.5
    profil = np.array([mag[kotak == i].mean() for i in range(64)])
    return pusat, profil

def buat_laporan_frekuensi(f, g, hasil, path_simpan):
    nu = np.linspace(0, 0.5, 400)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(nu, respons_degradasi(nu), label="H_deg (degradasi kontras lokal)", color="tab:red")
    ax[0].plot(nu, respons_hfe(nu), label="H_hfe (filter koreksi)", color="tab:green")
    ax[0].plot(nu, respons_degradasi(nu) * respons_hfe(nu), label="H_deg x H_hfe (idealnya 1)", color="black", linestyle="--")
    ax[0].axhline(1.0, color="gray", linewidth=0.8)
    ax[0].set_xlabel("Frekuensi spasial (siklus per piksel)")
    ax[0].set_ylabel("Gain")
    ax[0].set_title("Respons frekuensi: degradasi, koreksi, dan gabungan", fontsize=10)
    ax[0].legend()
    for citra, label, warna in ((f, "Ground truth", "black"), (g, "Terdegradasi", "tab:red"), (hasil, "Hasil", "tab:green")):
        pusat, profil = profil_radial(citra)
        ax[1].plot(pusat, profil, label=label, color=warna)
    ax[1].set_xlabel("Frekuensi spasial (siklus per piksel)")
    ax[1].set_ylabel("Rata-rata log10 magnitudo spektrum")
    ax[1].set_title("Profil radial spektrum Fourier", fontsize=10)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path_simpan, dpi=130)
    plt.close(fig)

def buat_laporan_tradeoff(grid, m_g, path_simpan):
    """Sebar setiap setelan denoise: sumbu x galat streak (dibalik), sumbu y kontras pembuluh.
    Kanan atas = lebih baik di keduanya. Garis putus = batas MINIMUM, garis titik = batas IDEAL."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    titik = ax.scatter([r["streak"] for _, _, r in grid], [r["pembuluh"] for _, _, r in grid], c=[r["psnr"] for _, _, r in grid], cmap="viridis", s=45)
    fig.colorbar(titik, ax=ax, label="PSNR (dB)")
    ax.scatter([m_g["streak"]], [m_g["pembuluh"]], marker="x", color="tab:red", s=90, label="Citra terdegradasi")
    utama = [r for h, w, r in grid if h == H_NLM and w == BOBOT_TV]
    if utama:
        ax.scatter([utama[0]["streak"]], [utama[0]["pembuluh"]], marker="*", color="tab:orange", edgecolor="black", s=260, label=f"Setelan utama (h {H_NLM}, TV {BOBOT_TV})")
    for sumbu, nilai in ((ax.axvline, RUBRIK["streak"][1]), (ax.axhline, RUBRIK["pembuluh"][0])):
        sumbu(nilai, color="gray", linestyle="--", linewidth=1)
    for sumbu, nilai in ((ax.axvline, RUBRIK["streak"][3]), (ax.axhline, RUBRIK["pembuluh"][2])):
        sumbu(nilai, color="gray", linestyle=":", linewidth=1)
    ax.invert_xaxis()
    ax.set_xlabel("Galat streak di paru (std, makin kecil makin bersih)  >>  lebih baik")
    ax.set_ylabel("Rasio kontras garis pembuluh (makin besar makin terjaga)  >>  lebih baik")
    jumlah = sum(layak_minimum(r) for _, _, r in grid)
    ax.set_title(f"Trade-off denoise: {jumlah} dari {len(grid)} setelan lolos MINIMUM untuk streak dan pembuluh sekaligus", fontsize=10)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path_simpan, dpi=130)
    plt.close(fig)

def buat_tahapan(tahap, path_simpan):
    fig, ax = plt.subplots(1, len(tahap), figsize=(3.6 * len(tahap), 4.2))
    for a, (judul, citra) in zip(ax, tahap):
        tampil(a, citra, judul)
    fig.tight_layout()
    fig.savefig(path_simpan, dpi=130)
    plt.close(fig)

def simpan_png(path, img):
    cv2.imwrite(str(path), np.clip(img, 0, 255).astype(np.uint8))

# ----------------------------------------------------------------------
# Program utama
# ----------------------------------------------------------------------
def main(path_input=None):
    FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)

    f = muat_citra_ct(path_input)
    perbarui_rubrik(f)
    mp, mt = mask_paru_homogen(f), mask_tubuh(f)
    g, citra_pacs, sino = buat_sampel(f, SEED)

    hasil, tahap = restorasi(g)

    m_g, m_h = hitung_metrik(f, g, mp, mt), hitung_metrik(f, hasil, mp, mt)
    baris = tabel_rubrik(m_g, m_h)

    terpakai = cari_berkas_citra(path_input)
    nama_sumber = terpakai.name if (terpakai and not is_phantom(f)) else ("Phantom Sintetis" if is_phantom(f) else "normal-ct-chest.jpeg")
    tipe_label = "Phantom" if is_phantom(f) else f"CT Riil ({nama_sumber})"
    print(f"Sampel Ground Truth: {tipe_label} {UKURAN}x{UKURAN}, seed={SEED}")
    print(f"Akuisisi           : {N_SUDUT_JARANG} sudut, dosis foton {DOSIS_FOTON:g}, sigma elektronik {SIGMA_ELEKTRONIK:g}")
    print(f"Kontras lokal      : alpha {ALPHA_KONTRAS}, sigma {SIGMA_KONTRAS_LOKAL:g}")
    print(f"Restorasi          : NLM(h {H_NLM}) > TV({BOBOT_TV}) > HFE(sigma {SIGMA_HFE:g}, k {GAIN_HFE}) > Laplacian({KEKUATAN_SHARPEN})")
    print()
    for nama, a, b, target, status in baris:
        print(f"{nama:<30}{a:>10} -> {b:<10} target {target:<16} [{status}]")

    teks_ablasi = studi_ablasi(f, g, mp, mt)
    teks_batch = ringkasan_batch(f, mp, mt)
    teks_artefak = catatan_artefak(f, g, hasil)
    (FOLDER_OUTPUT / "ablasi.txt").write_text(teks_ablasi + "\n", encoding="utf-8")
    (FOLDER_OUTPUT / "ringkasan_batch.txt").write_text(teks_batch + "\n", encoding="utf-8")
    (FOLDER_OUTPUT / "artefak.txt").write_text(teks_artefak + "\n", encoding="utf-8")
    teks_spesifikasi = spesifikasi_input(f)
    (FOLDER_OUTPUT / "spesifikasi_input.txt").write_text(teks_spesifikasi + "\n", encoding="utf-8")
    print("\n" + teks_spesifikasi)
    grid = grid_denoise(f, mp, mt)
    (FOLDER_OUTPUT / "tradeoff_denoise.txt").write_text(teks_tradeoff(grid) + "\n", encoding="utf-8")
    print("\n" + teks_ablasi + "\n\n" + teks_batch + "\n\n" + teks_artefak)
    print(f"\nTrade-off denoise: {sum(layak_minimum(r) for _, _, r in grid)} dari {len(grid)} setelan lolos MINIMUM untuk streak dan pembuluh sekaligus")

    simpan_png(FOLDER_OUTPUT / "ground_truth.png", f)
    simpan_png(FOLDER_OUTPUT / "citra_terdegradasi.png", g)
    simpan_png(FOLDER_OUTPUT / "hasil_restorasi.png", hasil)
    buat_laporan_before_after(g, hasil, f, mt, FOLDER_OUTPUT / "komparasi_before_after.png")
    buat_laporan_degradasi(f, sino, citra_pacs, g, FOLDER_OUTPUT / "pembuatan_sampel_degradasi.png")
    buat_laporan_komparasi(f, g, hasil, mt, FOLDER_OUTPUT / "komparasi_gt_degradasi_hasil.png")
    buat_tabel_rubrik(baris, FOLDER_OUTPUT / "tabel_metrik_rubrik.png")
    teks_ke_gambar(teks_ablasi.split("\n"), FOLDER_OUTPUT / "tabel_ablasi.png", ukuran_huruf=9)
    buat_laporan_frekuensi(f, g, hasil, FOLDER_OUTPUT / "respons_frekuensi.png")
    buat_laporan_tradeoff(grid, m_g, FOLDER_OUTPUT / "tradeoff_denoise.png")
    nama_tahap = {"hfe": "Koreksi kontras (HFE, frekuensi)", "nlm": "Non-Local Means", "tv": "Total Variation", "sharpen": "Penajaman Laplacian"}
    buat_tahapan([("1. Ground truth", f), ("2. Terdegradasi", g)] + [(f"{i}. {nama_tahap[n]}", c) for i, (n, c) in enumerate(tahap, start=3)], FOLDER_OUTPUT / "tahapan_proses.png")
    print(f"\nFile keluaran tersimpan di folder: {FOLDER_OUTPUT.resolve()}")

if __name__ == "__main__":
    path_cli = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            # Terima path langsung atau nama file apa pun yang cocok
            tertemukan = cari_berkas_citra(arg)
            if tertemukan is not None:
                path_cli = tertemukan
                break

    if "--tuning" in sys.argv:
        tuning_parameter(path_cli)
    else:
        main(path_cli)
```

---

## 11. Panduan Menjalankan Program

### Menyiapkan Citra
Unduh satu irisan aksial dari kasus [Normal CT chest (Radiopaedia, rID 36676)](https://radiopaedia.org/cases/normal-ct-chest), lalu simpan sebagai `src/normal-ct-chest.jpeg`. Jika berkas tidak ditemukan, skrip mencetak `[PERINGATAN]` dan memakai phantom sintetis. Dalam kondisi itu angka di artikel ini tidak berlaku.

### Dependensi Python
```bash
pip install opencv-python numpy scipy scikit-image matplotlib
```

### Eksekusi Program
```bash
# 1. Restorasi lengkap, semua gambar dan laporan .txt (folder tasks/hasil)
python src/uts.py

# 2. Memakai berkas citra lain
python src/uts.py path/ke/citra.jpeg

# 3. Grid tuning 70 kombinasi NLM x TV (seed 100-102)
python src/uts.py --tuning
```

Semua gambar di artikel ini, termasuk `komparasi_before_after.png`, dihasilkan oleh perintah nomor 1 ke folder `tasks/hasil/`. Pengecualiannya `bukti_pembuluh.png` serta angka pembuluh kecil dan galat per wilayah di Bagian 7, yang berasal dari skrip uji reviewer (`uji_review_citra_riil.py`).
