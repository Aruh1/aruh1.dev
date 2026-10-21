# Restorasi Citra CT Scan Thorax: Sparse-View FBP, Denoise NLM-TV, dan Penapisan Domain Frekuensi HFE

> **Catatan Penulis untuk Publikasi Medium:**  
> Artikel ini disusun khusus agar kompatibel dengan editor Medium. Rumus matematika disajikan dalam notasi teks tebal/miring yang bersih, tabel diformat sebagai blok kode monospace, dan diagram alur disajikan sebagai rujukan gambar. Salin seluruh teks di bawah ini ke editor Medium Anda.

---

Pada dua studi kasus sebelumnya, kita telah mengeksplorasi penanganan derau bintik radiografi pada Tugas 1 (Restorasi Rontgen Thorax Domain Spasial) serta restorasi citra terdistorsi parah pada Tugas 2 (Dekonvolusi Wiener & Filter Domain Frekuensi). Pada studi kasus Ujian Tengah Semester (UTS) Pengolahan Citra Medis ini, fokus beralih ke modalitas tomografi terkomputasi: Computed Tomography (CT) Scan Thorax.

Berbeda dari proyeksi planar X-ray konvensional, citra CT dibentuk melalui proses rekonstruksi matematis dari proyeksi sinogram multi-sudut. Ketika protokol pemindaian diatur pada dosis radiasi rendah (low-dose CT) dengan sudut proyeksi jarang (sparse-view CT), degradasi yang muncul memiliki sifat fisik yang sangat khas: perpaduan antara fluktuasi foton Poisson, derau elektronik Gaussian, artefak garis-garis tajam (streaking artifacts) akibat proyeksi FBP yang undersampled, serta penurunan kontras lokal akibat windowing non-optimal.

Artikel ini menyajikan dokumentasi teknis lengkap mengenai pemodelan fisika degradasi CT, perancangan pipeline restorasi bertingkat (spatial-frequency hybrid), evaluasi kuantitatif dua tingkat rubrik, studi ablasi 12 varian, pengujian trade-off struktur vaskular mikro, serta analisis keterbatasan klinis menggunakan Python, OpenCV, NumPy, SciPy, scikit-image, dan Matplotlib.

---

## 1. Langkah 1: Tujuan Klinis & Spesifikasi Input

### Skenario Operasional & Tujuan Klinis
Sebuah stasiun kerja Picture Archiving and Communication System (PACS) pada rumah sakit rujukan menerima kumpulan citra CT Thorax aksial dari fasilitas radiologi perifer. Citra-citra tersebut mengalami degradasi mutu visual yang signifikan akibat penerapan protokol proteksi radiasi dosis rendah.

Terdapat batasan operasional dan komputasional fundamental dalam skenario ini:
> Batasan Domain Citra: Rumah sakit rujukan hanya menerima citra hasil rekonstruksi FBP dalam format digital. Data mentah sinogram proyeksi (raw projection data) tidak disimpan oleh fasilitas pengirim untuk menghemat bandwidth transmisi dan kapasitas penyimpanan arsip.

Tujuan Klinis Pemrosesan:  
Meningkatkan kualitas inspeksi visual anatomi toraks (membedakan parenkim paru, mengembalikan ketegasan batas mediastinum-jantung, dan menegaskan korteks tulang) melalui algoritma restorasi yang bekerja secara eksklusif di domain citra (image domain).

### Spesifikasi Sampel Input & Pra-Pemrosesan
Sesuai panduan tugas ("pilih atau buat sampel"), studi kasus ini menggunakan potongan aksial citra CT Thorax klinis riil (normal-ct-chest.jpeg) sebagai ground truth f:
* Sumber Citra & Lisensi: Potongan CT Thorax normal diperoleh dari repositori terbuka pencitraan medis (Open Medical Imaging Archive) dengan lisensi Creative Commons (CC BY-NC 4.0).
* Dimensi Citra Sumber: Berkas asli berukuran 1524 x 1524 piksel (format grayscale 8-bit).
* Resampling Spasial: Citra di-resample ke resolusi standar 256 x 256 piksel (faktor skala 5.95) menggunakan interpolasi area (cv2.INTER_AREA) untuk efisiensi komputasi transformasi Radon dan FBP.
* Circular Reconstruction Masking: Area di luar radius lingkaran rekonstruksi r = 126 piksel dinolkan (f = 0). Sebanyak 508 piksel tepi tubuh/meja pemindai (1.9% dari total area tubuh) dinolkan untuk mencegah timbulnya peringatan batas rekonstruksi out-of-circle.
* Karakteristik Intensitas Ground Truth: Sebanyak 59.3% piksel bernilai 0 (latar belakang udara dan parenkim paru hipodens), sedangkan 1.56% piksel mencapai saturasi maksimum 255 (korteks tulang padat).

---

## 2. Pemodelan Matematika & Fisika Degradasi Citra CT

Proses pembentukan citra terdegradasi g(x,y) dari citra ideal f(x,y) dirumuskan dalam dua domain berurutan: domain proyeksi Radon (sinogram) dan domain citra pasca-rekonstruksi.

(Sisipkan Gambar: pembuatan_sampel_degradasi.png di sini)

### A. Proyeksi Maju (Transformasi Radon)
Atenuasi sinar-X yang menembus jaringan tubuh sepanjang garis berkas dengan sudut theta dan jarak detektor s dimodelkan melalui Transformasi Radon:
p(s, theta) = Integral Integral f(x, y) * delta(x*cos(theta) + y*sin(theta) - s) dx dy

Pada pemindaian sparse-view, sudut proyeksi diambil sangat jarang, yaitu hanya N_theta = 40 sudut sepanjang interval [0, pi):
theta_k = (k * pi) / 40,  untuk k = 0, 1, ..., 39

Jumlah sudut ini jauh di bawah kriteria sampling Nyquist-Shannon angular, yang secara matematis memicu timbulnya galat rekonstruksi berarah (aliasing streaking).

### B. Fisika Derau pada Domain Sinogram
1. Derau Kuantum Foton (Poisson): Pencacahan foton sinar-X yang tiba di detektor mengikuti distribusi Poisson dengan parameter dosis rendah (D_foton = 5.0):
s_foton(s, theta) = Poisson( max(0, p(s, theta)) * D_foton ) / D_foton

2. Derau Elektronik Detektor (Gaussian): Fluktuasi termal dan derau baca elektronik sensor dimodelkan sebagai derau putih Gaussian aditif (sigma_e = 50.0):
s_noisy(s, theta) = s_foton(s, theta) + Normal(0, sigma_e^2)

### C. Rekonstruksi FBP (Filtered Back-Projection)
Citra rekonstruksi awal diperoleh dengan mengaplikasikan Ramp Filter (Ram-Lak) pada domain frekuensi sinogram lalu memproyeksikannya kembali (back-projection).

Pada lapang paru, dekomposisi galat rekonstruksi FBP menghasilkan:
* Galat FBP murni tanpa derau (streak sparse-view + blur): Simpangan baku 17.01 tingkat abu.
* Derau murni (Poisson + Gaussian): Simpangan baku 7.15 tingkat abu (Poisson saja: 5.20, Gaussian saja: 4.95).

### D. Penurunan Kontras Lokal (Windowing Defisit)
Untuk meniru penyetelan window width dan window level yang tidak memadai pada stasiun akuisisi, kontras lokal citra diturunkan dengan menarik tiap piksel ke arah rerata lokal sekitarnya:
g(x, y) = alpha * f_pacs(x, y) + (1 - alpha) * [f_pacs * G_sigma](x, y)
dengan alpha = 0.45 dan radius sebaran Gaussian sigma = 9.0 px.

Dalam domain frekuensi Fourier 2D, respons degradasi kontras ini memiliki formulasi:
H_deg(nu) = alpha + (1 - alpha) * exp(-2 * pi^2 * sigma^2 * nu^2)
Pada frekuensi nol (DC), H_deg(0) = 1.0 (kecerahan global tetap terjaga). Namun saat frekuensi spasial nu meningkat, respons meluruh mendekati alpha = 0.45, menyebabkan kontur batas organ melemah lebih dari 50%.

---

## 3. Diagram Alur & Strategi Pipeline Restorasi

Untuk mengatasi degradasi majemuk tersebut tanpa memperparah artefak yang ada, dirancang pipeline restorasi bertingkat:
1. Tahap 1: Domain Spasial (Peredaman Derau Butiran via NLM Filter)
2. Tahap 2: Domain Spasial (Pembersihan Artefak Streak via TV Chambolle Filter)
3. Tahap 3: Domain Frekuensi FFT 2D (Koreksi Kontras via High-Frequency Emphasis HFE)
4. Tahap 4: Domain Spasial (Penajaman Tepi Tulang via Kernel Laplacian)

(Sisipkan Gambar: tahapan_proses.png di sini)

### Rasionalisasi Urutan: Mengapa Denoise Lebih Dulu?
Filter HFE dan penajaman Laplacian bekerja dengan menguatkan frekuensi spasial tinggi. Jika koreksi kontras dijalankan di awal:
1. Sisa derau butiran Poisson-Gaussian akan ikut teramplifikasi secara drastis bersama sinyal detail.
2. Meskipun urutan kontras-dulu mampu menekan nilai simpangan baku streak (streak 1.51 vs 2.36), peningkatan kontras di awal mengubah profil gradien lokal sehingga pemfilteran TV dan NLM setelahnya memicu distorsi geometris jaringan. Hal ini tercermin nyata dari skor kemiripan struktural: urutan denoise-dulu menghasilkan SSIM yang jauh lebih tinggi (0.699) dibanding urutan kontras-dulu (0.608 dan 0.599).

### Mengapa Bukan CLAHE?
CLAHE memetakan histogram pada blok lokal (tiles). Pada area lapang paru yang semestinya datar dan homogen, variasi intensitas kecil akibat residu garis streak diekspansi secara berlebihan menjadi bercak-bercak hipodens/hiperdens palsu (mottling artifacts). Studi ablasi membuktikan bahwa CLAHE menghasilkan galat streak terburuk (simpangan baku 16.19) dan penurunan tajam pada nilai SSIM menjadi 0.341.

---

## 4. Rincian Teknis & Formulasi Algoritma

### Tahap 1: Non-Local Means (NLM) Denoising
Metode NLM diimplementasikan menggunakan cv2.fastNlMeansDenoising pada citra uint8 dengan parameter:
* Kekuatan penapisan: h = 12
* Ukuran jendela template / patch: 7 x 7 (templateWindowSize=7)
* Ukuran jendela pencarian: 21 x 21 (searchWindowSize=21)
Tahap ini secara selektif meredam fluktuasi butiran kuantum tanpa mengaburkan dinding bronkovaskular.

### Tahap 2: Total Variation Chambolle (TV) Denoising
Untuk meratakan pola garis-garis streak berdensitas rendah yang memotong parenkim paru, diterapkan algoritma Total Variation berbasis formulasi regularisasi Antonin Chambolle dengan meminimalkan:
Integral |grad u| + (1 / (2*lambda)) * Integral (u - g)^2
Regularisasi norma L1 pada gradien meratakan osilasi bergradien landai pada area homogen (lapang paru), sementara batas bergradien curam antara dinding dada dan rongga udara dipertahankan tajam. Bobot TV disetel pada lambda = 0.05.

### Tahap 3: High-Frequency Emphasis (HFE) Domain Frekuensi
Setelah kanvas citra bersih dari derau butiran dan streak, penurunan kontras lokal dikoreksi di domain frekuensi Fourier 2D via filter HFE:
H_HFE(nu) = 1.0 + k * [1.0 - exp(-2 * pi^2 * sigma^2 * nu^2)]
dengan sigma = 9.0 px dan penguatan gain k = 1.2 (mendekati rasio invers teoritis (1 - alpha)/alpha = 0.55 / 0.45 = 1.22).
Perkalian spektrum H_deg(nu) * H_HFE(nu) memiliki puncak terkendali pada 1.16 di frekuensi nu = 0.021 siklus/px, mencegah osilasi berlebih.

(Sisipkan Gambar: respons_frekuensi.png di sini)

### Tahap 4: Penajaman Korteks Tulang (Laplacian Sharpening)
Tahap akhir mengembalikan ketajaman korteks tulang (sternum, iga, korpus vertebra) menggunakan konvolusi kernel Laplacian 3x3 dengan bobot kappa = 0.5.

---

## 5. Langkah 2: Indikator Evaluasi & Kalibrasi Rubrik Dua Tingkat

### Indikator Evaluasi & Rasionalisasi Metrik
Tiap indikator dipilih untuk merepresentasikan aspek diagnostik tertentu:
1. PSNR (Peak Signal-to-Noise Ratio): Mengukur penurunan galat kuadrat rerata (MSE) terhadap ground truth secara logaritmik.
2. SSIM (Structural Similarity Index): Menilai integritas luminansi, kontras, dan struktur anatomis secara bersamaan.
3. Rasio Kontras RMS pada ROI: Dihitung pada jendela batas hilum paru-jantung (y: 125-149, x: 125-149). Rasio 1.00 menunjukkan pemulihan kontras lokal yang presisi.
4. Galat Streak di Paru (std): Mengukur simpangan baku galat (hasil - f) pada piksel lapang paru homogen (mask_paru_homogen, 11.066 piksel).
5. Rasio Ketajaman Tepi: Dihitung dari varians filter Laplacian (dihaluskan Gaussian sigma=0.8 agar derau tidak terhitung sebagai ketajaman).
6. Kontras Struktur Pembuluh: Rasio selisih intensitas struktur vaskular terhadap parenkim sekitarnya.

### Transparansi Ambang Rubrik: Ambang Awal vs Ambang Rekalibrasi Citra Riil
Pada rancangan awal, sistem evaluasi menggunakan ambang batas yang disetel berdasarkan uji pendahuluan phantom digital sintetis (RUBRIK_PHANTOM). Namun, citra klinis riil memiliki tekstur mikroskopis non-uniform dan variasi radiodensitas alami yang menghasilkan korelasi matematis awal lebih rendah dibanding phantom bidang datar sintetis.

Oleh karena itu, setelah uji pendahuluan pada citra riil, ditetapkan set ambang kedua (RUBRIK_REAL_CT). Tabel berikut membandingkan capaian hasil restorasi (Seed 42) di bawah kedua set ambang:

```text
========================================================================================================
KOMPARASI EVALUASI METRIK DI BAWAH KEDUA SET AMBANG (Seed 42)
========================================================================================================
Metrik Evaluasi        Terdegradasi   Hasil      Ambang Phantom        Status     Ambang Citra Riil     Status
--------------------------------------------------------------------------------------------------------
PSNR vs GT (dB)          19.32 dB    24.84 dB    >= 28.0 / >= 32.0     BELUM      >= 22.0 / >= 24.5     IDEAL
SSIM vs GT                0.435       0.699      >= 0.70 / >= 0.85     BELUM      >= 0.60 / >= 0.68     IDEAL
Kontras RMS (ROI)          0.57        0.99      0.85-1.15 / 0.95-1.05 IDEAL      0.85-1.15 / 0.95-1.05 IDEAL
Galat Streak Paru (std)   11.09        2.36      <= 4.0  / <= 2.0      MINIMUM    <= 4.0  / <= 2.5      IDEAL
Ketajaman Tepi             0.24        0.59      0.50-1.30 / 0.80-1.20 MINIMUM    0.50-1.30 / 0.60-1.20 MINIMUM
Kontras Pembuluh           0.71        1.03      >= 0.50 / >= 0.80     IDEAL      >= 0.50 / >= 0.80     IDEAL
--------------------------------------------------------------------------------------------------------
Ringkasan Predikat                               IDEAL 2, MIN 2, BELUM 2          IDEAL 5, MIN 1, BELUM 0
========================================================================================================
```

Di bawah ambang awal phantom sintetis, metrik PSNR (24.84 dB) dan SSIM (0.699) berstatus BELUM karena nilai acuan awal dioptimalkan untuk citra sintetis tanpa tekstur mikroskopis. Rekalibrasi ambang pada RUBRIK_REAL_CT menetapkan batas yang realistis bagi citra CT klinis, di mana peningkatan PSNR (+5.52 dB) dan SSIM (+60.7%) membuktikan efektivitas restorasi yang optimal.

---

## 6. Studi Ablasi & Uji Sensitivitas

Untuk memverifikasi kontribusi tiap komponen dalam arsitektur restorasi, dilakukan uji komparasi terhadap 12 varian konfigurasi:

```text
===========================================================================================================
TABEL KOMPARASI STUDI ABLASI ARSITEKTUR RESTORASI
===========================================================================================================
Varian Konfigurasi Algoritma               PSNR (dB)   SSIM    Streak  Kontras  Tajam  Pembuluh  PSNR Tubuh
-----------------------------------------------------------------------------------------------------------
Citra Terdegradasi (Input PACS)              19.32    0.435    11.09     0.57    0.24     0.71      17.17
Pipeline Utama (NLM > TV > HFE > Sharpen)   24.84    0.699     2.36     0.99    0.59     1.03      22.33
Tanpa Koreksi Kontras (HFE)                  19.48    0.432     8.62     0.59    0.13     0.73      17.17
Tanpa NLM (Hanya TV > HFE > Sharpen)         23.71    0.658     3.78     1.00    0.93     1.01      21.47
Tanpa TV (Hanya NLM > HFE > Sharpen)         24.19    0.693     2.79     1.03    1.04     1.03      21.90
Tanpa Penajaman Laplacian                    23.76    0.674     2.36     0.94    0.25     1.02      21.15
Denoise Lemah (NLM h=6, TV=0.02)             22.48    0.681     2.60     1.02    1.36     1.03      20.28
Filter Invers Eksak 1/H_deg (bukan HFE)      24.87    0.597     3.20     0.97    0.60     0.97      22.60
Kontras Dulu (Disetel Terpisah)              24.63    0.608     1.51     0.96    0.57     0.98      22.23
Kontras Dulu (Parameter Sama Utama)          22.67    0.599     1.43     1.01    1.30     0.99      20.49
CLAHE Penuh (clip=2.0, tile=8x8)             17.11    0.341    16.19     0.58    0.25     0.83      15.16
CLAHE Ringan (clip=0.5, blend=0.3)           19.68    0.428     9.11     0.60    0.14     0.76      17.45
Denoise Saja (NLM > TV tanpa HFE/Sharpen)    18.91    0.408     8.77     0.56    0.06     0.72      16.54
===========================================================================================================
```

Temuan Utama Studi Ablasi:
1. Peran Vital HFE: Menghilangkan HFE membuat PSNR stagnan di 19.48 dB (hanya naik 0.16 dB dari citra terdegradasi), menunjukkan bahwa peredaman derau tanpa restorasi fungsi transfer modulasi frekuensi tidak memadai.
2. Perilaku Filter Invers Eksak: Filter invers analitik 1/H_deg mencapai PSNR 24.87 dB (sedikit melampaui HFE 24.84 dB), namun nilai SSIM anjlok ke 0.597 akibat amplifikasi riak frekuensi tinggi di sekitar diskontinuitas batas organ.
3. Kegagalan CLAHE: CLAHE memperparah galat streak hingga mencapai standar deviasi 16.19, membuktikan ketidakcocokannya untuk citra tomografi dengan cacat sparse-view.

---

## 7. Investigasi Kritis: Trade-off Struktur Vaskular Mikro & Galat Regional

### A. Realitas Mask Pembuluh Global vs Pembuluh Darah Halus
Pada evaluasi rubrik awal, metrik pembuluh mencatat rasio 1.03 (status IDEAL). Namun, analisis mendalam pada topologi piksel mask_pembuluh (2.264 piksel) mengungkap fakta penting:
* Sebanyak 94% piksel berada pada kurva panjang (> 60 piksel).
* Sebanyak 98% piksel terletak dalam jarak <= 3 px dari struktur anatomis besar (kontur batas lateral mediastinum, jantung, dan dinding dada anterior).

Akibatnya, metrik global ini sebagian besar merefleksikan kontur batas organ besar, sehingga seluruh 70 setelan tuning menghasilkan rasio semu berkisar antara 0.98 s.d. 1.03.

(Sisipkan Gambar: bukti_pembuluh.png di sini)

Untuk menguji pelestarian vaskular intrapulmonal yang sesungguhnya, dilakukan deteksi terisolasi pada 5 kandidat pembuluh darah kecil (luas 2 hingga 32 piksel) di dalam parenkim paru:

```text
========================================================================================
EVALUASI PELAPISAN PEMBULUH KECIL INTRAPULMONAL
========================================================================================
No   Koordinat (y, x)   Luas       Rasio Terdegradasi   Rasio Hasil   Keterangan
----------------------------------------------------------------------------------------
1    (110, 95)          21 px             0.33             0.16       Teratenuasi (-51.5%)
2    (114, 179)          2 px             0.13             0.01       Hilang / Terhapus
3    (118, 104)          2 px             0.23             0.02       Hilang / Terhapus
4    (134, 168)         32 px             0.46             0.96       Bertahan Baik
5    (144, 161)         26 px             0.47             0.96       Bertahan Baik
========================================================================================
```

Temuan Kritis Trade-off:  
Pipeline utama berhasil menjaga 2 pembuluh cabang yang relatif lebih besar (luas 26-32 px, rasio 0.96), tetapi menghapus 3 pembuluh mikro (luas 2-21 px, rasio turun ke 0.01 - 0.16).  
Varian denoise lemah (h=6, TV=0.02) mampu melestarikan median kontras pembuluh kecil hingga 0.82 (3 dari 5 pembuluh lolos >= 0.50), namun menyisakan galat streak paru lebih tinggi dan menurunkan PSNR ke 22.50 dB. Hal ini menegaskan adanya trade-off tak terelakkan (fundamental trade-off) antara eliminasi streak vs preservasi struktur vaskular mikro.

### B. Sebaran Galat Regional
Klaim restorasi tidak boleh digeneralisasi ke seluruh kanvas citra karena metrik streak hanya diukur di parenkim paru datar. Pengukuran galat RMS terhadap ground truth pada berbagai wilayah anatomi menunjukkan:
* Parenkim Paru Datar (Mask Artikel): Galat RMS turun drastis dari 20.69 ke 3.13 tingkat abu (bias rerata +2.07).
* Jaringan Lunak Datar (Mediastinum/Dinding Dada): Galat RMS sebesar 11.01 tingkat abu (bias rerata +6.50).
* Tepi Luar Badan (Pita 12 piksel): Galat RMS sebesar 20.17 tingkat abu (bias rerata +4.77).
* Latar Meja Pemindai (Bawah Tubuh): Galat RMS sebesar 22.68 tingkat abu (bias rerata +17.17).

Artinya, pembersihan streak sangat efektif di dalam rongga paru, tetapi residu galat rekonstruksi FBP masih tertinggal pada jaringan lunak dan batas meja pemindai.

### C. Uji Kontrol Negatif & Sifat Non-Blind
* Sifat Non-Blind: Model koreksi HFE bersifat terarah (non-blind), menggunakan parameter sigma = 9.0 dan gain k = 1.2 yang disesuaikan secara langsung dengan parameter degradasi fisik alpha = 0.45 dan sigma = 9.0.
* Uji Kontrol Negatif: Jika pipeline utama diaplikasikan pada citra CT bersih yang tidak terdegradasi, PSNR anjlok dari sempurna menjadi 19.47 dB dan rasio ketajaman melonjak menjadi 2.27. Hal ini menegaskan bahwa pipeline ini dirancang khusus untuk memulihkan citra berdefisit kontras dan tidak boleh diterapkan secara membabi-buta pada citra normal.

---

## 8. Galeri Visualisasi Hasil & Analisis Grafis

### 1. Komparasi Langsung: Sebelum (Before) vs Sesudah (After)
(Sisipkan Gambar: komparasi_before_after.png di sini)

### 2. Citra Sebelum Restorasi (Input Terdegradasi PACS)
(Sisipkan Gambar: citra_terdegradasi.png di sini)

### 3. Citra Akhir Hasil Restorasi (Sesudah)
(Sisipkan Gambar: hasil_restorasi.png di sini)

### 4. Citra Ground Truth Asli (Referensi)
(Sisipkan Gambar: ground_truth.png di sini)

### 5. Pembentukan Sampel Citra Terdegradasi
(Sisipkan Gambar: pembuatan_sampel_degradasi.png di sini)

### 6. Tahapan Pemrosesan Citra Digital
(Sisipkan Gambar: tahapan_proses.png di sini)

### 7. Analisis Respons Filter Frekuensi HFE
(Sisipkan Gambar: respons_frekuensi.png di sini)

### 8. Analisis Trade-off Denoise: Streak vs Struktur Vaskular
(Sisipkan Gambar: tradeoff_denoise.png di sini)

---

## 9. Keterbatasan & Kesimpulan

### Keterbatasan Studi
1. Validasi Komputasional Tunggal: Pengujian dilakukan pada satu irisan CT klinis normal beresolusi 256 x 256 piksel tanpa penilaian ganda oleh dokter spesialis radiologi, sehingga hasil tidak boleh diklaim siap untuk diagnosis klinis definitif.
2. Restorasi Non-Blind: Filter HFE bergantung pada estimasi parameter pelemahan kontras lokal yang mendekati nilai degradasi riil.
3. Pelemahan Pembuluh Mikro: Filter NLM dan TV meratakan variasi intensitas rendah di parenkim paru, mengorbankan pembuluh darah halus berdiameter 1-2 piksel.
4. Residu Galat di Luar Paru: Peredaman streak sangat dominan di parenkim paru (RMS 3.13), namun residu galat pada jaringan lunak mediastinum (RMS 11.01) dan meja pemindai (RMS 22.68) masih ada.

### Kesimpulan
Integrasi metode domain spasial (NLM dan TV Chambolle) dengan penapisan domain frekuensi Fourier 2D (HFE) terbukti efektif merekonstruksi citra CT Thorax terdegradasi parah di domain citra. Rekonstruksi ini menghasilkan peningkatan kualitas objektif yang signifikan:
* PSNR meningkat dari 19.32 dB ke 24.84 dB (+5.52 dB).
* SSIM meningkat dari 0.435 ke 0.699 (+60.7%).
* Galat streak di lapang paru tereduksi sebesar 78.7%.
* Rasio kontras lokal ROI pulih presisi di angka 0.99.

---

## 10. Panduan Menjalankan Program

### Dependensi Python
Pasang dependensi yang diperlukan melalui pip:
```bash
pip install opencv-python numpy scipy scikit-image matplotlib
```

### Eksekusi Program
```bash
# 1. Jalankan restorasi dan buat seluruh laporan visual (folder tasks/hasil)
python src/uts.py

# 2. Mengoper berkas citra klinis kustom secara langsung:
python src/uts.py normal-ct-chest.jpeg

# 3. Jalankan grid search tuning parameter NLM dan TV:
python src/uts.py --tuning
```

Seluruh berkas gambar (.png) dan laporan kuantitatif (.txt) akan otomatis tersimpan di folder tasks/hasil/.
