---
title: Penjadwalan KuliahKu
author: [Aruh1]
publish_date: 2026-09-23
post_slug: Penjadwalan-Kuliahku
featured: false
tags: ["kuliahku", "ui-ux", "product-design", "case-study"]
description: Dokumentasi analisis kebutuhan, persona, fitur utama, dan rancangan UI/UX aplikasi manajemen tugas dan penjadwalan KuliahKu.
lang: id
---

Link Canva: https://canva.link/sw8wewvwsx02afn

## BAB 1. PENDAHULUAN

### 1.1 Latar Belakang

Mahasiswa umumnya mengambil beberapa mata kuliah sekaligus dalam satu semester, dan tiap mata kuliah punya tugas dengan tenggat yang berbeda-beda. Informasi tugas biasanya tersebar di grup WhatsApp, LMS kampus, dan catatan pribadi masing-masing mahasiswa.

Kondisi ini membuat mahasiswa rawan lupa deadline, kesulitan menentukan tugas mana yang harus dikerjakan lebih dulu, dan kesulitan memantau progres pengerjaan tugas, baik tugas individu maupun tugas kelompok.

Masalah ini muncul karena tidak ada satu tempat yang menyatukan seluruh informasi tugas kuliah. Aplikasi KuliahKu dibangun untuk menjawab kebutuhan ini dengan memusatkan pencatatan, pemantauan, dan pengingat tugas kuliah dalam satu platform.

### 1.2 Rumusan Masalah

Masalah pengelolaan tugas kuliah muncul dalam bentuk berbeda tergantung situasi tiap mahasiswa. Rincian masalah dan dampaknya dirangkum pada tabel berikut:

| Situasi Mahasiswa        | Masalah                                                                                                   | Dampak                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Tugas Menumpuk**       | Informasi tugas tersebar di banyak grup dan platform berbeda, mudah lupa deadline.                        | Tugas terlewat dan nilai jadi taruhannya.                      |
| **Aktif Organisasi**     | Waktu terbagi antara kuliah, tugas, dan kegiatan organisasi, sulit menentukan prioritas.                  | Tugas dikerjakan mepet deadline, kualitas menurun.             |
| **Ketua Kelompok**       | Pembagian tugas lewat chat, sulit memantau siapa yang sudah dan belum mengerjakan.                        | Kerja kelompok tidak merata, tugas berisiko telat dikumpulkan. |
| **Sering Menunda Tugas** | Tidak ada pengingat, baru sadar deadline saat sudah dekat.                                                | Tugas dikerjakan terburu-buru, kualitas rendah.                |
| **Tingkat Akhir**        | Mengelola tugas kuliah, bimbingan, dan pekerjaan akademik lain secara bersamaan tanpa sistem terstruktur. | Beban mental tinggi, ada risiko tanggung jawab terlewat.       |

### 1.3 Tujuan Pengembangan Aplikasi

- Menyatukan seluruh informasi tugas kuliah dalam satu tempat.
- Membantu mahasiswa mengetahui deadline dan prioritas tugas dengan jelas.
- Mempermudah pemantauan progres pengerjaan tugas.
- Memudahkan koordinasi dan pembagian tugas kelompok.
- Mengurangi risiko tugas terlewat lewat pengingat otomatis.

### 1.4 Manfaat Aplikasi

- Mahasiswa tidak lagi kehilangan informasi tugas di tengah banyak grup chat.
- Prioritas tugas lebih jelas, membantu manajemen waktu sehari-hari.
- Kerja kelompok lebih transparan karena progres tiap anggota terlihat.
- Riwayat tugas membantu mahasiswa mengevaluasi beban kerja tiap semester.

---

## BAB 2. DESKRIPSI APLIKASI

### 2.1 Gambaran Umum

KuliahKu adalah website yang membantu mahasiswa mengelola tugas kuliah secara terorganisir. Pengguna dapat mencatat tugas, menentukan deadline dan prioritas, memantau progres pengerjaan, serta mengelola tugas kelompok dalam satu tempat. Website ini dipakai secara individu maupun bersama anggota lain saat mengerjakan tugas kelompok.

### 2.2 Fitur Utama

#### 2.2.1 Dashboard

Menampilkan ringkasan kondisi tugas begitu mahasiswa membuka aplikasi, mencakup tugas yang sedang berjalan, akan jatuh tempo, dan sudah selesai.

- Ringkasan jumlah tugas per status.
- Daftar tugas dengan deadline terdekat.

#### 2.2.2 Manajemen Tugas

Mahasiswa dapat menambahkan, mengubah, menghapus, dan melihat detail tugas. Tiap tugas mencatat mata kuliah, deadline, dan tingkat prioritas: rendah, sedang, atau tinggi.

- Input tugas per mata kuliah.
- Deadline dan prioritas sebagai atribut wajib tiap tugas.

#### 2.2.3 Kalender Tugas

Menampilkan seluruh tugas dalam tampilan kalender bulanan, sehingga mahasiswa bisa melihat sebaran deadline sekaligus, bukan hanya sebagai daftar memanjang.

- Tampilan bulanan dengan tanda pada tanggal deadline.
- Membantu mengenali minggu yang padat tugas lebih awal.

#### 2.2.4 Progress dan Status Tugas

Tiap tugas punya status yang bisa diperbarui: **Belum Dimulai**, **Sedang Dikerjakan**, dan **Selesai**. Status ini membantu mahasiswa maupun ketua kelompok memantau perkembangan pengerjaan.

- Tiga tahap status yang sederhana dan jelas.
- Progress terlihat oleh diri sendiri maupun anggota kelompok.

#### 2.2.5 Tugas Kelompok

Mahasiswa dapat membuat kelompok, menambahkan anggota, membagi bagian tugas ke tiap anggota, dan memantau status pengerjaan masing-masing anggota dalam satu tampilan.

- Pembagian tugas per anggota kelompok.
- Status progres tiap anggota terlihat oleh ketua kelompok.

#### 2.2.6 Reminder dan Riwayat Tugas

Sistem mengirim pengingat otomatis saat deadline semakin dekat, dan menyimpan tugas yang sudah selesai sebagai riwayat yang bisa dilihat kembali.

- Pengingat otomatis menjelang deadline.
- Arsip tugas selesai sebagai riwayat akademik.

#### 2.2.7 Jadwal Kuliah

Menampilkan jadwal kuliah mingguan per hari, mencakup nama mata kuliah, jam, ruang kelas, dan dosen pengampu. Jadwal ini terhubung dengan tugas, sehingga tiap tugas otomatis terkait ke mata kuliah yang sesuai.

- Tampilan jadwal per hari dalam satu minggu.
- Info ruang dan dosen tiap mata kuliah.
- Pengingat sebelum jam kelas dimulai.

#### 2.2.8 Catatan Dosen

Kolom catatan bebas yang bisa ditambahkan ke tiap tugas, tempat mahasiswa menulis instruksi tambahan, syarat khusus, atau pesan dosen yang disampaikan lisan di kelas dan tidak tertulis di deskripsi tugas resmi.

- Catatan tambahan per tugas, terpisah dari deskripsi utama.
- Bisa diedit kapan saja saat ada info baru dari dosen.

---

## BAB 3. ANALISIS PENGGUNA (PERSONA)

### 3.1 Pengertian Persona

Persona adalah profil pengguna rekaan yang mewakili satu kelompok target pengguna nyata. Persona bukan orang sungguhan, tapi rangkuman dari kebiasaan, tujuan, dan masalah yang dialami pengguna asli, digabung menjadi satu karakter yang mudah dipahami. Persona membantu tim desain membangun fitur berdasarkan kebutuhan nyata, bukan asumsi semata.

Aplikasi KuliahKu melibatkan lima persona, mewakili lima situasi berbeda yang umum dialami mahasiswa dalam mengelola tugas kuliah.

### 3.2 Daftar Persona

#### Persona 1: Rina Putri

- **Usia / Status**: 19 tahun, Mahasiswa Semester 2
- **Tujuan**: Mengetahui seluruh tugas dan deadline dari berbagai mata kuliah agar tidak ada yang terlupakan.
- **Frustrasi**: Tugas datang dari banyak mata kuliah dan tersebar di berbagai grup, sering lupa deadline.
- **Kebutuhan dari Aplikasi**: Dashboard dan Kalender Tugas untuk melihat seluruh tugas dalam satu tempat, ditambah Reminder supaya tidak ada deadline yang terlewat.

> _"Aku cuma pengen tahu, minggu ini tugas apa aja yang harus dikumpulkan."_

#### Persona 2: Fajar

- **Usia / Status**: 21 tahun, Mahasiswa Semester 4, Aktif Organisasi
- **Tujuan**: Mengatur waktu antara kuliah, tugas, dan kegiatan organisasi kampus.
- **Frustrasi**: Jadwal yang padat membuat sulit menentukan kapan harus mengerjakan tugas.
- **Kebutuhan dari Aplikasi**: Fitur Prioritas dan Reminder untuk membantu menyusun waktu di tengah jadwal yang padat.

> _"Aku butuh tahu mana yang harus dikerjain duluan, soalnya waktu aku kepotong buat organisasi."_

#### Persona 3: Andi

- **Usia / Status**: 20 tahun, Mahasiswa Semester 3, Ketua Kelompok
- **Tujuan**: Memastikan semua anggota kelompok menyelesaikan bagian tugasnya tepat waktu.
- **Frustrasi**: Pembagian tugas lewat chat membuat sulit mengetahui siapa yang sudah dan belum mengerjakan bagiannya.
- **Kebutuhan dari Aplikasi**: Fitur Tugas Kelompok untuk membagi tugas dan memantau status progres tiap anggota.

> _"Aku pengen tahu progress temen temen tanpa harus nanya satu satu di grup."_

#### Persona 4: Salsabila

- **Usia / Status**: 20 tahun, Mahasiswa Semester 3
- **Tujuan**: Mengetahui tugas mana yang harus dikerjakan lebih dulu.
- **Frustrasi**: Sering menunda tugas dan baru sadar deadline sudah dekat.
- **Kebutuhan dari Aplikasi**: Prioritas Tugas dan Reminder yang mengingatkan jauh sebelum deadline tiba.

> _"Kadang aku baru sadar deadline udah besok. Aku butuh diingetin dari jauh jauh hari."_

#### Persona 5: Rizky

- **Usia / Status**: 23 tahun, Mahasiswa Semester 8
- **Tujuan**: Mengatur tugas kuliah, bimbingan skripsi, dan pekerjaan akademik lain secara bersamaan.
- **Frustrasi**: Banyak tanggung jawab berjalan bersamaan tanpa sistem yang membantu mengaturnya.
- **Kebutuhan dari Aplikasi**: Dashboard dan Riwayat Tugas untuk melihat seluruh tanggung jawab dan progres yang sudah dilalui.

> _"Di semester akhir ini, aku nggak mau ada satu pun tanggung jawab yang kelewat begitu aja."_

---

## BAB 4. RANCANGAN UI/UX

### 4.1 Pendekatan Desain

Desain KuliahKu mengutamakan kejelasan informasi tugas, terutama deadline dan prioritas, agar mahasiswa bisa langsung menangkap apa yang perlu dikerjakan lebih dulu tanpa membaca detail satu per satu. Warna dipakai sebagai penanda status: merah untuk prioritas tinggi atau deadline dekat, hijau untuk tugas yang sudah selesai, sehingga kondisi tugas mudah dikenali sekilas.

### 4.2 Struktur Halaman

Rancangan KuliahKu terdiri dari lima halaman utama: halaman masuk, dashboard ringkasan tugas, daftar tugas lengkap dengan filter status, form tambah tugas, dan kalender tugas bulanan. Alur penggunaan dimulai dari dashboard, lalu mahasiswa membuka daftar tugas untuk melihat detail, atau membuka kalender untuk melihat sebaran deadline dalam satu bulan.

- **Halaman Masuk**: Form email dan kata sandi untuk mengakses akun.
- **Dashboard**: Ringkasan jumlah tugas per status dan tugas dengan deadline terdekat.
- **Daftar Tugas**: Seluruh tugas dengan filter berdasarkan status pengerjaan.
- **Tambah Tugas**: Form input nama tugas, mata kuliah, deadline, prioritas, dan deskripsi.
- **Kalender**: Tampilan bulanan dengan tanda pada tanggal yang punya deadline.

---

## BAB 5. KESIMPULAN

KuliahKu menjawab masalah pengelolaan tugas kuliah yang selama ini tersebar di berbagai platform dan bergantung pada ingatan pribadi tiap mahasiswa. Dengan memusatkan pencatatan tugas, deadline, prioritas, dan progres dalam satu aplikasi, mahasiswa mendapat kejelasan tentang apa yang harus dikerjakan lebih dulu, sementara ketua kelompok mendapat visibilitas atas kerja tiap anggota.

Analisis lima persona menunjukkan bahwa meski situasi tiap mahasiswa berbeda, mulai dari yang kewalahan dengan banyak tugas, aktif berorganisasi, memimpin kelompok, sering menunda pekerjaan, hingga mahasiswa tingkat akhir dengan banyak tanggung jawab, kebutuhan inti mereka sama: kejelasan atas apa yang harus dikerjakan dan kapan tenggatnya. Rancangan fitur KuliahKu disusun berdasarkan kebutuhan tersebut, bukan asumsi semata.
