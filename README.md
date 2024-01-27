# Predict Customer Personality to Boost Marketing Campaign

**Tool** : Jupyter Notebook | [Link Notebook](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Predict%20Customer%20Personality.ipynb)<br>
**Programming Language** : Python <br>
**Libraries** : Pandas, NumPy, sklearn <br>
**Visualization** : Matplotlib, Seaborn, yellow-brick<br>
**Source Dataset** : Rakamin Academy

**Table of Contents**

- [STAGE 0: Problem Statement](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning#-stage-0-problem-statement)
  - [Introduction](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/tree/main#introduction)
  - [Goal](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/tree/main#goal)
  - [Objective](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/tree/main#objective)
- STAGE 1: Data Preparation
  - Data Quality Asssessment
  - Feature Engineering
- STAGE 2: Data Exploration
  - Conversion Rate by Income, Spending, and Age
  - Income and Total Spending
- STAGE 3: Data Modeling with K-Means Clustering
  - Pre-processing
  - Modeling
  - Evaluation
- STAGE 4: Customer Personality Analysis
- STAGE 5: Business Recommendation

## 📂 STAGE 0: Problem Statement

### **Introduction**

- Masalah: Bagaimana memahami karakteristik pelanggan untuk meningkatkan efektivitas strategi pemasaran?
- Solusi: Menggunakan analisis clustering untuk mengelompokkan pelanggan ke dalam segmen-segmen yang berbeda berdasarkan karakteristik atau perilaku mereka.
- Manfaat:
  - Memberikan insight berharga dalam menyusun strategi marketing yang lebih efektif.
  - Memenuhi kebutuhan setiap kelompok pelanggan dengan lebih baik.
  - Meningkatkan performa penjualan secara keseluruhan.

**Penjelasan**

Karakteristik atau perilaku pelanggan merupakan faktor penting yang perlu dipahami oleh perusahaan untuk menyusun strategi pemasaran yang efektif. Dengan memahami karakteristik atau perilaku pelanggan, perusahaan dapat memberikan treatment yang tepat untuk setiap individu berdasarkan permasalahan yang dihadapinya.

Salah satu pendekatan yang dapat digunakan untuk memahami karakteristik atau perilaku pelanggan adalah analisis clustering. Analisis clustering adalah metode untuk mengelompokkan data ke dalam segmen-segmen yang berbeda berdasarkan kesamaan karakteristik atau perilakunya.

Dengan menggunakan analisis clustering, perusahaan dapat mengelompokkan pelanggan ke dalam segmen-segmen yang berbeda, seperti:

- Segmen berdasarkan usia
- Segmen berdasarkan pendapatan
- Segmen berdasarkan minat
- Segmen berdasarkan perilaku pembelian

Dengan memahami karakteristik masing-masing segmen pelanggan, perusahaan dapat menyusun strategi pemasaran yang lebih efektif dan memenuhi kebutuhan setiap kelompok pelanggan dengan lebih baik.

Berikut adalah beberapa contoh manfaat yang dapat diperoleh perusahaan dengan memahami karakteristik pelanggan melalui analisis clustering:

- Perusahaan dapat menargetkan iklan atau promosi kepada kelompok pelanggan yang tepat.
- Perusahaan dapat mengembangkan produk atau layanan yang sesuai dengan kebutuhan pelanggan di masing-masing segmen.
- Perusahaan dapat menawarkan diskon atau promosi khusus kepada pelanggan di masing-masing segmen untuk meningkatkan loyalitas pelanggan.

Oleh karena itu, memahami karakteristik pelanggan melalui analisis clustering merupakan langkah penting dalam mengoptimalkan strategi penjualan dan mencapai keberhasilan jangka panjang bagi perusahaan.

### **Goal**

Tujuan dari analisis profil dan perilaku pelanggan dengan pendekatan clustering adalah untuk:

- **Memahami pelanggan dengan lebih baik**
  Dengan memahami karakteristik dan perilaku pelanggan, perusahaan dapat mengetahui kebutuhan dan keinginan mereka. Hal ini dapat membantu perusahaan untuk memberikan layanan yang lebih personal dan sesuai dengan kebutuhan pelanggan.

- **Menyediakan layanan yang lebih personal**
  Dengan memahami pelanggan dengan lebih baik, perusahaan dapat memberikan layanan yang lebih personal dan sesuai dengan kebutuhan mereka. Hal ini dapat meningkatkan kepuasan pelanggan dan mendorong mereka untuk melakukan pembelian kembali.

- **Meningkatkan performa penjualan**
  Dengan memahami karakteristik dan perilaku pelanggan, perusahaan dapat menyusun strategi pemasaran yang lebih efektif. Hal ini dapat membantu perusahaan untuk meningkatkan penjualan dan mencapai target bisnis.

- **Membangun hubungan yang kuat dengan pelanggan**
  Dengan memahami pelanggan dengan lebih baik, perusahaan dapat membangun hubungan yang kuat dengan pelanggan. Hal ini dapat meningkatkan loyalitas pelanggan dan mendorong mereka untuk menjadi pelanggan jangka panjang.

**Manfaat**

Berikut adalah beberapa manfaat yang dapat diperoleh perusahaan dengan melakukan analisis profil dan perilaku pelanggan dengan pendekatan clustering:

- **Menargetkan pelanggan yang tepat**
  Dengan memahami karakteristik dan perilaku pelanggan, perusahaan dapat menargetkan iklan atau promosi kepada kelompok pelanggan yang tepat. Hal ini dapat meningkatkan efektivitas strategi pemasaran dan meningkatkan penjualan.

- **Mengembangakan produk atau layanan yang sesuai dengan kebutuhan pelanggan**
  Dengan memahami karakteristik dan perilaku pelanggan, perusahaan dapat mengembangkan produk atau layanan yang sesuai dengan kebutuhan pelanggan. Hal ini dapat meningkatkan kepuasan pelanggan dan mendorong mereka untuk melakukan pembelian kembali.

- **Meningkatkan loyalitas pelanggan**
  Dengan memahami karakteristik dan perilaku pelanggan, perusahaan dapat memberikan layanan yang lebih personal dan sesuai dengan kebutuhan mereka. Hal ini dapat meningkatkan loyalitas pelanggan dan mendorong mereka untuk menjadi pelanggan jangka panjang.

- **Meningkatkan kepuasan pelanggan**
  Dengan memahami karakteristik dan perilaku pelanggan, perusahaan dapat memberikan layanan yang lebih personal dan sesuai dengan kebutuhan mereka. Hal ini dapat meningkatkan kepuasan pelanggan dan mendorong mereka untuk melakukan pembelian kembali.

Secara keseluruhan, analisis profil dan perilaku pelanggan dengan pendekatan clustering merupakan langkah penting yang dapat dilakukan perusahaan untuk meningkatkan efektivitas strategi pemasaran dan mencapai keberhasilan jangka panjang.

### **Objective**

- Membuat model mechine learning yang dapat mengelompokkan pelanggan ke dalam segmen-segmen yang berbeda berdasarkan karakteristik atau perilaku mereka.
- Mengekstraksi insight yang lebih mendalam tentang profil dan perilaku pelanggan.
- Menentukan strategi bisnis yang efektif dari hasil clustering.

## 📂 STAGE 1: Data Preparation

### Data Quality Asssessment

Dataset memiliki 2240 baris dan 30 fitur. Asesmen data dilakukan untuk memastikan bahwa data yang digunakan untuk analisis selanjutnya sudah siap dan sesuai dengan kebutuhan analisis. Hal yang dilakukan:

- Memeriksa missing value pada data
- Memeriksa duplikasi data
- Memeriksa tipe dan konsistensi nilai
- Memeriksa outlier atau data yang tidak biasa (anomali)

  Tabel 1 — Hasil Data Quality Assessment
  | Data Assessment | Finding | Cleaning |
  | --- | --- | --- |
  | Missing values | Tidak terdapat missing value | - |
  | Duplikat | Tidak terdapat duplikat data | - |
  | Fitur atau nilai yang tidak sesuai | TTipe data `Dt_Customer` sebaikkanya datettime | Mengubah tipe data menjadi datteime |
  | Anomali atau outlier | Secara keseluruhan fitur memiliki outlier. Terlihat juga fitur `Income` dan `Year_Birth` memiliki nilai yang ekstrim | Handling outlier menggunakan IQR. |

### Feature Engineering

Pada tahap feature engineering, dilakukan pembuatan feature baru berdasarkan feature yang sudah ada dengan tujuan untuk membuat analisis menjadi lebih insightful. Feature baru ini dapat mengungkap informasi tambahan atau menggabungkan beberapa fitur yang saling berhubungan untuk membentuk fitur yang lebih kuat.

Tabel 2 — Feature Engineering
| New Feature | Source |
| --- | --- |
| Membership Duration | 2023 - Dt_Customer |
| Age_Categories | age |
| Total_Children | Kidhome + Teenhome |
| Total_Transaction | NumDealsPurchases + NumWebPurchases + NumCatalogPurchases + NumStorePurchases |
| Total_Spending | MntCoke + MntFruits + MntMeatProducts + MntFishProducts + MntSweet |
| Total_Accepted_Campaign | AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 |
| CVR | Total_Transaction x NumWebVisitsMonth/100 |

## 📂 STAGE 2: Data Exploration

### Conversion Rate by Income, Spending, and Age

Pada tahap ini, dilakukan analisis konversi rate untuk mendapatkan wawasan tentang persentase pengunjung situs web dan tindakan yang dilakukan selama kunjungan mereka. Tujuan analisis ini adalah untuk melihat apakah tindakan pengunjung tersebut berujung pada transaksi pembelian atau tidak. Dengan demikian, perusahaan dapat memahami perilaku pengunjung dan mengidentifikasi peluang untuk meningkatkan tingkat konversi serta keberhasilan campaign pemasaran mereka.
![alt text](<https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%201%20—%20Plot%20Korelasi%20Conversion%20Rate%20(CVR)%20dengan%20Pendapatan%2C%20Total%20Pengeluaran%2C%20dan%20Usia.png>)

<h5 align="center">Gambar 1 — Plot Korelasi Conversion Rate (CVR) dengan Pendapatan, Total Pengeluaran, dan Usia</h5>

Terdapat temuan bahwa **pendapatan dan total spending memiliki korelasi positif yang signifikan terhadap tingkat konversi.** Hal ini menunjukkan bahwa **semakin tinggi pendapatan dan total spending seseorang, semakin besar kemungkinan mereka melakukan pembelian.** Faktor-faktor seperti kemampuan finansial yang lebih baik dan persepsi nilai yang tinggi terhadap produk dapat menjadi penyebab korelasi positif ini. Oleh karena itu, perusahaan dapat memanfaatkan temuan ini untuk mengoptimalkan strategi pemasaran mereka. Mereka dapat fokus pada target audiens dengan pendapatan dan total spending yang lebih tinggi, dengan tujuan meningkatkan peluang konversi dan keberhasilan marketing campaign secara keseluruhan. Di sisi lain, **fitur usia cenderung tidak memiliki korelasi yang signifikan terhadap tingkat konversi.** Hal ini berarti usia tidak menjadi faktor dominan yang mempengaruhi keputusan konsumen dalam melakukan konversi atau pembelian.

### Income and Total Spending

Analisis korelasi antara Income dan total spending penting dilakukan karena kedua fitur ini memiliki hubungan yang erat dalam konteks keuangan dan pengeluaran individu atau pelanggan. Dengan menganalisis korelasi antara kedua fitur ini, dapat dipahami sejauh mana tingkat pendapatan seseorang mempengaruhi pola pengeluaran mereka.
![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%202%20—%20Plot%20Korelasi%20Pendapatan%20dengan%20Total%20Pengeluaran.png)

<h5 align="center">Gambar 2 — Plot Korelasi Pendapatan dengan Total Pengeluaran</h5>

Hubungan korelasi positif yang kuat antara Income dan total spending menunjukkan **adanya hubungan yang signifikan antara tingkat pendapatan seseorang dengan pola pengeluaran mereka.** Hal ini mengindikasikan bahwa **semakin tinggi pendapatan seseorang, kemungkinan besar mereka juga memiliki pengeluaran yang lebih tinggi.** Dalam konteks bisnis, pemahaman ini dapat membantu perusahaan dalam mengenali segmen pelanggan yang memiliki potensi pembelian yang lebih tinggi dan merancang strategi pemasaran yang tepat untuk meningkatkan keterlibatan dan kepuasan pelanggan.

## 📂 STAGE 3: Data Modeling with K-Means Clustering

### Pre-processing

Sebelum melakukan data modeling, terdapat beberapa tahap pre-processing data yang perlu dilakukan yaitu:

- Fitur yang tidak diperlukan untuk model akan dihapus agar data lebih terfokus.
- Fitur kategorikal akan di-encoding agar dapat diolah oleh algoritma machine learning.
- Dilakukan standardisasi fitur untuk memastikan skala data seragam dan menghindari bias dalam model.

### Modeling

Setelah pre-processing data selesai, tahap berikutnya adalah menggunakan metode **Principal Component Analysis (PCA).** PCA digunakan untuk mengurangi dimensi data dengan mempertahankan informasi yang signifikan. Dengan mengurangi dimensi data, dapat mengoptimalkan kinerja model dan mengatasi masalah multicollinearity antara fitur. Selanjutnya, langkah penting dalam proses ini adalah menentukan jumlah cluster terbaik. Dalam analisis ini, **Distortion Score dan Elbow Method** digunakan untuk memilih jumlah cluster yang optimal. Berdasarkan hasil analisis, **jumlah cluster terbaik yang ditemukan adalah 4.**
![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%203%20—%20Plot%20Distortion%20Scoce%20Elbow.png)

<h5 align="center">Gambar 3 — Plot Distortion Scoce Elbow</h5>

Setelah menentukan jumlah cluster yang optimal, dilakukan **clustering menggunakan algoritma K-means.** Algoritma ini akan mengelompokkan data ke dalam cluster berdasarkan kesamaan fitur. Dengan melakukan clustering, dapat mengidentifikasi pola atau kelompok yang ada dalam data dan memahami karakteristik masing-masing cluster.
![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%204%20—%20Hasil%20Clustering%20menggunakan%20K-means.png)

<h5 align="center">Gambar 4 — Hasil Clustering menggunakan K-means</h5>

Dari plot hasil pemodelan dan pengelompokan data menggunakan metode clustering, terlihat bahwa **cluster-cluster yang terbentuk terpisah dengan baik** dan mengelompokkan data ke dalam kelompok yang berbeda-beda. Hal ini menunjukkan bahwa algoritma clustering yang digunakan berhasil dalam membedakan dan menggolongkan data berdasarkan karakteristik yang dimiliki.

### Evaluation

![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%205%20—%20Hasil%20Evaluasi.png)

<h5 align="center">Gambar 5 — Hasil Evaluasi</h5>

Evaluasi hasil model menggunakan **Silhouette Score memberikan rekomendasi bahwa jumlah cluster terbaik adalah 4.** Hal ini didasarkan pada fakta bahwa nilai Silhouette Score pada jumlah cluster tersebut adalah yang tertinggi, yaitu 0.535. Silhouette Score merupakan metrik evaluasi yang menggambarkan seberapa baik objek-objek dalam satu cluster berada dalam kumpulan data mereka sendiri dibandingkan dengan cluster lainnya. Semakin tinggi nilai Silhouette Score, semakin baik cluster-cluster tersebut terpisah.

## 📂 STAGE 4: Customer Personality Analysis

Customer Personality Analysis bertujuan untuk **memahami perbedaan dan kesamaan antara cluster-cluster tersebut, serta mengidentifikasi karakteristik unik yang mungkin dimiliki oleh setiap kelompok.** Dengan pemahaman yang lebih mendalam tentang karakteristik antar cluster, perusahaan dapat mengambil tindakan yang lebih tepat dan mengarahkan strategi bisnis yang lebih spesifik untuk setiap kelompok pelanggan.
![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%206%20—%20Plot%20Pendapatan%20dan%20Total%20Pengeluaran%20Berdasarkan%20Cluster.png)

<h5 align="center">Gambar 6 — Plot Pendapatan dan Total Pengeluaran Berdasarkan Cluster</h5>

Berdasarkan plot korelasi antara pendapatan (Income) dan total pengeluaran (Total Spending), terlihat bahwa terdapat pembentukan cluster atau kelompok yang dapat dibedakan. Dalam hal ini, **cluster 1 dan 3 cenderung berada dalam satu kelompok yang menunjukkan adanya persamaan dan perbedaan karakteristik di antara kedua cluster tersebut.** Ketika dua cluster berada dalam satu kelompok, hal ini mengindikasikan bahwa terdapat kemiripan atau keterkaitan dalam pola pendapatan dan pengeluaran di antara anggota-anggota cluster tersebut. Secara visual, terlihat bahwa **kedua cluster tersebut mungkin memiliki tingkat pendapatan dan pengeluaran yang relatif mirip atau memiliki tren yang serupa.**
![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%207.png)

<h5 align="center">Gambar 7 — Plot Karakteristik Mayoritas/Rata-rata Total Transaksi, Pengeluaran, Pendapatan, Recency, dan Conversion Rate Berdasarkan Cluster</h5>

Berdasarkan hasil analisis yang lebih mendalam dapat diketahui karakteristik rata-rata/mayoritas dari setiap cluster berdasarkan pola transaksi pelanggan dan dapat dikelompokkan berdasarkan beberapa kategori.

- Cluster 0

  - Angka transaksi dan spending teredndah yaitu mayoritas 7 transaksi dan Rp.58.000 perbulan
  - Pendapatan terendah, mayoritas Rp.33.297.500/tahun
  - CConversion terendah, yaitu 1%
  - Kategori : **"Low-Transaction Low-Spending Group" - Low Customer**

- Cluster 1

  - Angka transaksi dan spending terendah yaitu mayoritas hanya 7 transaksi dan Rp.58.000/bulan
  - Pendapatan terendah, mayoritas Rp.33.297.500/tahun
  - Conversion terendah, yaitu 1%
  - Kategori : **"High-Transaction High-Spending Group" - High Customer A**

- Cluster 2

  - Angka transaksi dan spending cukup tinggi yaitu mayoritas 20 transaksi dan Rp.1.040.000/bulan
  - Pendapatan tertinggi, mayoritas Rp.71.488.000/tahun
  - Conversion rate tertinggi, yaitu 8%
  - Kategori : **"High-Income High-Conversion Group" - High Customer B**

- Cluster 3
  - Angka transaksi dan spending sedang yaitu mayoritas 17 transaksi dan Rp.434.000/bulan
  - Pendapatan cukup sedang, mayoritas Rp.52.597.000/tahun
  - Conversion rate cukup sedang, yaitu 3%
  - Kategori : **"Moderate-Transaction Moderate-Spending Group" - Moderate Customer**

Analisis distribusi beberapa fitur masing-masing cluster dilakukan juga dilakukan untuk mendapatkan wawasan yang lebih dalam. Melalui analisis ini, ditemukan beberapa insight menarik yang dapat memberikan pemahaman yang lebih baik tentang perilaku pengguna dalam setiap cluster, khususnya terkait kunjungan website dan respon terhadap campaign.

![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%208.png)

<h5 align="center">Gambar 8 — Plot Distribusi Berdasarkan Cluster</h5>

Berikut temuan yang menarik:

- Low Customer (Cluster 1) yang memiliki distribusi jumlah kunjungan website yang tinggi, namun memiliki total acceptance campaign yang rendah. Ini menunjukkan bahwa kelompok ini sangat sering mengunjungi website perusahaan, tetapi tidak sepenuhnya menyadari atau tidak responsif terhadap campaign yang ditawarkan. Mengingat kelompok ini memiliki populasi yang paling banyak, perusahaan perlu mengembangkan strategi yang tepat untuk menarik perhatian dan meningkatkan keterlibatan mereka.
- Cluster yang paling banyak merespon campaign adalah High Customer A (Cluster 0) dengan tingkat konversi yang sedang. Ini menunjukkan bahwa mayoritas pelanggan dalam kelompok ini sangat responsif terhadap campaign yang ditawarkan oleh perusahaan. Hal ini dapat menjadi kesempatan yang baik untuk meningkatkan interaksi dan pembelian dari kelompok ini dengan meluncurkan campaign yang lebih menarik dan relevan sesuai dengan preferensi mereka.
- High Customer B (Cluster 2), mayoritas pelanggannya tidak terlalu sering mengunjungi website perusahaan, namun memiliki distribusi konversi rate yang lebih tinggi dengan respon campaign yang sedang. Fenomena ini menunjukkan bahwa kelompok ini memiliki kecenderungan pengeluaran yang tinggi dan cenderung merespons positif terhadap campaign yang ditawarkan, meskipun mereka tidak begitu aktif dalam kunjungan ke website. Perusahaan dapat memanfaatkan informasi ini dengan mengoptimalkan saluran komunikasi lain seperti email, media sosial, atau platform online lainnya untuk efektif menjangkau kelompok ini.

![alt text](https://github.com/imalfunadam/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/assets/Gambar%209.png)

<h5 align="center">Gambar 9 — Plot Presentase Populasi Cluster</h5>

Berdasarkan persentase populasi masing-masing cluster, ditemukan bahwa **50.22% dari keseluruhan pelanggan termasuk dalam kelompok Low Customer (Cluster 0).** Meskipun kelompok ini memiliki angka transaksi dan pengeluaran yang rendah, namun karena populasi mereka yang besar. Perusahaan dapat fokus untuk menarik perhatian mereka. Sedangkan populasi **High Customer A (Cluster 1) dan B (Cluster 2) cenderung rendah,** namun memiliki potensi transaksi dan spending yang tinggi. Perusahaan dapat mempertimbangkan strategi pemasaran yang lebih personal dan eksklusif untuk menarik minat mereka.

## 📂 STAGE 5: Business Recommendation

Berdasarkan analisis yang telah dilakukan, dapat diidentifikasi personalitas atau karakteristik pelanggan berdasarkan cluster yang terbentuk. Mengetahui karakteristik ini sangat berharga dalam merancang strategi pemasaran yang lebih efektif. Dengan memahami preferensi, kebutuhan, dan perilaku konsumen dalam setiap cluster, perusahaan dapat menghasilkan campaign yang lebih relevan dan menarik bagi setiap kelompok pelanggan.

### High Customer A

Summary:

- Populasi 12.61%.
- High-Transaction High-Spending Group.
- Paling responsif terhadap campaign, dengan tingkat kunjungan website dan konversi ke pembelian sedang.

  Rekomendasi:<br>

Mengingat kelompok High Customer A cenderung memiliki total transaksi dan total spending yang tinggi, perusahaan dapat memberikan penawaran khusus dan insentif tambahan untuk mendorong pelanggan melakukan pembelian secara terus-menerus. Perusahaan dapat menerapkan program diskon eksklusif, hadiah loyalitas, atau akses ke produk atau layanan khusus untuk kelompok ini.
Perusahaan dapat meningkatkan kualitas pengalaman pengguna dalam berselancar di website, mengingat tingkat kunjungan website yang sedang. Perusahaan dapat memastikan tampilan yang menarik, customer journey yang efisien, dan lain sebagainya.
Mengingat kelompok High Customer A sangat responsif terhadap campaign, memanfaatkan kepuasan mereka dengan memperkenalkan program referral dapat menjadi strategi yang efektif. Memberikan insentif kepada pelanggan untuk merekomendasikan produk atau layanan perusahaan kepada teman dan keluarga dapat membantu dalam memperluas jangkauan dan memperoleh pelanggan baru.
Perusaan dapat mingirimkan pesan yang dipersonalisasikan seperti info promo atau diskon berdasarkan preferensi kelompok ini. Hal ini dilakukan untuk menjaga loyalitas pelanggan.

### High Customer B

Summary :

Populasi 13.42%.
High-Income High-Conversion Group.
Sama seperti High Customer B dalam segi income dan total spending, namun memiliki income paling tinggi.
Tingkat konversi paling tinggi, respon terhadap campaign relatif sedang, kurang mengunjungi website secara aktif.
Rekomendasi:

Sama halnya dengan High Customer A, perusahaan dapat memberikan penawaran khusus seperti diskon, program loyalti, dan sebagainya agar pelanggan selalu tertarik untuk berbelanja terus menerus.
Mengingat kelompok ini kurang aktif dalam kunjungan website, perusahaan dapat memanfaatkan saluran komunikasi alternatif untuk campaign seperti email, pesan teks, atau media sosial. Hal ini dapat membantu meningkatkan interaksi dan kesadaran pelanggan.
Untuk meningkatkan respon pelanggan terhadap campaign, perusahaan dapat memberikan campaign-campaign yang tertarget sesuai dengan preverensi dan kebutuhan pelanggan.
Mengingat pelanggan dalam kelompok High Customer B memiliki tingkat konversi yang tinggi, perusahaan dapat mempertimbangkan untuk meluncurkan program loyalitas yang memberikan insentif tambahan, penghargaan khusus, atau akses ke acara atau produk eksklusif dapat memperkuat loyalitas pelanggan.

### Moderate Customer

Summmary:

- Populasi 23.75%.
- Moderate-Transaction Moderate-Spending Group
- Tingkat konversi, kunjungan website dan respon terhadap campaign relatif sedang.<br>

  Rekomendasi:

- Perusahaan dapat memberikan penawaran khusus dan diskon untuk mendorong pembelian lebih lanjut. Hal ini dapat memberikan insentif tambahan kepada pelanggan dalam kelompok ini untuk memilih produk atau layanan perusahaan dibandingkan dengan pesaing.
- Perusahaan dapat mengirim pesan yang relevan dan menarik kepada pelanggan untuk melakukan transaksi.
- Memastikan pengalaman pengguna yang baik saat mengunjungi website atau berinteraksi dengan produk atau layanan perusahaan.
- Membangun program hadiah atau loyalitas dapat membantu memperkuat keterikatan pelanggan. Seperti dengan memberikan poin, penghargaan, atau manfaat khusus kepada pelanggan setia, perusahaan dapat mendorong mereka untuk terus memilih dan membeli produk atau layanan perusahaan.

### Low Customer

Summary:

- Populasi 50.22%, pelanggan didominasi oleh kategori ini.
- Low-Transaction Low-Spending Group.
- Tingkat konversi paling rendah, cenderung tidak merespon campaign, namun kategori ini paling sering mengunjungi website.<br>
  Rekomendasi:

- Mengingat kelompok Low Customer sering mengunjungi website, perusahaan dapat memanfaatkan informasi kunjungan website untuk menyajikan konten yang personalisasi dan penawaran khusus yang sesuai dengan minat dan preferensi mereka.
- Perusahaan dapat melakukan retargeting campaign dengan mengingatkan pelanggan dalam kelompok ini tentang produk atau layanan yang mereka telah kunjungi di website. Dengan menampilkan iklan yang disesuaikan di berbagai platform digital yang mereka gunakan, perusahaan dapat membangun kesadaran dan mendorong mereka untuk melanjutkan proses pembelian.
- Mengingat kelompok Low Customer memiliki tingkat konversi yang rendah dan cenderung tidak merespon campaign dengan baik, perusahaan dapat menggunakan strategi konten yang lebih fokus pada edukasi dan informasi (softselling). Memberikan konten yang memberikan nilai tambah, memberikan solusi untuk masalah atau kebutuhan pelanggan, dan membantu mereka membuat keputusan yang lebih informatif dapat meningkatkan keterlibatan dan kepercayaan pelanggan dalam kelompok ini.
