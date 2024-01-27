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
