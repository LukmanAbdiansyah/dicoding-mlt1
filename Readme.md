# Klasifikas Fasies Menggunakan Machine Learning - Lukman Abdiansyah
## Domain Proyek
Klasifikasi facies memainkan peran penting dalam memahami lingkungan deposisi dan sifat-sifat reservoir dari formasi batuan bawah permukaan. Secara tradisional, klasifikasi facies dilakukan secara manual oleh ahli geologi berdasarkan keahlian mereka dan analisis dari data sumur, sampel inti, dan data seismik. Namun, proses ini memakan waktu lama, subjektif, dan rentan terhadap kesalahan manusia.

Tujuan dari proyek ini adalah untuk mengotomatisasi proses klasifikasi facies menggunakan teknik pembelajaran mesin. Dengan memanfaatkan data sumur, sampel inti, dan mungkin juga data seismik, tujuannya adalah untuk melatih model yang dapat memprediksi jenis facies dengan akurat berdasarkan atribut geologi seperti litologi, porositas, permeabilitas, dan komposisi mineral. Hal ini akan memungkinkan klasifikasi facies yang lebih cepat dan lebih objektif, yang akan menghasilkan karakterisasi reservoir yang lebih baik dan pengambilan keputusan yang lebih baik dalam industri minyak dan gas.

Dataset untuk penelitian ini berasal dari Hugoton dan Panoma Fields di Amerika Utara yang digunakan dalam latihan kelas di The University of Kansas. Ini terdiri dari data log dari sembilan sumur. digunakan data log ini untuk melatih pengklasifikasi dalam memprediksi kelompok fasies.

## Alasan penting yang mendasari proyek ini:
- Efisiensi dan Waktu: Proses klasifikasi facies secara manual oleh ahli geologi memakan waktu yang lama dan membutuhkan upaya yang intensif. Dengan mengembangkan model pembelajaran mesin yang dapat mengotomatisasi proses ini, akan terjadi peningkatan efisiensi dan penghematan waktu yang signifikan. Ini akan memungkinkan para profesional di industri minyak dan gas untuk fokus pada analisis yang lebih mendalam dan pengambilan keputusan yang lebih baik.

- Objektivitas: Klasifikasi facies manual rentan terhadap subjektivitas individu, di mana ahli geologi yang berbeda mungkin memberikan interpretasi yang berbeda untuk data yang sama. Dengan menggunakan metode pembelajaran mesin yang berdasarkan pada pola dan fitur data, akan ada peningkatan objektivitas dalam klasifikasi facies. Hal ini mengurangi variabilitas interpretasi antara ahli geologi dan menghasilkan hasil yang lebih konsisten.

- Skalabilitas: Dalam industri minyak dan gas, terdapat volume data yang besar dan kompleks yang harus dianalisis. Dengan menggunakan pembelajaran mesin, kita dapat mengatasi skalabilitas masalah ini dengan melibatkan model untuk mengklasifikasikan data dalam skala yang lebih besar dan lebih cepat daripada yang dapat dilakukan oleh ahli manusia.

- Peningkatan Keakuratan: Manusia rentan terhadap kesalahan dan kelelahan saat menganalisis data. Dengan menggunakan model pembelajaran mesin yang terlatih dengan benar, dapat diharapkan peningkatan keakuratan dalam klasifikasi facies. Ini memberikan informasi yang lebih dapat diandalkan dalam karakterisasi reservoir, yang pada akhirnya dapat mengarah pada pengambilan keputusan yang lebih baik dalam eksploitasi sumber daya alam.

- Pembelajaran dari Data: Dengan menggunakan teknik pembelajaran mesin, model dapat mempelajari pola-pola yang kompleks dan relasi antara atribut geologi yang berkontribusi pada klasifikasi facies. Hal ini dapat menghasilkan pemahaman yang lebih dalam tentang proses geologi dan faktor-faktor yang mempengaruhi pembentukan facies. Pengetahuan ini dapat digunakan untuk meningkatkan pemodelan dan prediksi lingkungan deposisi serta memahami sifat reservoir dengan lebih baik.
## Referensi Terkait
- [Facies classification with different machine learning algorithm – An efficient artificial intelligence technique for improved classification](https://www.researchgate.net/publication/337166562_Facies_classification_with_different_machine_learning_algorithm_-_An_efficient_artificial_intelligence_technique_for_improved_classification)

# Business Understanding

Pengembangan model pembelajaran mesin untuk klasifikasi facies dalam industri minyak dan gas akan meningkatkan efisiensi operasional, pengambilan keputusan yang lebih baik, dan memberikan keunggulan kompetitif melalui informasi yang lebih handal tentang karakteristik reservoir, penggunaan sumber daya yang optimal, dan peningkatan kemampuan eksplorasi.

## Problem Statements
Berdasarkan studi kasus:

- Tidak ada pengotomatisasian: Tidak ada sistem otomatis yang digunakan untuk melakukan klasifikasi facies menggunakan machine learning di industri minyak dan gas, menyebabkan ketergantungan pada metode manual yang memakan waktu lama dan rentan terhadap kesalahan manusia.
- Keterbatasan metode manual: Metode klasifikasi facies manual yang saat ini digunakan cenderung subjektif dan membutuhkan penilaian ahli geologi, yang dapat menghasilkan hasil yang bervariasi antara ahli yang berbeda. Hal ini mengurangi objektivitas dan konsistensi dalam proses klasifikasi facies.
- Integrasi data: Tantangan dalam mengintegrasikan dan mengolah data sumur, data sampel core, dan data seismik dari berbagai sumber dan format menjadi satu dataset yang terpadu dan siap untuk digunakan dalam model pembelajaran mesin. Integrasi data yang tidak efisien dapat menyulitkan proses klasifikasi facies.
- Kurangnya akurasi dan efisiensi: Metode manual yang digunakan saat ini mungkin tidak menghasilkan prediksi facies yang akurat dan membutuhkan waktu yang lama untuk menganalisis data yang kompleks. Dibutuhkan solusi yang lebih akurat dan efisien untuk mengklasifikasikan facies berdasarkan atribut geologi yang relevan.
- Kesulitan dalam generalisasi: Model yang dikembangkan harus mampu generalisasi dengan baik dan dapat digunakan untuk memprediksi facies dalam situasi yang belum pernah dilihat sebelumnya. Kemampuan model untuk menghadapi variasi kondisi geologi dan kualitas data yang berbeda juga menjadi tantangan yang harus diatasi.

## Goals
Berdasarkan problem statements:
- Mengembangkan sistem otomatis klasifikasi facies menggunakan machine learning.
- Meningkatkan akurasi dan konsistensi klasifikasi facies.
- Meningkatkan efisiensi operasional dengan otomatisasi klasifikasi facies.
- Memungkinkan pengambilan keputusan berbasis data.
- Meningkatkan pemahaman tentang lingkungan geologi.
- Mengoptimalkan penggunaan sumber daya.

## Solution Statements
Berdasarkan Goals maka:
- Mencari pengetahuan yang ada pada data proses EDA
- Pengembangan model pembelajaran mesin untuk klasifikasi facies secara otomatis.
- Pelatihan dan validasi model menggunakan dataset yang diklasifikasikan manual.
- Implementasi dan validasi model.
- evaluasi pembanding model akan menggunakan MSE atau Mean Squared Error

# Data Understanding
Dataset yang digunakan berdasal dari kaggle dengan nama [Well log Dataset](https://www.kaggle.com/datasets/imeintanis/well-log-facies-dataset) yang memiliki jumlah data 1359 baris dan 21 kolom. Dataset merupakan hasil dari penggabungan dari website [Gadget360](ttps://www.kgs.ku.edu/Hugoton/TypeLogs/index.html). Dataset berisikan tentang 5 pengukuran log wireline, dua indikator dan label fasies dalam interval setengah kaki.

## Sample Data
Ada pun sample data yang bisa dilihat seperti dibawah ini.
|   | Facies | Formation | Well Name | Depth  | GR     | ILD_log10 | DeltaPHI | PHIND  | PE    | NM_M | RELPOS |
|---|--------|-----------|-----------|--------|--------|-----------|----------|--------|-------|------|--------|
| 0 | 3      | A1 SH     | SHRIMPLIN | 2793.0 | 77.450 | 0.664     | 9.900    | 11.915 | 4.600 | 1    | 1.000  |
| 1 | 3      | A1 SH     | SHRIMPLIN | 2793.5 | 78.260 | 0.661     | 14.200   | 12.565 | 4.100 | 1    | 0.979  |
| 2 | 3      | A1 SH     | SHRIMPLIN | 2794.0 | 79.050 | 0.658     | 14.800   | 13.050 | 3.600 | 1    | 0.957  |
| 3 | 3      | A1 SH     | SHRIMPLIN | 2794.5 | 86.100 | 0.655     | 13.900   | 13.115 | 3.500 | 1    | 0.936  |
| 4 | 3      | A1 SH     | SHRIMPLIN | 2795.0 | 74.580 | 0.647     | 13.500   | 13.300 | 3.400 | 1    | 0.915  |

Tabel 1 : Sample Data
Untuk sample data bisa dilihat seperti pada tabel 1.

## Data Variabel
Variabel fitur termasuk lima dari pengukuran log wireline dan dua variabel geologi yang berasal dari pengetahuan geologi. Ketujuh variabel tersebut adalah:

- GR: Alat log wireline ini mengukur emisi gamma dari formasi. Indeks yang tinggi mengindikasikan konten shale.
- ILD_log10: Ini adalah pengukuran resistivitas yang dapat digunakan untuk identifikasi konten fluida reservoir.
- PE: Log efek fotolistrik dapat digunakan untuk identifikasi litologi (kandungan mineral batuan).
- DeltaPHI: Phi adalah indeks porositas dalam petrofisika. Untuk mengukur porositas, terdapat beberapa metode seperti neutron dan densitas.
- PNHIND: Rata-rata dari log neutron dan densitas.
- NM_M: Indikator nonmarin-marine.
- RELPOS: posisi relatif.

Dengan sembilan kelas fasies tersebut yaitu:

(SS) Nonmarine sandstone
(CSiS) Nonmarine coarse siltstone
(FSiS) Nonmarine fine siltstone
(SiSH) Marine siltstone and shale
(MS) Mudstone (limestone)
(WS) Wackestone (limestone)
(D) Dolomites
(PS) Packstone-grainstone (limestone)
(BS) Phylloid-algal bafflestone (limestone)

Tabel berikut mencantumkan fasies, label singkatannya, dan perkiraan tetangganya.

| Facies | Label | Adjacent Facies |
|--------|-------|-----------------|
| 1      | SS    | 2               |
| 2      | CSiS  | 1,3             |
| 3      | FSiS  | 2               |
| 4      | SiSh  | 5               |
| 5      | MS    | 4,6             |
| 6      | WS    | 5,7             |
| 7      | D     | 6,8             |
| 8      | PS    | 6,7,9           |
| 9      | BS    | 7,8             |

Penomoran ini akan digunakan sebagai label untuk setiap fasies di dataset. semakin tinggi angkanya maka semakin baik pula kualitas (permebailitas dan porositas) dari batuan

## Langkah-Langkah dalam pendalaman Data Understanding.
- Melakukan tahapan EDA seperti mendeskripsikan variabel, mencari outliers, Univariate hingga Multi-variate analysis.
- Untuk Mendeskripsikan variabel bisa menggunakan library pandas dan fungsi .describe() dan .info()
- Melakukan visualisasi data pada saat melakukan analisa data.
- Mengecek data missing value dan membersihkan data missing value dengan membuat simple logic program
- Menggunakan histogram untuk melihat penyebaran data dengan library pandas fungsi .hist()
- Mencari Keterkaitaan antar fitur dengan correlation matrix dengan fungsi pandas dan visualisasi heatmap dengan seaborn

missing value checking tidak dilakukan karena dalam melakukan analisa well log data missing value memang digunakan, hal ini dilakukan karena pada beberapa zona kedalaman tertentu terdapat kandungan mineral batuan yang dominan.

### Hasil Visualisasi tahapan EDA
| #  | Column    | Non-Null Count | Dtype   |
|----|-----------|----------------|---------|
| 0  | Facies    | 3232 non-null  | int64   |
| 1  | Formation | 3232 non-null  | object  |
| 2  | Well Name | 3232 non-null  | object  |
| 3  | Depth     | 3232 non-null  | float64 |
| 4  | GR        | 3232 non-null  | float64 |
| 5  | ILD_log10 | 3232 non-null  | float64 |
| 6  | DeltaPHI  | 3232 non-null  | float64 |
| 7  | PHIND     | 3232 non-null  | float64 |
| 8  | PE        | 3232 non-null  | float64 |
| 9  | NM_M      | 3232 non-null  | int64   |
| 10 | RELPOS    | 3232 non-null  | float64 |

Tabel 2 : melihat kolom dan tipe data pada dataset
Pada gambar dapat dilihat pada data memiliki kolom 9 kolom numerik atau angka sedangkan sisanya non-numerik atau kategorikal

| Column       | NaN Value Count |
|--------------|-----------------|
| Facies       | 0               |
| Formation    | 0               |
| Well Name    | 0               |
| Depth        | 0               |
| GR           | 0               |
| ILD_log10    | 0               |
| DeltaPHI     | 0               |
| PHIND        | 0               |
| PE           | 0               |
| NM_M         | 0               |
| RELPOS       | 0               |
| FaciesLabels | 0               |

Tabel 3: pengecekan NaN 
Pada tabel 2 dan 3 telah di lakukan pengecekan apakah ada missing value pada data kategorikal dan nilai null/0 pada data numerikal. Dan hasilnya tidak ada data yang missing dan null

![top10](https://user-images.githubusercontent.com/60514291/239754897-6b95dec0-3029-47af-a5c7-9042d6cd1699.png)
Gambar 1 : urutan fasies terbanyak pada formasi hugoton dan panoma
Pada dataset yang memiliki data ponsel paling banyak adalah Ponsel bermerk Intex dan motorola berada di urutan ke 10.

![hist](https://user-images.githubusercontent.com/60514291/239754920-9debd797-1c19-4df3-9e67-45c0942e55d6.png)
Gambar 2: Persebaran Dataset

Pada gambar diatas ada beberapa kesimpulan yaitu :
*   Pada kolom price, ponsel dengan harga mahal cenderung lebih sedikit daripada ponsel murah 
*   Histogram pada 'price' lebih miring ke kanan atau distribusi miring kanan
*   Pada pada, ponsel paling banyak berada pada harga di bawah 10000

![image](https://user-images.githubusercontent.com/60514291/239755894-c780e92e-ef5f-473c-9713-f505c76788dc.png)
Gambar 4 : Multivariate Analysis OS with Average Price

Pada gambar diatas , OS android memiliki rata-rata harga paling tinggi

![image](https://user-images.githubusercontent.com/60514291/239755998-ea9c9ca4-dbbf-425b-8974-26206fe3b62a.png)
Gambar 5 : Multivariate Analysis gps with Average Price

Harga Handphone rata-rata lebih mahal apabila memiliki GPS.

![image](https://user-images.githubusercontent.com/60514291/239756106-4606ae46-2031-452c-9650-500496067d69.png)
Gambar 6 : Multivariate Analysis 4g with Average Price
Pada gambar diatas, harga rata-rata ponsel lebih mahal apabila memiliki sinyal 4g

![pairplotall](https://user-images.githubusercontent.com/60514291/239754965-9193e343-4d0e-4fd9-8336-68c0e95e5129.png)
Gambar 7 : Korelasi antar price dengan fitur yang lain secara pairplot

Walaupun terlihat acak, namun apabila diperhatikan tiap fitur numerik memiliki korelasi positif terhadap 'price', semakin ke kanan harga pun semakin naik

![corrmat](https://user-images.githubusercontent.com/60514291/239755032-44b5c830-e116-46ae-b587-a76b54ae1565.png)
Gambar 8:Correlation Matrix
Untuk tiap fitur memiliki korelasi positif namun tidak terlalu tinggi, dimana fitur yang paling berpengaruh yaitu ram,internal dan resolusi layar.
Sedangkan pada fitur baterai, memiliki korelasi yang paling kecil diantara fitur yang lainnya

### Result EDA
Sejauh tahap yang dilakukan, seperti menghapus outliers maka tersisa data sebanyak 938 baris data yang sudah bersih dari outliers.

# Data Preparation
## Proses yang dilakukan
-   Menerapkan One Hot Encoding pada data Categorical dengan menggunakan pandas library pada fungsi pd.get_dummies()
-   Menerapkan PCA pada data yang memiliki kesamaan arti dan nilai dengan library sklearn PCA
-   Membagi data set antara training dan testing dengan library sklearn dengan fungsi train_test_split() dengan perbandingan 80:20 sehingga memiliki 750 data train dan 188 data testing
-   Menerapan Standard Scaler pada data numerikal dengan library sklearn dengan fungsi StandarScaler

## Alasan Pengunaan
- One Hot encoding digunakan untuk mengubah variabel kategorikal menjadi representasi numerik yang dapat digunakan dalam model machine learning. serta dapat meningkatan performa model dalam melakukan prediksi karena hanya menggunakan nilai biner. Hal ini sangat berguna karena algoritma machine learning umumnya membutuhkan data numerik sebagai input.
- Penggunaan PCA digunakan untuk mengurangi dimensi dari dataset yang memiliki fitur yang sangat banyak dimana data yang dimiliki mengandung informasi yang redundan atau sama antar banyak fitur yang serupa sehingga cukup dijadikan satu dimensi saja. Hal ini juga bertujuan untuk meningkatkan performa model. Karena terlalu banyak fitur dapat mengakibatkan masalah dalam pemodelan seperti overfitting atau kompleksitas yang berlebihan.Dengan menggunakan PCA, kita dapat mengurangi dimensi fitur-fitur tersebut menjadi sejumlah komponen utama yang paling mengandung informasi.
- Membagi dataset kedalam bentuk training dan testing adalah agar model dapat di evaluasi nantinya. Selain itu, pembagian ini dapat juga untuk mendeteksi apakah model mengalami overfitting jika model memiliki performa yang sangat baik pada data pelatihan tetapi performa yang buruk pada data pengujian.
- Standar Scaler digunakan untuk menormalkan atau menskalakan data numerik dalam skala yang sama. Menggunakan data dengan skala yang berbeda dapat mengganggu performa model yang menggunakan metrik jarak seperti algoritma SVM. Sehingga, model dapat menormalkan data numerik sehingga memiliki rata-rata nol dan standar deviasi satu. Ini membantu dalam menghilangkan perbedaan skala dan memastikan bahwa setiap fitur diperlakukan secara adil dan tidak mendominasi pengaruhnya terhadap hasil model.

# Modeling
Pada proyek ini akan menggunakan model SVR sesuai dengan referensi sebelumnya dan juga menggunakan algoritma Huber Regessor berdasarkan hasil dari running data menggunakan library pycaret.

|   Model   |           Algorithm            |    MAE    |       MSE       |     RMSE    |     R2     |   RMSLE   |   MAPE   | TT (Sec) |
|-----------|-------------------------------|-----------|-----------------|-------------|------------|-----------|----------|----------|
|   huber   |         Huber Regressor        | 4926.8217 | 109476114.4566  |  10220.0214 |   0.3062   |   0.5269  |  0.4096  |  0.4150  |
|    knn    |    K Neighbors Regressor       | 6014.3109 | 131721542.8240  |  11249.8804 |   0.1568   |   0.6684  |  0.6604  |  0.4230  |
|  lightgbm | Light Gradient Boosting Machine| 7512.7058 | 149481917.6927  |  12030.7421 |   0.0284   |   0.8183  |  1.0080  |  0.4430  |
|     et    |    Extra Trees Regressor       | 7575.7432 | 155341761.4113  |  12271.8095 |  -0.0116   |   0.8261  |  1.0060  |  0.5900  |
|    ada    |     AdaBoost Regressor         | 7227.6931 | 156073006.1999  |  12294.8434 |  -0.0144   |   0.7995  |  0.9053  |  0.3480  |
|  xgboost  |   Extreme Gradient Boosting     | 7580.2099 | 155719354.5086  |  12288.7833 |  -0.0148   |   0.8276  |  1.0065  |  0.4730  |
|    omp    |   Orthogonal Matching Pursuit   | 7582.6384 | 155842850.6352  |  12293.4675 |  -0.0156   |   0.8280  |  1.0070  |  0.6200  |
|    llar   |  Lasso Least Angle Regression  | 7582.6383 | 155842848.6985  |  12293.4674 |  -0.0156   |   0.8280  |  1.0070  |  0.5040  |
|    lar    |      Least Angle Regression    | 7582.6384 | 155842850.6352  |  12293.4675 |  -0.0156   |   0.8280  |  1.0070  |  0.7020  |
|    br     |         Bayesian Ridge         | 7582.6385 | 155842851.2002  |  12293.4675 |  -0.0156   |   0.8280  |  1.0070  |  0.7680  |
|    lr     |       Linear Regression        | 7582.6384 | 155842850.6352  |  12293.4675 |  -0.0156   |   0.8280  |  1.0070  |  0.5620  |
|    en     |          Elastic Net           | 7582.6377 | 155842828.1476  |  12293.4666 |  -0.0156   |   0.8280  |  1.0070  |  0.5610  |
|   ridge   |        Ridge Regression        | 7582.6384 | 155842850.5385  |  12293.4675 |  -0.0156   |   0.8280  |  1.0070  |  0.3050  |
|   lasso   |        Lasso Regression        | 7582.6384 | 155842848.8967  |  12293.4674 |  -0.0156   |   0.8280  |  1.0070  |  0.3680  |
|   dummy   |        Dummy Regressor         | 7582.6384 | 155842850.6352  |  12293.4675 |  -0.0156   |   0.8280  |  1.0070  |  0.3110  |
|    rf     |   Random Forest Regressor      | 7614.6345 | 156392402.3966  |  12313.1076 |  -0.0184   |   0.8302  |  1.0131  |  0.3910  |
|    gbr    |  Gradient Boosting Regressor   | 7609.6841 | 156583555.8053  |  12318.9634 |  -0.0191   |   0.8294  |  1.0118  |  0.4490  |
|    dt     |   Decision Tree Regressor     | 7656.8606 | 158612810.0040  |  12391.4475 |  -0.0305   |   0.8327  |  1.0205  |  0.3230  |
|    par    | Passive Aggressive Regressor   | 7442.2487 | 156193958.7692  |  12125.3283 |  -0.0439   |   1.1905  |  0.7329  |  0.3170  |

Tabel 5 : Algorithm Reference

Pada saat menjalankan library yang ada di pycaret, huber regressor memiliki skor mse paling rendah daripada algoritma lain.

## Tahapan yang dilakukan
- Melatih Model dengan data training dengan menggunakan algoritma Huber regressor dan SVR
- Pada tahap training ini akan dilakukan pengujian model dengan parameter default yang ada pada library
- Melakukan pengujian dengan data training
- Kemudian, lanjut pengujian dengan data testing
- Pengukuran menggunakan metriks MSE,MAE,RMSE dan R2 dengan menggunakan lirary sklearn.
- Melihat hasil performa model antara hasil training dan testing
- Kemudian tingkatkan performa model dengan menerapkan grid search atau hyper parameter pada model.
- Untuk hyper param yang digunakan pada Huber Regressor adalah param_grid = { 'epsilon': [1.0, 1.5, 2.0],'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [100, 200, 300]}
- Pada SVR param_grid = {'kernel': ['linear', 'rbf'],'C': [0.1, 1, 10],'epsilon': [0.1, 0.2, 0.3]}
- Dari pengujian hyperparam , mendapatkan param yang terbaik yaitu {'alpha': 0.01, 'epsilon': 2.0, 'max_iter': 200} pada model huber
- Sedangkan SVR {'C': 10, 'epsilon': 0.1, 'kernel': 'linear'}

### Hasil Running Model
|       | Huber_MAE |  SVR_MAE  |    Huber_MSE    |    SVR_MSE     |   Huber_RMSE   |   SVR_RMSE    |   Huber_R2   |   SVR_R2   |
|-------|-----------|-----------|-----------------|----------------|----------------|---------------|--------------|------------|
| train | 1905.190  | 2584.369  | 7774691.441     | 13188739.278   | 2788.313       | 3631.630      | 0.391        | -0.034     |
| test  | 2189.424  | 2907.168  | 11186580.685    | 17854936.225   | 3344.635       | 4225.510      | 0.325        | -0.078     |
Table 6 : Hasil training dan testing tanpa param
Dari tabel diatas dapat dijelaskan bahwa:
* Huber_MAE: Rata-rata absolut dari selisih antara nilai prediksi dan nilai aktual pada data train adalah sekitar 1905.190, sedangkan pada data test sekitar 2189.424. Ini menunjukkan bahwa model memiliki tingkat kesalahan yang sedikit lebih tinggi pada data test dibandingkan dengan data train.

* SVR_MAE: Rata-rata absolut dari selisih antara nilai prediksi dan nilai aktual pada data train adalah sekitar 2584.369, sedangkan pada data test sekitar 2907.168. Hal ini menunjukkan bahwa model memiliki tingkat kesalahan yang sedikit lebih tinggi pada data test dibandingkan dengan data train.

* Huber_MSE: Rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual pada data train adalah sekitar 7,774,691.441, sedangkan pada data test sekitar 11,186,580.685. Ini menunjukkan bahwa model memiliki tingkat kesalahan yang sedikit lebih tinggi pada data test dibandingkan dengan data train.

* SVR_MSE: Rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual pada data train adalah sekitar 13,188,739.278, sedangkan pada data test sekitar 17,854,936.225. Hal ini menunjukkan bahwa model memiliki tingkat kesalahan yang sedikit lebih tinggi pada data test dibandingkan dengan data train.

* Huber_RMSE: Akar kuadrat dari Huber_MSE pada data train adalah sekitar 2,788.313, sedangkan pada data test sekitar 3,344.635. Ini menunjukkan bahwa prediksi model memiliki tingkat kesalahan yang sedikit lebih tinggi pada data test dibandingkan dengan data train.

* SVR_RMSE: Akar kuadrat dari SVR_MSE pada data train adalah sekitar 3,631.630, sedangkan pada data test sekitar 4,225.510. Hal ini menunjukkan bahwa prediksi model memiliki tingkat kesalahan yang sedikit lebih tinggi pada data test dibandingkan dengan data train.

* Huber_R2: Koefisien determinasi (R-squared) pada data train adalah sekitar 0.391, sedangkan pada data test sekitar 0.325. Ini menunjukkan bahwa model memiliki kemampuan yang lebih baik dalam menjelaskan variasi data pada data train dibandingkan dengan data test.

* SVR_R2: Koefisien determinasi (R-squared) pada data train adalah sekitar -0.034, sedangkan pada data test sekitar -0.078. Hal ini menunjukkan bahwa model memiliki kinerja yang buruk dalam menjelaskan variasi data baik pada data train maupun data test.

Kesimpulannya, terdapat perbedaan kinerja model antara data train dan data test. Model cenderung memiliki tingkat kesalahan yang sedikit lebih tinggi pada data test, menunjukkan adanya overfitting pada data train. Selain itu, koefisien determinasi (R-squared) juga menunjukkan bahwa model memiliki kemampuan yang lebih baik dalam menjelaskan variasi data pada data train dibandingkan dengan data test. Sehingga, perlu adanya penerapan Hyperparam setelah model masuk ke evaluasi

## Kelebihan dan kekurangan masing-masing algoritma
- Dari hasil pengujian, Algoritma Huber lebih unggul daripada SVR terhadap data yang dimiliki.
- Pada Huber memiliki keunggulan yaitu Lebih toleran terhadap outliers dan cenderung memberikan estimasi parameter yang lebih stabil
- Pada SVR memiliki keunggulan yaitu mampu menangani data yang memiliki hubungan non-linear, kemampuan menangani outliers serta memiliki berbagai macam kernel sesuai dengan yang dibutuhkan.
- Namun, Huber memiliki kelemahan yaitu kurang fleksibel menangani data non-linear dan membutuhkan penyetelan param yang tepat untuk menghasilkan model terbaik
- Sedangkan SVR memiliki kelemahan yaitu apabila model semakin kompleks akan berdampak kepada performa processing yang lebih lama dan sangat sensitif terhadap param yang digunakan

# Evaluasi

### MSE
Untuk metode evaluasi menggunakan metriks MSE atau Mean Squared Error terhadap model machine learning yang di kembangkan.Cara kerja metriks ini sendiri cukup simpel yaitu semakin kecil angka yang keluar maka model yang dihasilkan semakin baik. MSE memberikan bobot yang lebih besar pada perbedaan yang besar dan juga menghasilkan nilai non-negatif karena nilai dikuadratkan [2]. MSE dihitung dengan cara mengambil perbedaan antara nilai prediksi (ŷ) dan nilai sebenarnya (y) untuk setiap data poin, mengkuadratkannya, dan kemudian mengambil rata-rata dari seluruh perbedaan kuadrat tersebut.

Untuk rumusnya yaitu : MSE = $$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* Dimana n adalah jumlah data poin dalam dataset 
* y adalah nilai sebenarnya dari target atau variabel yang diprediksi 
* ŷ adalah nilai prediksi dari target atau variabel yang diprediksi.

### MAE
MAE merupakan metrik evaluasi yang mengukur rata-rata selisih absolut antara nilai prediksi dan nilai sebenarnya [2]. Dimana, semakin kecil angka maka performa semakin baik. MAE juga memberikan gambaran tentang seberapa besar kesalahan prediksi dalam satuan yang sama dengan variabel target.

Rumus : MAE = $$\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
### RMSE 
RMSE merupakan akar kuadrat dari MSE dan digunakan untuk mengukur akurasi rata-rata prediksi [2].memberikan gambaran tentang seberapa besar kesalahan prediksi dalam satuan yang sama dengan variabel target.

Rumus : RMSE = $$\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### R-Squared (R2)
R-squared (R2) merupakan metrik evaluasi yang mengukur seberapa baik model regresi dapat menjelaskan variasi data [2]. Metriks ini memiliki rentang nilai antara 0 hingga 1, dimana nilai 1 menunjukkan bahwa model dapat menjelaskan seluruh variasi data dengan sempurna.

RUMUS = R2 = $$1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

|     Index    | Huber_MAE | SVR_MAE |  Huber_MSE  |   SVR_MSE    | Huber_RMSE |  SVR_RMSE   |  Huber_R2   |  SVR_R2   |
|:------------:|:---------:|:-------:|:-----------:|:------------:|:----------:|:-----------:|:-----------:|:---------:|
|    train     | 1905.190  | 2584.369| 7774691.441 | 13188739.278 | 2788.313   | 3631.630    | 0.390       | -0.034    |
|     test     | 2189.424  | 2907.168| 11186580.685| 17854936.225 | 3344.635   | 4225.510    | 0.325       | -0.078    |
| eval_train   | 1916.916  | 1894.411| 7575414.075 | 8072194.776  | 2752.347   | 2841.161    | 0.406       | 0.367     |
| eval_test    | 2232.534  | 2181.098| 11010488.333| 11648441.539 | 3318.206   | 3412.981    | 0.335       | 0.297     |

Tabel 7 : Hasil Evaluasi Model 
Berdasarkan data yang diberikan,dapat diambil kesimpulan bahwa model terbaik adalah model Huber Regression.Hal ini berdasarkan hasil dari setelah model masuk ketahap evaluasi dengan menerapkan hyperparam. Dimana, model menunjukkan nilai skor yang lebih baik secara keseluruhan matriks yang ada. Model huber memiliki tingkat kesalahan yang lebih rendah pada data pelatihan dan evaluasi, serta memiliki kemampuan yang lebih baik dalam menjelaskan variasi dalam data.

## Kesimpulan
Dari proyek ini dapat disimpulkan Bahwa:
- Spesifikasi ponsel memiliki keterkaitan terhadap harga ponsel yang dijual, dimana semakin bagus spesifikasi maka semakin mahal pula harganya.
- Sangat memungkinkan untuk mengimplementasikan model machine learning yang mampu memprediksi harga ponsel berdasarkan spesifikasi yang dimiliki.
- Fitur yang cukup mempengaruhi harga yaitu Memory Internal, RAM, Resolusi layar dan ukuran layar
- Sedangkan Baterai tidak terlalu mempengaruhi harga ponsel
- Huber Memiliki skor MAE 2232.534, MSE 11010488.333 , RMSE 3318.206 dan R2 0.335
- Sedangkan SVR Memiliki skor MAE 2181.098, MSE 11648441.539 , RMSE 3412.981 dan R2 0.297
- Dari hasil pengujian Training dan Testing, maka algoritma yang cocok untuk studi kasus ini adalah Huber Regessor yang memiliki nilai evaluasi yang paling bagus.
- Huber memiliki kemampuan memprediksi kesalahan lebih rendah dan mampu menjelaskan lebih baik variasi dalam data berdasarkan metriks evaluasi.

Dalam proyek ini juga memiliki beberapa kekurangan yaitu :
* Data testing yang kurang banyak dimana hanya memilki 1359 baris data yang setelah masuk processing menjadi 938. Hal ini dapat membuat model belajar dari sedikit data. Dimana semakin banyak data maka model akan semakin baik dalam mempelajari data yang ada.
* Perlu adanya pembanding algoritma yang lain seperti Penggunaan algoritma Random Forest, Lasso regression , AdaBoost Regressor dan masih banyak algoritma regresi lainnya.

### Referensi
[1]. T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning: Data Mining, Inference, and Prediction," Springer, 2009. [Online]. Available: [https://www.statlearning.com/](https://www.statlearning.com/)

[2]. G. James, D. Witten, T. Hastie, and R. Tibshirani, "An Introduction to Statistical Learning: with Applications in R," Springer, 2013. [Online]. Available: [https://www.statlearning.com/](https://www.statlearning.com/)

[3]. R. Patel and A. Sharma, "Predictive Analysis of Smartphone Prices Using Machine Learning Techniques," in *Proceedings of the 3rd International Conference on Computing Methodologies and Communication*, 2021, pp. 471-479.

[4]. C. T. Chou, Y. H. Ho, W. J. Chen, and C. W. Tsai, "A Machine Learning-Based Framework for Smartphone Price Prediction," *Information Systems Frontiers*, vol. 22, no. 6, pp. 1749-1763, 2020. [Online]. Available: [https://doi.org/10.1007/s10796-020-10068-0](https://doi.org/10.1007/s10796-020-10068-0)

[5]. K. M. Reddy, K. V. Kumar, and K. R. Reddy, "Prediction of Smartphone Prices Using Machine Learning Algorithms," *International Journal of Advanced Science and Technology*, vol. 28, no. 19, pp. 2060-2072, 2019.

[6]. S. Subhiksha, S. Thota, and J. Sangeetha, "Prediction of Phone Prices Using Machine Learning Techniques," in *Data Engineering and Communication Technology*, K. Raju, R. Senkerik, S. Lanka, and V. Rajagopal (eds), Advances in Intelligent Systems and Computing, vol. 1079, Springer, Singapore, 2020. [Online]. Available: [https://doi.org/10.1007/978-981-15-1097-7_65](https://doi.org/10.1007/978-981-15-1097-7_65)
[7]. E. Güvenç, G. Çetin, and H. Koçak, "Comparison of KNN and DNN Classifiers Performance in Predicting Mobile Phone Price Ranges," *Advances in Artificial Intelligence Research*, vol. 1, no. 1, pp. 19-28, Jan. 2021.
[8]. Garai, P. (2021). Mobile Phone Specifications and Prices. Kaggle. [Online]. Available: https://www.kaggle.com/datasets/pratikgarai/mobile-phone-specifications-and-prices. [Accessed: 22-05-2023].


