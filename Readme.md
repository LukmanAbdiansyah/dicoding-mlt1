
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
Dataset yang digunakan berdasal dari kaggle dengan nama [Well log Dataset](https://www.kaggle.com/datasets/imeintanis/well-log-facies-dataset) yang memiliki jumlah data 3233 baris dan 11 kolom. Dataset merupakan hasil dari penggabungan dari website [KGS](ttps://www.kgs.ku.edu/Hugoton/TypeLogs/index.html). Dataset berisikan tentang 5 pengukuran log wireline, dua indikator dan label fasies dalam interval setengah kaki.

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

![urutan fasies](https://github.com/LukmanAbdiansyah/dicoding-mlt1/assets/74924298/09b73870-6309-41a0-ad66-bacad6fe5e92)
Gambar 1 : urutan sumur dengan jumlah sampel terbanyak pada formasi hugoton dan panoma
Pada gambar 1 yang memiliki data sumur paling banyak yaitu sumur Cross H Castle.
![urutan fasies](https://github.com/LukmanAbdiansyah/dicoding-mlt1/assets/74924298/d92edbd1-a9c4-48bc-906a-c82293121769)
Gambar 2 : urutan fasies terbanyak pada formasi hugoton dan panoma
Pada gambar 2 yang memiliki data fasies paling banyak yaitu Nonmarine coarse siltstone.

![hist](https://github.com/LukmanAbdiansyah/dicoding-mlt1/assets/74924298/333033fd-8866-4f10-b434-11897bb04ad8)
Gambar 3: Persebaran Dataset

Pada gambar diatas ada beberapa kesimpulan yaitu :
*   Pada kolom RELPOS (Relative Position), jumlah nilai relpos cenderung sama untuk setiap titik
*   Pada kolom PHIND (porositas neutron-density), persebaran porositas cenderung pada 5-22% dan memiliki skewness positif
*   Pada kolom GR (Gamma-ray) memiliki skewness positif, ini menyimpulkan bahwa formasi hugoton panoma cukup clean (kandungan shale dan silt tidak sebesar resrvoir pada umumnya).

![hist](https://user-images.githubusercontent.com/74924298/242853417-c881f986-2778-4219-9f7c-3a98fa0767f1.jpg)
Gambar 4: Well Log Visualization untuk sumur SHANKLE (sebagai representasi)
Dari gambar 3 terlihat bahwa zona reservoir berada pada kedalaman 2975-2990 ft, hal ini karena pada zona tersebut GR kecil, resistivitas besar (ILD), dan porositas masih cukup baik (PHIND).

![pairplotall](https://github.com/LukmanAbdiansyah/dicoding-mlt1/assets/74924298/aa961e0b-4216-43f0-9595-f48070233fb1)
Gambar 5 : Korelasi antar fasies dengan fitur yang lain secara pairplot
pada pairplot diatas terlihat bahwa fasies memiliki distribusi yang berkolerasi positif dengan feature NM_M dan PE

![heatmap](https://github.com/LukmanAbdiansyah/dicoding-mlt1/assets/74924298/b9375e19-e3d9-4d35-bdb8-4f790afa9df9)
Gambar 6: Correlation Matrix
Dari heatmap diatas fasies memiliki korelasi positif yang cukup tinggi untuk , sedangkan fitur yang lain memiliki korelasi yang tidak terlalu besar .
Dan pada fitur relpos, memiliki korelasi yang paling kecil diantara fitur yang lainnya.

### Result EDA
Sejauh tahap yang dilakukan, karena data sudah clean dan tidak perlu melakukan handling outlier karena outlier tetap digunakan.

# Data Preparation
## Proses yang dilakukan
-   Memberi label untuk setiap fasies dengan angka
-   Memisahkan data sumur NEWBY untuk dilakukan blind data testing untuk satu log sumur
-   Melakukan standardisasi untuk data feature
-   Membagi data set antara training dan testing dengan library sklearn dengan fungsi train_test_split() dengan perbandingan 80:20 sehingga memiliki 2215 data train dan 554 data testing

## Alasan Pengunaan
- pemberian label angka memungkinkan representasi numerik untuk data kategori atau kualitatif. Hal ini memungkinkan algoritma pembelajaran mesin atau model statistik untuk memproses data tersebut, karena umumnya algoritma-algoritma tersebut membutuhkan input berupa angka.

- Pemisahan data sumur NEWBY digunakan untuk dilakukan blind data testing dalam satu log sumur
- Data splitting dilakukan untuk evaluasi obyektif kinerja model, mencegah overfitting, dan memisahkan data pengujian yang tidak digunakan dalam proses pelatihan untuk menghindari informasi bocor. Pembagian data memungkinkan tuning parameter yang efektif dan pemahaman yang lebih baik tentang variasi kinerja model.
- Standar Scaler digunakan untuk menormalkan atau menskalakan data numerik dalam skala yang sama. Menggunakan data dengan skala yang berbeda dapat mengganggu performa model yang menggunakan metrik jarak seperti algoritma SVM. Sehingga, model dapat menormalkan data numerik sehingga memiliki rata-rata nol dan standar deviasi satu. Ini membantu dalam menghilangkan perbedaan skala dan memastikan bahwa setiap fitur diperlakukan secara adil dan tidak mendominasi pengaruhnya terhadap hasil model.

# Modeling
Digunakan empat algoritma teratas yang memiliki performa yang terbaik berdasarkan hasil dari running data menggunakan library LazyClassifier.

|           **Model**           | **Accuracy** | **Balanced Accuracy** | **F1 Score** | **Time Taken** |
|:-----------------------------:|:------------:|:---------------------:|-------------:|---------------:|
|      ExtraTreesClassifier     |     0.74     |          0.74         |     0.74     |      0.68      |
|         LabelSpreading        |     0.72     |          0.74         |     0.72     |      0.83      |
|        LabelPropagation       |     0.71     |          0.73         |     0.71     |      0.40      |
|     RandomForestClassifier    |     0.72     |          0.72         |     0.72     |      1.05      |
|       BaggingClassifier       |     0.69     |          0.69         |     0.69     |      0.15      |
|         LGBMClassifier        |     0.71     |          0.68         |     0.71     |      5.43      |
|      KNeighborsClassifier     |     0.64     |          0.64         |     0.64     |      0.06      |
|     DecisionTreeClassifier    |     0.62     |          0.62         |     0.62     |      0.04      |
|      ExtraTreeClassifier      |     0.61     |          0.61         |     0.61     |      0.03      |
|              SVC              |     0.61     |          0.56         |     0.60     |      0.88      |
|        NearestCentroid        |     0.50     |          0.53         |     0.50     |      0.16      |
|   LinearDiscriminantAnalysis  |     0.55     |          0.52         |     0.53     |      0.08      |
|       LogisticRegression      |     0.55     |          0.50         |     0.54     |      0.22      |
|           LinearSVC           |     0.55     |          0.49         |     0.53     |      0.68      |
|     CalibratedClassifierCV    |     0.54     |          0.48         |     0.53     |      1.82      |
|         SGDClassifier         |     0.50     |          0.45         |     0.47     |      0.33      |
|          BernoulliNB          |     0.46     |          0.41         |     0.44     |      0.03      |
|           Perceptron          |     0.37     |          0.39         |     0.35     |      0.06      |
|  PassiveAggressiveClassifier  |     0.35     |          0.37         |     0.31     |      0.06      |
|           GaussianNB          |     0.25     |          0.35         |     0.20     |      0.02      |
|        RidgeClassifier        |     0.48     |          0.34         |     0.42     |      0.07      |
|       RidgeClassifierCV       |     0.48     |          0.33         |     0.42     |      0.11      |
|       AdaBoostClassifier      |     0.35     |          0.31         |     0.31     |      0.28      |
| QuadraticDiscriminantAnalysis |     0.21     |          0.29         |     0.14     |      0.06      |
|        DummyClassifier        |     0.23     |          0.11         |     0.09     |      0.02      |

Tabel 5 : Model Reference

Pada saat menjalankan library lazyclassifier, ExtraTreesClassifier memiliki skor Accuracy dan F-1 Score paling tinggi daripada algoritma lain.

## Tahapan yang dilakukan
- Melatih Model dengan data training dengan menggunakan algoritma Random Forest, ExtraTreesClassifier, Label Spreading, dan Label Propagation

- Pada tahap awal training, akan dilakukan training model dengan parameter default yang ada pada library
- Melakukan testing dengan data training
- Kemudian, lanjut pengujian dengan data testing
- metrik evaluasi menggunakan accuracy dan f1-score dengan menggunakan library sklearn.
- Melihat hasil performa model antara hasil training dan testing
- Kemudian tingkatkan performa model dengan menerapkan grid search atau hyperparameter tuning pada model.
- Untuk hyperparameter tuning yang digunakan pada Random Forest adalah 
param_grid = { 'n_estimators': [100,200,300,400,500],
'max_features': ['auto', 'sqrt', 'log2'],
'max_depth' : [3,4,5,6,7,8],
'criterion' :['gini', 'entropy']}
- Pada ExtraTreesClassifier 
param_grid1 =
{'n_estimators': [10,50,100,300,500,1000],
'max_features': ['auto', 'sqrt', 'log2',None],
'max_depth' : [2,3,4,5,6,7,8],
'criterion' :['gini', 'entropy', 'log_loss']}
- Dari hyperparameter tuning , mendapatkan parameter yang terbaik yaitu {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 400} pada model Random Forest
- Sedangkan untuk ExtraTreesClassifier {'criterion': 'entropy', 'max_depth': 8, 'max_features': None, 'n_estimators': 500}

### Hasil Penerapan Model untuk Blind Testing Sumur NEWBY
|   |       **Model type** | **Acuracy** | **F-1 Score** |
|--:|---------------------:|------------:|--------------:|
| 0 |        Random Forest |        0.48 |          0.49 |
| 1 | ExtraTreesClassifier |        0.51 |          0.51 |
| 2 |       LabelSpreading |        0.46 |          0.48 |
| 3 |     LabelPropagation |        0.46 |          0.47 |
Table 6 : Hasil Penerapan Model dengan model default
|   |           Model type | Acuracy | F-1 Score |
|--:|---------------------:|--------:|----------:|
| 0 |        Random Forest |    0.53 |      0.53 |
| 1 | ExtraTreesClassifier |    0.52 |      0.50 |
| 2 |       LabelSpreading |    0.46 |      0.48 |
| 3 |     LabelPropagation |    0.46 |      0.47 |
Table 7 : Hasil Penerapan Model dengan setelah hyperparameter tuning

Dari kedua tabel diatas dapat dijelaskan bahwa:
-   Perbedaan terlihat pada kolom "Acuracy" dan "F-1 Score" untuk setiap model.
-   Pada Tabel 6 (model default), akurasi model Random Forest adalah 0.48, sedangkan setelah tuning pada Tabel 7, akurasi model tersebut meningkat menjadi 0.53. Ini menunjukkan peningkatan yang signifikan dalam kinerja model setelah tuning.
-   Pada model ExtraTreesClassifier, akurasi pada Tabel 6 adalah 0.51, sedangkan pada Tabel 7, akurasi tersebut sedikit menurun menjadi 0.52 setelah tuning. Namun, F-1 Score pada model ini mengalami penurunan dari 0.51 menjadi 0.50 setelah tuning.
-   Dengan demikian, hasil tuning pada hyperparameter hanya memiliki dampak yang signifikan pada model Random Forest, dengan peningkatan kinerja yang cukup besar. Sementara itu, model ExtraTreesClassifier mengalami penurunan sedikit dalam hal akurasi dan F-1 Score setelah tuning, dan model LabelSpreading serta LabelPropagation tidak terpengaruh oleh tuning hyperparameter.
- Walaupun accuracy saat training model menurun, tetapi jarak overfitting antara hasil testing dengan trainin tidak sejauh sebelum dilakukan hyperparameter tuning

Kesimpulannya, Secara keseluruhan, hasil tuning hyperparameter dapat memberikan perbaikan pada kinerja model, tetapi dampaknya dapat bervariasi tergantung pada jenis model yang digunakan.

## Kelebihan dan kekurangan masing-masing algoritma
|       Algoritma      |                           Kelebihan                          |                                      Kekurangan                                      |
|:--------------------:|:------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
|     Random Forest    | Mampu menangani banyak fitur dengan baik                     | Kompleksitas komputasi yang tinggi                                                   |
|                      | Mampu menangani atribut tidak relevan atau nilai yang hilang | Rentan terhadap overfitting                                                          |
|                      | Stabil dan variasi yang rendah dalam estimasi                | Sulit menginterpretasi hubungan antara variabel                                      |
|                      | Cocok untuk klasifikasi dan regresi                          | Kurang efektif dalam data dengan noise atau outlier tinggi                           |
|                      | Mampu mengukur pentingnya fitur dalam prediksi               |                                                                                      |
| ExtraTreesClassifier | Pembentukan model yang lebih cepat                           | Tidak memberikan interpretasi hubungan antara variabel                               |
|                      | Menangani atribut tidak relevan atau nilai yang hilang       | Kompleksitas komputasi yang tinggi                                                   |
|                      | Kurang rentan terhadap overfitting                           | Sulit mengevaluasi keberagaman prediksi dalam ensemble                               |
|                      | Cocok untuk klasifikasi dan regresi                          |                                                                                      |
|                      | Mampu mengukur pentingnya fitur dalam prediksi               |                                                                                      |
|    LabelSpreading    | Menangani label terdistribusi tidak merata atau noise        | Kompleksitas komputasi yang tinggi                                                   |
|                      | Menangani atribut tidak relevan atau nilai yang hilang       | Bergantung pada parameter yang tepat untuk menghindari overfitting atau underfitting |
|                      | Propagasi label pada data yang belum diberi label            | Sulit menginterpretasi hubungan antara variabel                                      |
|                      | Hasil yang baik pada data dengan struktur yang kompleks      |                                                                                      |
|   LabelPropagation   | Menangani label terdistribusi tidak merata atau noise        | Kompleksitas komputasi yang tinggi                                                   |
|                      | Menangani atribut tidak relevan atau nilai yang hilang       | Bergantung pada parameter yang tepat untuk menghindari overfitting atau underfitting |
|                      | Propagasi label pada data yang belum diberi label            | Sulit menginterpretasi hubungan antara variabel                                      |
|                      | Menangani data dengan struktur kompleks dan interaksi rumit  |                                                                                      |

# Evaluasi

### Accuracy
Akurasi adalah metrik umum yang digunakan untuk mengevaluasi kinerja model klasifikasi. Metrik ini mengukur proporsi prediksi yang benar dari total jumlah instance dalam dataset. Akurasi memberikan penilaian keseluruhan tentang sejauh mana model mampu memprediksi label kelas yang benar.

Untuk rumusnya yaitu : Akurasi = (Jumlah prediksi yang benar) / (Total jumlah instance)

### F-1 Score
Nilai F1-Score berkisar antara 0 hingga 1, di mana nilai 1 menunjukkan kinerja model yang sempurna dalam memprediksi kelas yang benar. F1-Score memberikan informasi tentang keseimbangan antara presisi dan kepekaan, yang penting dalam kasus di mana kelas yang tidak seimbang atau biaya kesalahan yang tidak merata.

-   Presisi (precision) mengukur sejauh mana model memberikan prediksi yang benar dari semua prediksi positif yang dilakukan. Presisi diperoleh dengan membagi jumlah prediksi positif yang benar dengan jumlah total prediksi positif (benar dan salah).
-   Kepekaan (recall) mengukur sejauh mana model mampu mengidentifikasi semua instance positif yang sebenarnya. Kepekaan diperoleh dengan membagi jumlah prediksi positif yang benar dengan jumlah total instance positif yang sebenarnya dalam dataset.

F1-Score memberikan nilai yang lebih baik daripada menggunakan presisi atau kepekaan saja, karena menggabungkan keduanya dalam satu ukuran yang seimbang. F1-Score lebih tepat digunakan dalam kasus ketimpangan kelas, di mana penilaian yang seimbang antara prediksi benar dan pengabaian instance negatif sangat penting.
Untuk rumusnya yaitu : F1-Score = 2 * (Presisi * recall) / (Presisi + Recall)


|   |           Model type | Acuracy | F-1 Score |
|--:|---------------------:|--------:|----------:|
| 0 |        Random Forest |    0.53 |      0.53 |
| 1 | ExtraTreesClassifier |    0.52 |      0.50 |
| 2 |       LabelSpreading |    0.46 |      0.48 |
| 3 |     LabelPropagation |    0.46 |      0.47 |

Tabel 8 : Hasil Evaluasi Model 
Berdasarkan tabel 8 yang diberikan,dapat diambil kesimpulan bahwa model terbaik adalah model random forest setelah dilakukan hyperparameter tuning. karena model memberikan skor pengujian paling baik diantara keempat model. Model Random Forest memiliki tingkat kesalahan yang lebih rendah antara data pelatihan dan evaluasi.

### Confusion Matrix
![messageImage_1685728685966](https://github.com/LukmanAbdiansyah/dicoding-mlt1/assets/74924298/e85c930c-13da-4dd2-b3b6-ecc65c7727ff)
## Kesimpulan
Dari proyek ini dapat disimpulkan Bahwa:
- Dengan menggunakan model machine learning, proses klasifikasi facies dapat diotomatisasi, yang akan menghasilkan peningkatan efisiensi dan penghematan waktu yang signifikan. sehingga sangat memungkinkan untuk mengimplementasikannya.
- Fitur NN_M dan PE memiliki pengaruh paling besar terhadap fasies
- Model machine learning terbaik pada proyek ini adalah Random Forest dengan hyperparameter tuning dimana accuracy training yang didapatkan sebesar 65% dan accuracy blind testing sebesar 53%
- Random Forest Memiliki F1-score sebesar 53%
- Sedangkan ExtraTreesClassifier Memiliki skor 50%
- Fasies Nonmarine Sandstone merupakan fasies dengan hasil prediksi paling banyak benar
- Dalam proyek ini, model machine learning dapat mempelajari pola-pola kompleks dan relasi antara atribut geologi yang berkontribusi pada klasifikasi facies. Pengetahuan ini dapat digunakan untuk meningkatkan pemodelan dan prediksi lingkungan deposisi serta memahami sifat reservoir hidrokarbon dengan lebih baik.

terdapat beberapa kekurangan dari proyek ini yaitu :
* Data yang digunakan cukup sedikit, hanya sebanyak 3232 row, dengan data training sebesar 2215 row, data testing sebesar 554 row, dan blind data testing 463 row. sehingga hasil prediksi model masih banyak kesalahan, karena itu untuk meningkatkan akurasi perlu adanya penambahan data.


### Referensi
[1]. Hall, B. (2016) ‘Facies classification using machine learning’, _The Leading Edge_, 35(10), pp. 906–909. doi:10.1190/tle35100906.1.

[2]. Saroji, S. _et al._ (2021) ‘The implementation of machine learning in lithofacies classification using Multi Well Logs Data’, _Aceh International Journal of Science and Technology_, 10(1), pp. 9–17. doi:10.13170/aijst.10.1.18749.

[3]. Alaudah, Y. _et al._ (2019) _A machine learning benchmark for facies classification_, _arXiv.org_. Available at: https://arxiv.org/abs/1901.07659 (Accessed: 03 June 2023).

[4]. Erzikova, J. (2019) ‘Facies classification FROMWELL logs using machine learning methods: A survey’, _SGEM International Multidisciplinary Scientific GeoConference EXPO Proceedings_ [Preprint]. doi:10.5593/sgem2019/2.1/s07.037.

[5]. Jaikla, C. (2019). FaciesNet: Machine Learning Applications for Facies Classification in Well Logs.

