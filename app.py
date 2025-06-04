import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
from io import StringIO # Diperlukan untuk menangkap output train_df.info()

warnings.filterwarnings('ignore')

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis & Prediksi Penjualan Kendaraan Listrik",
    page_icon="🚗",
    layout="wide" # Menggunakan layout lebar untuk tampilan yang lebih baik
)

# Judul utama dashboard
st.title('🚗 Analisis dan Prediksi Penjualan Kendaraan Listrik')
st.markdown("""
    Dashboard interaktif ini menyajikan analisis mendalam tentang data penjualan kendaraan listrik,
    meliputi eksplorasi data, segmentasi pelanggan dengan K-Means clustering,
    dan prediksi penjualan menggunakan model Logistic Regression.
""")
st.markdown("---") # Garis pemisah

# Sidebar untuk navigasi
analysis_option = st.sidebar.selectbox(
    'Pilih Tipe Analisis',
    ['Eksplorasi Data', 'K-Means Clustering', 'Model Prediksi Penjualan', 'Prediksi Individu']
)

# Fungsi untuk memuat data dengan caching untuk performa
@st.cache_data
def load_and_preprocess_data():
    """Memuat dan melakukan pra-pemrosesan data (menangani nilai hilang, duplikat, outlier)."""
    try:
        # Mencoba membaca file CSV
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')

        st.success("File data 'train.csv' dan 'test.csv' berhasil dimuat.")

        # --- Pra-pemrosesan Data ---
        # Menangani nilai yang hilang
        numeric_cols = ['Battery_Capacity_kWh', 'Discount_Percentage']
        categorical_cols = ['Region', 'Brand', 'Model', 'Vehicle_Type', 'Customer_Segment', 'Fast_Charging_Option']

        for col in numeric_cols:
            if train_df[col].isnull().sum() > 0:
                train_df[col].fillna(train_df[col].mean(), inplace=True)
            if test_df[col].isnull().sum() > 0:
                test_df[col].fillna(test_df[col].mean(), inplace=True)

        for col in categorical_cols:
            if train_df[col].isnull().sum() > 0:
                 mode_val = train_df[col].mode()
                 if not mode_val.empty:
                     train_df[col].fillna(mode_val[0], inplace=True)
                 else:
                     train_df[col].fillna('Unknown', inplace=True) # Fallback
            if test_df[col].isnull().sum() > 0:
                 mode_val = test_df[col].mode()
                 if not mode_val.empty:
                     test_df[col].fillna(mode_val[0], inplace=True)
                 else:
                     test_df[col].fillna('Unknown', inplace=True) # Fallback


        # Menangani duplikat
        if train_df.duplicated().sum() > 0:
            train_df.drop_duplicates(inplace=True)
        if test_df.duplicated().sum() > 0:
            test_df.drop_duplicates(inplace=True)

        # Mengatasi outlier dengan capping (Winsorization)
        def detect_outliers_iqr(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return lower_bound, upper_bound

        outlier_cols = ['Battery_Capacity_kWh', 'Discount_Percentage', 'Units_Sold', 'Revenue'] # Termasuk Revenue
        for col in outlier_cols:
            if col in train_df.columns:
                lower_bound, upper_bound = detect_outliers_iqr(train_df, col)
                train_df[col] = np.where(train_df[col] < lower_bound, lower_bound, train_df[col])
                train_df[col] = np.where(train_df[col] > upper_bound, upper_bound, train_df[col])


        st.sidebar.success("Pra-pemrosesan data selesai.")
        return train_df, test_df
    except FileNotFoundError:
        # Jika file tidak ada, buat data dummy untuk demo
        st.warning("⚠️ File data ('train.csv' atau 'test.csv') tidak ditemukan. Menggunakan data dummy untuk demonstrasi.")

        np.random.seed(42)
        n_samples = 1000

        train_df = pd.DataFrame({
            'Battery_Capacity_kWh': np.random.normal(60, 15, n_samples).clip(10, 150),
            'Discount_Percentage': np.random.normal(15, 5, n_samples).clip(0, 30),
            'Units_Sold': np.random.poisson(25, n_samples).clip(1, 100),
            'Revenue': np.random.normal(500000, 100000, n_samples).clip(10000, 1000000),
            'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
            'Brand': np.random.choice(['Tesla', 'BMW', 'Audi', 'Mercedes', 'Nissan', 'Ford', 'Hyundai'], n_samples),
            'Model': np.random.choice([f'Model_{chr(65+i)}' for i in range(15)], n_samples),
            'Vehicle_Type': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Truck', 'Van'], n_samples),
            'Customer_Segment': np.random.choice(['Premium', 'Mid-range', 'Economy', 'Commercial', 'Fleet'], n_samples),
            'Fast_Charging_Option': np.random.choice(['Yes', 'No'], n_samples)
        })

        test_df = train_df.sample(200, replace=True, random_state=42).copy()

        # Lakukan pra-pemrosesan dasar juga pada data dummy
        for col in ['Battery_Capacity_kWh', 'Discount_Percentage', 'Units_Sold', 'Revenue']:
             lower, upper = train_df[col].quantile(0.01), train_df[col].quantile(0.99) # Capping ringan
             train_df[col] = train_df[col].clip(lower, upper)
             test_df[col] = test_df[col].clip(lower, upper)


        st.sidebar.info("Data dummy berhasil dibuat dan dipra-proses.")
        return train_df, test_df

# Memuat dan pra-memproses data
train_df, test_df = load_and_preprocess_data()

# Bagian Eksplorasi Data
if analysis_option == 'Eksplorasi Data':
    st.header('📊 Eksplorasi Data')

    st.markdown("Bagian ini menampilkan ringkasan statistik dan visualisasi awal dari dataset penjualan kendaraan listrik.")

    # Ikhtisar Dataset
    st.subheader('📋 Ikhtisar Dataset')
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Ukuran Dataset Train:** `{train_df.shape}`")
        st.write(f"**Ukuran Dataset Test:** `{test_df.shape}`")

        st.write("**5 Baris Pertama Dataset Train:**")
        st.dataframe(train_df.head())

    with col2:
        st.write("**Statistik Deskriptif Dataset Train:**")
        st.dataframe(train_df.describe())

    # Informasi Kolom
    st.subheader('📊 Informasi Kolom dan Tipe Data')
    buffer = StringIO()
    train_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


    # Visualisasi Data
    st.subheader('📈 Visualisasi Data Utama')

    viz_type = st.selectbox(
        'Pilih Jenis Visualisasi',
        ['Distribusi Kapasitas Baterai (kWh)', 'Distribusi Persentase Diskon', 'Distribusi Unit Terjual',
         'Total Penjualan berdasarkan Wilayah', 'Total Penjualan berdasarkan Tipe Kendaraan',
         'Hubungan Kapasitas Baterai vs Unit Terjual', 'Hubungan Diskon vs Unit Terjual', 'Hubungan Revenue vs Unit Terjual']
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    if viz_type == 'Distribusi Kapasitas Baterai (kWh)':
        sns.histplot(train_df['Battery_Capacity_kWh'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribusi Kapasitas Baterai')
        ax.set_xlabel('Kapasitas Baterai (kWh)')
        ax.set_ylabel('Frekuensi')

    elif viz_type == 'Distribusi Persentase Diskon':
        sns.histplot(train_df['Discount_Percentage'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribusi Persentase Diskon')
        ax.set_xlabel('Persentase Diskon (%)')
        ax.set_ylabel('Frekuensi')

    elif viz_type == 'Distribusi Unit Terjual':
        sns.histplot(train_df['Units_Sold'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribusi Unit Terjual')
        ax.set_xlabel('Unit Terjual')
        ax.set_ylabel('Frekuensi')

    elif viz_type == 'Total Penjualan berdasarkan Wilayah':
        region_sales = train_df.groupby('Region')['Units_Sold'].sum().sort_values(ascending=False)
        region_sales.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Total Penjualan berdasarkan Wilayah')
        ax.set_xlabel('Wilayah')
        ax.set_ylabel('Total Unit Terjual')
        plt.xticks(rotation=45, ha='right') # Rotasi dan aligment
        plt.tight_layout() # Menyesuaikan layout

    elif viz_type == 'Total Penjualan berdasarkan Tipe Kendaraan':
        sns.boxplot(x='Vehicle_Type', y='Units_Sold', data=train_df, ax=ax)
        ax.set_title('Unit Terjual berdasarkan Tipe Kendaraan')
        ax.set_xlabel('Tipe Kendaraan')
        ax.set_ylabel('Unit Terjual')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    elif viz_type == 'Hubungan Kapasitas Baterai vs Unit Terjual':
        sns.scatterplot(x='Battery_Capacity_kWh', y='Units_Sold', data=train_df, ax=ax, alpha=0.6)
        ax.set_title('Hubungan antara Kapasitas Baterai dan Unit Terjual')
        ax.set_xlabel('Kapasitas Baterai (kWh)')
        ax.set_ylabel('Unit Terjual')

    elif viz_type == 'Hubungan Diskon vs Unit Terjual':
        sns.scatterplot(x='Discount_Percentage', y='Units_Sold', data=train_df, ax=ax, alpha=0.6)
        ax.set_title('Hubungan antara Persentase Diskon dan Unit Terjual')
        ax.set_xlabel('Persentase Diskon (%)')
        ax.set_ylabel('Unit Terjual')

    elif viz_type == 'Hubungan Revenue vs Unit Terjual':
        sns.scatterplot(x='Revenue', y='Units_Sold', data=train_df, ax=ax, alpha=0.6)
        ax.set_title('Hubungan antara Revenue dan Unit Terjual')
        ax.set_xlabel('Revenue')
        ax.set_ylabel('Unit Terjual')


    st.pyplot(fig)

# Bagian K-Means Clustering
elif analysis_option == 'K-Means Clustering':
    st.header('🎯 Analisis K-Means Clustering')
    st.markdown("Bagian ini melakukan segmentasi data penjualan berdasarkan fitur numerik menggunakan algoritma K-Means.")

    st.subheader('🔧 Konfigurasi Clustering')

    # Pemilihan fitur untuk clustering
    clustering_features = st.multiselect(
        'Pilih fitur numerik untuk clustering',
        ['Battery_Capacity_kWh', 'Discount_Percentage', 'Units_Sold', 'Revenue'],
        default=['Battery_Capacity_kWh', 'Discount_Percentage', 'Units_Sold']
    )

    if len(clustering_features) < 2:
        st.warning('⚠️ Harap pilih minimal 2 fitur untuk melakukan clustering.')
    else:
        # Menentukan jumlah cluster
        n_clusters = st.slider('Pilih Jumlah Cluster (k)', min_value=2, max_value=10, value=4)

        # Persiapan data dan standardisasi
        X_cluster = train_df[clustering_features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # Menerapkan K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Menggunakan n_init=10
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Menambahkan label cluster ke dataframe asli
        train_df_clustered = train_df.copy()
        train_df_clustered['Cluster'] = cluster_labels

        st.subheader('📊 Hasil Clustering')

        # Visualisasi Cluster (menggunakan 2 fitur pertama jika lebih dari 2 fitur dipilih)
        if len(clustering_features) >= 2:
            st.write(f"Visualisasi 2D menggunakan fitur **'{clustering_features[0]}'** dan **'{clustering_features[1]}'** (setelah scaling).")
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
                                 c=cluster_labels, cmap='viridis', alpha=0.7, s=50) # s= ukuran titik
            centers = kmeans.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.8, marker='X', label='Centroids') # s= ukuran centroid
            ax.set_title('Visualisasi Cluster K-Means')
            ax.set_xlabel(f'{clustering_features[0]} (scaled)')
            ax.set_ylabel(f'{clustering_features[1]} (scaled)')
            plt.colorbar(scatter, label='Cluster')
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Pilih minimal 2 fitur untuk visualisasi 2D clustering.")


        # Karakteristik Setiap Cluster
        st.subheader('📈 Karakteristik Rata-rata Setiap Cluster')
        cluster_stats_numeric = train_df_clustered.groupby('Cluster')[clustering_features].mean().round(2)
        st.dataframe(cluster_stats_numeric)

        st.subheader('📊 Distribusi Fitur Kategori di Setiap Cluster')

        category_col = st.selectbox(
            'Pilih kolom kategori untuk melihat distribusinya per cluster',
            ['Region', 'Brand', 'Vehicle_Type', 'Customer_Segment', 'Fast_Charging_Option']
        )

        if category_col in train_df_clustered.columns:
            # Hitung persentase untuk visualisasi yang lebih informatif
            cluster_category_counts = pd.crosstab(train_df_clustered['Cluster'], train_df_clustered[category_col])
            cluster_category_percent = cluster_category_counts.divide(cluster_category_counts.sum(axis=1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(12, 8))
            cluster_category_percent.plot(kind='bar', stacked=True, ax=ax, cmap='viridis')
            ax.set_title(f'Distribusi Persentase {category_col} di Setiap Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Persentase')
            plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        else:
             st.warning(f"Kolom '{category_col}' tidak ditemukan di dataset.")


# Bagian Model Prediksi Penjualan (Logistic Regression)
elif analysis_option == 'Model Prediksi Penjualan':
    st.header('📈 Model Prediksi Penjualan (Logistic Regression)')
    st.markdown("""
        Bagian ini melatih model Logistic Regression untuk memprediksi apakah penjualan kendaraan akan tinggi
        (didefinisikan sebagai penjualan di atas median `Units_Sold` di dataset train) atau rendah.
        Hasil pelatihan dan evaluasi model ditampilkan di bagian 'Model Performance'.
    """)

    st.subheader('🔧 Konfigurasi dan Pelatihan Model')

    # Membuat variabel target biner: High_Sales
    median_units = train_df['Units_Sold'].median()
    train_df_pred = train_df.copy()
    train_df_pred['High_Sales'] = (train_df_pred['Units_Sold'] > median_units).astype(int)

    st.info(f"**Definisi Target:** Penjualan Tinggi = Unit Terjual > {median_units:.1f} (Median Unit Terjual)")

    # Fitur yang digunakan untuk prediksi
    # Mengambil daftar fitur yang relevan untuk prediksi (sesuai dengan notebook)
    all_possible_features = ['Battery_Capacity_kWh', 'Discount_Percentage', 'Region', 'Brand',
                   'Vehicle_Type', 'Customer_Segment', 'Fast_Charging_Option', 'Revenue'] # Menambahkan Revenue

    selected_features_pred = st.multiselect(
        'Pilih fitur untuk digunakan dalam model prediksi',
        all_possible_features,
        default=['Battery_Capacity_kWh', 'Discount_Percentage', 'Vehicle_Type', 'Region', 'Customer_Segment'] # Default yang lebih representatif
    )

    if len(selected_features_pred) < 1:
        st.warning('⚠️ Harap pilih minimal 1 fitur untuk membangun model prediksi.')
    else:
        if st.button('Latih Model'):
            # Persiapan data
            X = train_df_pred[selected_features_pred]
            y = train_df_pred['High_Sales']

            # Handle missing values (konsisten dengan fungsi load_data)
            for col in X.columns:
                 if X[col].dtype == 'object':
                     mode_val = X[col].mode()
                     if not mode_val.empty:
                         X[col].fillna(mode_val[0], inplace=True)
                     else:
                         X[col].fillna('Unknown', inplace=True) # Fallback if mode is empty
                 else:
                     X[col].fillna(X[col].mean(), inplace=True)


            # Membagi data menjadi set pelatihan dan validasi
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Menambah stratify

            # Identifikasi kolom kategorikal dan numerikal dalam fitur yang dipilih
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()

            # Membuat preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough' # Biarkan kolom lain yang tidak diproses
            )

            # Membuat pipeline model Logistic Regression
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')) # Ganti solver
            ])

            # Melatih model
            st.subheader('⏳ Melatih Model...')
            with st.spinner('Model sedang dilatih...'):
                model_pipeline.fit(X_train, y_train)
            st.success('✅ Model selesai dilatih!')

            # Menyimpan model dan preprocessor (untuk Prediksi Individu dan Model Performance)
            try:
                joblib.dump(model_pipeline, 'logistic_regression_pipeline.pkl')
                joblib.dump(preprocessor, 'prediction_preprocessor_fitted.pkl')
                joblib.dump(selected_features_pred, 'model_features_list.pkl') # Simpan daftar fitur
                st.info("Model dan preprocessor berhasil disimpan. Anda sekarang dapat melihat 'Model Performance' atau melakukan 'Prediksi Individu'.")
            except Exception as e:
                st.error(f"⚠️ Gagal menyimpan model atau preprocessor: {e}")


# Bagian Evaluasi Model (dipisahkan untuk kejelasan)
elif analysis_option == 'Model Performance':
    st.header('📊 Evaluasi Model Prediksi')
    st.markdown("Bagian ini menampilkan metrik evaluasi dari model Logistic Regression yang telah dilatih pada dataset train.")

    # Memuat model, preprocessor, dan daftar fitur yang sudah dilatih
    try:
        model_pipeline = joblib.load('logistic_regression_pipeline.pkl')
        preprocessor_fitted = joblib.load('prediction_preprocessor_fitted.pkl')
        selected_features_pred = joblib.load('model_features_list.pkl')

        st.info("Model dan preprocessor berhasil dimuat.")

        # Membuat variabel target biner (lagi, untuk memastikan konsistensi)
        median_units = train_df['Units_Sold'].median()
        train_df_pred = train_df.copy()
        train_df_pred['High_Sales'] = (train_df_pred['Units_Sold'] > median_units).astype(int)

        # Menggunakan fitur yang sama saat melatih model
        X = train_df_pred[selected_features_pred]
        y = train_df_pred['High_Sales']

        # Handle missing values (konsisten dengan pelatihan)
        for col in X.columns:
             if X[col].dtype == 'object':
                 mode_val = X[col].mode()
                 if not mode_val.empty:
                     X[col].fillna(mode_val[0], inplace=True)
                 else:
                     X[col].fillna('Unknown', inplace=True) # Fallback
             else:
                 X[col].fillna(X[col].mean(), inplace=True)


        # Membagi data (konsisten dengan pelatihan)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


        # Prediksi pada data validasi
        y_pred = model_pipeline.predict(X_val)
        y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]

        st.subheader('Metrik Evaluasi pada Dataset Validasi')

        col1, col2 = st.columns(2)

        with col1:
            # Akurasi
            accuracy = accuracy_score(y_val, y_pred)
            st.metric("Akurasi", f"{accuracy:.3f}")

            # Classification Report
            st.subheader('Classification Report')
            report = classification_report(y_val, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3))

        with col2:
            # Confusion Matrix
            st.subheader('Confusion Matrix')
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            st.pyplot(fig)

        # ROC Curve
        st.subheader('Kurva ROC')
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Kurva ROC (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Feature Importance
        st.subheader('Importansi Fitur')
        st.markdown("""
            Untuk Logistic Regression, "importansi" fitur sering diinterpretasikan dari
            magnitudo (nilai absolut) koefisien setelah fitur distandardisasi dan di-encode. Nilai absolut koefisien yang lebih tinggi
            menunjukkan pengaruh yang lebih besar terhadap kemungkinan (log-odds) target prediksi.
        """)

        try:
            classifier = model_pipeline.named_steps['classifier']

            # Dapatkan nama fitur setelah one-hot encoding dari preprocessor yang sudah fit
            feature_names_out = preprocessor_fitted.get_feature_names_out()
            coef = classifier.coef_[0]

            feature_importance = pd.DataFrame({
                'Fitur': feature_names_out,
                'Koefisien (Absolut)': np.abs(coef)
            }).sort_values('Koefisien (Absolut)', ascending=False)

            # Visualisasikan top N fitur terpenting
            top_n_features = st.slider("Tampilkan N Fitur Teratas", min_value=5, max_value=min(len(feature_importance), 30), value=15)

            fig, ax = plt.subplots(figsize=(10, min(top_n_features*0.4, 10))) # Sesuaikan tinggi plot
            sns.barplot(x='Koefisien (Absolut)', y='Fitur', data=feature_importance.head(top_n_features), ax=ax, palette='viridis')
            ax.set_title(f'Top {top_n_features} Fitur Paling Penting (Koefisien Absolut)')
            ax.set_xlabel('Nilai Koefisien Absolut')
            ax.set_ylabel('Fitur')
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
             st.error(f"⚠️ Tidak dapat menampilkan importansi fitur. Error: {e}")
             st.info("Pastikan model dan preprocessor telah dilatih dan disimpan dengan benar, serta versi scikit-learn mendukung `get_feature_names_out()`.")


    except FileNotFoundError:
        st.warning("⚠️ File model atau preprocessor tidak ditemukan. Silakan jalankan bagian 'Model Prediksi Penjualan' terlebih dahulu untuk melatih dan menyimpan model.")
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat model atau preprocessor: {e}")


# Bagian Prediksi Individu
elif analysis_option == 'Prediksi Individu':
    st.header('🔮 Prediksi Penjualan Kendaraan Listrik')
    st.markdown("Masukkan detail kendaraan untuk memprediksi apakah penjualannya akan tinggi atau rendah berdasarkan model yang sudah dilatih.")

    # Memuat model dan preprocessor
    try:
        model_pipeline = joblib.load('logistic_regression_pipeline.pkl')
        preprocessor_fitted = joblib.load('prediction_preprocessor_fitted.pkl')
        selected_features_pred = joblib.load('model_features_list.pkl') # Muat daftar fitur

        st.info("Model dan preprocessor berhasil dimuat. Siap untuk prediksi.")

        st.subheader('Masukkan Data Kendaraan')

        input_data = {}

        # Buat form input berdasarkan fitur yang digunakan model
        with st.form("prediction_form"):
            # Urutan input sesuai dengan selected_features_pred
            for feature in selected_features_pred:
                # Cek tipe data asli di train_df untuk menentukan jenis input
                if feature in train_df.columns:
                    dtype = train_df[feature].dtype

                    # Tampilkan label fitur
                    st.write(f"**{feature}**") # Menampilkan nama fitur sebagai label

                    if pd.api.types.is_numeric_dtype(dtype):
                        # Untuk numerik, gunakan slider atau number_input
                        min_val = float(train_df[feature].min()) if not train_df[feature].isnull().all() else 0.0
                        max_val = float(train_df[feature].max()) if not train_df[feature].isnull().all() else 100.0
                        mean_val = float(train_df[feature].mean()) if not train_df[feature].isnull().all() else (min_val + max_val) / 2

                        # Menggunakan number_input untuk kebebasan input nilai numerik
                        # Tambahkan label kosong '' karena label sudah ditampilkan di st.write
                        input_data[feature] = st.number_input(
                            '', # Label kosong
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val) / 100.0 or 1.0, # Langkah input
                            key=f"input_{feature}" # Tambahkan key unik
                        )
                    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                         # Untuk kategorikal, gunakan selectbox
                         unique_values = sorted(train_df[feature].dropna().unique().tolist())
                         if 'Unknown' not in unique_values: # Tambahkan 'Unknown' jika belum ada
                             unique_values.insert(0, 'Unknown')
                         # Tambahkan label kosong '' karena label sudah ditampilkan di st.write
                         input_data[feature] = st.selectbox(
                             '', # Label kosong
                             unique_values,
                             key=f"input_{feature}" # Tambahkan key unik
                         )
                    else:
                        # Fallback untuk tipe data lain
                         # Tambahkan label kosong '' karena label sudah ditampilkan di st.write
                        input_data[feature] = st.text_input('', key=f"input_{feature}") # Tambahkan key unik

                    st.markdown("---") # Tambahkan garis pemisah setelah setiap input

            submitted = st.form_submit_button("Prediksi")

            if submitted:
                # Konversi input menjadi DataFrame
                input_df = pd.DataFrame([input_data])

                # Pastikan kolom di input_df sama dengan fitur yang digunakan model, tambahkan missing jika perlu
                for feature in selected_features_pred:
                     if feature not in input_df.columns:
                         # Tentukan nilai default berdasarkan tipe data asli
                         if pd.api.types.is_numeric_dtype(train_df[feature].dtype):
                             input_df[feature] = train_df[feature].mean() # Isi dengan mean jika numerik
                         elif pd.api.types.is_object_dtype(train_df[feature].dtype) or pd.api.types.is_categorical_dtype(train_df[feature].dtype):
                             input_df[feature] = 'Unknown' # Isi dengan 'Unknown' jika kategorikal
                         else:
                             input_df[feature] = None # Default None untuk tipe lain

                # Lakukan prediksi
                try:
                    prediction = model_pipeline.predict(input_df)
                    prediction_proba = model_pipeline.predict_proba(input_df)[:, 1]

                    st.subheader('Hasil Prediksi')
                    if prediction[0] == 1:
                        st.success(f"Prediksi: **Penjualan Tinggi**")
                    else:
                        st.info(f"Prediksi: **Penjualan Rendah**")

                    st.write(f"Probabilitas Penjualan Tinggi: **{prediction_proba[0]:.2f}**")

                    # Interpretasi sederhana (opsional)
                    if prediction_proba[0] > 0.7:
                        st.write("💡 Interpretasi: Model sangat yakin bahwa penjualan kendaraan ini akan tinggi.")
                    elif prediction_proba[0] < 0.3:
                         st.write("💡 Interpretasi: Model sangat yakin bahwa penjualan kendaraan ini akan rendah.")
                    else:
                         st.write("💡 Interpretasi: Prediksi berada di area abu-abu, menunjukkan ketidakpastian.")


                except Exception as e:
                    st.error(f"⚠️ Terjadi kesalahan saat melakukan prediksi: {e}")
                    st.error("Pastikan input data sesuai dengan fitur yang diharapkan model.")
                    st.write("Fitur yang digunakan oleh model:", selected_features_pred)


    except FileNotFoundError:
        st.warning("⚠️ File model atau preprocessor tidak ditemukan. Silakan jalankan bagian 'Model Prediksi Penjualan' terlebih dahulu untuk melatih dan menyimpan model.")
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat model atau preprocessor: {e}")
