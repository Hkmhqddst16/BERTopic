
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px

# Judul Aplikasi
st.title("Pencarian Kalimat dengan Sentence-BERT")

# Deskripsi Aplikasi
st.write("""
Aplikasi ini memungkinkan Anda mencari kalimat yang paling mirip dari daftar kalimat yang telah disediakan menggunakan model Sentence-BERT.
""")

# Daftar Kalimat
kalimat = [
    "AI digunakan untuk mengolah data dalam jumlah besar.",
    "AI sering diterapkan dalam pengolahan data skala besar.",
    "Penggunaan AI dapat membantu menganalisis data yang sangat banyak.",
    "Data dalam jumlah besar menjadi lebih mudah diproses dengan bantuan AI.",
    "AI mempermudah analisis data besar dengan lebih efisien.",
    "Machine learning memungkinkan mesin belajar dari pengalaman.",
    "Algoritma machine learning memproses data untuk menghasilkan prediksi.",
    "Jaringan saraf tiruan sering digunakan dalam deep learning.",
    "Model AI berkembang dengan pelatihan menggunakan dataset berkualitas.",
    "Kecerdasan buatan banyak diterapkan di berbagai industri.",
    "Deep learning meningkatkan kemampuan AI dalam memahami pola kompleks.",
    "AI dapat mengenali gambar dengan akurasi tinggi.",
    "Natural Language Processing memungkinkan AI memahami bahasa manusia.",
    "Robotik menggunakan AI untuk meningkatkan otomatisasi produksi.",
    "AI membantu dalam deteksi dini penyakit melalui analisis data medis.",
    "Sistem rekomendasi menggunakan AI untuk menyarankan produk kepada pengguna.",
    "AI berperan dalam pengembangan kendaraan otonom.",
    "Pengolahan bahasa alami digunakan AI dalam penerjemahan otomatis.",
    "AI meningkatkan efisiensi operasional di sektor logistik.",
    "Pengenalan suara oleh AI memudahkan interaksi manusia dengan teknologi.",
    "AI digunakan dalam analisis sentimen media sosial.",
    "Keamanan siber memanfaatkan AI untuk mendeteksi ancaman.",
    "AI membantu dalam pengelolaan energi secara lebih efektif.",
    "Pendidikan diperkaya dengan AI melalui pembelajaran yang dipersonalisasi.",
    "AI digunakan dalam peramalan cuaca untuk prediksi yang lebih akurat.",
    "Teknologi AI mendukung pengembangan smart home.",
    "AI membantu dalam optimasi rantai pasokan perusahaan.",
    "Penggunaan AI dalam industri finansial meningkatkan analisis risiko.",
    "AI digunakan dalam pencarian informasi yang lebih relevan di internet.",
    "AI membantu dalam pengelolaan sampah dengan sistem pengenalan objek.",
    "Sistem keamanan menggunakan AI untuk pengawasan yang lebih canggih.",
    "AI digunakan dalam pembuatan konten digital secara otomatis.",
    "Analisis pasar dengan AI membantu bisnis memahami tren konsumen.",
    "AI meningkatkan pengalaman pengguna melalui personalisasi layanan.",
    "Penggunaan AI dalam kesehatan mental membantu dalam diagnosis awal.",
    "AI digunakan dalam industri hiburan untuk menciptakan efek visual yang realistis.",
    "Sistem pembayaran otomatis memanfaatkan AI untuk deteksi penipuan.",
    "AI membantu dalam penelitian ilmiah dengan menganalisis data eksperimen.",
    "Penggunaan AI dalam agrikultur meningkatkan hasil panen melalui analisis tanah.",
    "AI digunakan dalam desain arsitektur untuk menciptakan bangunan yang efisien.",
    "AI membantu dalam pelacakan inventaris secara real-time.",
    "Penggunaan AI dalam pemasaran digital meningkatkan efektivitas kampanye.",
    "AI digunakan dalam pengembangan obat dengan mempercepat penemuan molekul baru.",
    "Sistem pendukung keputusan perusahaan memanfaatkan AI untuk strategi bisnis.",
    "AI membantu dalam manajemen risiko keuangan dengan analisis prediktif.",
    "Penggunaan AI dalam transportasi umum meningkatkan ketepatan waktu layanan."
]


# Load Model Sentence-BERT
@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Model yang mendukung Bahasa Indonesia
    return model

model = load_model()

# Generate Embeddings untuk Kalimat
# @st.cache_data
def generate_embeddings(model, sentences):
    embeddings = model.encode(sentences)
    return embeddings

embeddings = generate_embeddings(model, kalimat)

# Input Query dari Pengguna
query = st.text_input("Masukkan kalimat pencarian Anda:")

if query:
    # Generate Embedding untuk Query
    query_embedding = model.encode([query])

    # Hitung Cosine Similarity
    cosine_similarities = cosine_similarity(query_embedding, embeddings)
    similarity_scores = cosine_similarities[0]

    # Urutkan dan Ambil 5 Teratas
    sorted_indices = np.argsort(similarity_scores)[::-1]
    lima_besar = sorted_indices[:5]

    # Siapkan Hasil
    results = [
        {
            "Document": index + 1,
            "Kalimat": kalimat[index],
            "Skor Kemiripan": f"{similarity_scores[index]:.2f}"
        }
        for index in lima_besar if similarity_scores[index] > 0
    ]

    if results:
        st.subheader("Hasil Pencarian:")
        for result in results:
            st.write(f"**Document {result['Document']}**: {result['Kalimat']} (Skor Kemiripan: {result['Skor Kemiripan']})")

 # Visualisasi dengan Plotly
        st.subheader("Visualisasi Skor Kemiripan:")
        df = {
            "Kalimat": [result["Kalimat"] for result in results],
            "Skor Kemiripan": [result["Skor Kemiripan"] for result in results]
        }

        fig = px.bar(
            df,
            x="Kalimat",
            y="Skor Kemiripan",
            title="Skor Kemiripan Top 5 Kalimat",
            labels={"Kalimat": "Kalimat", "Skor Kemiripan": "Skor"},
            height=400
        )
        st.plotly_chart(fig)

    else:
        st.write("Data tidak ada yang cocok dengan query Anda.")

st.subheader("Daftar Semua Kalimat:")
for i, kal in enumerate(kalimat, 1):
    st.write(f"{i}. {kal}")

# Visualisasi Heatmap
if results:
    # ... (Kode untuk menghitung matriks kemiripan)
    fig = px.imshow(similarity_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig)

