import streamlit as st
from bertopic import BERTopic

# Judul aplikasi
st.title("Antarmuka BERTopic dengan Streamlit")

# Deskripsi
st.write("Selamat datang di antarmuka BERTopic! Ini adalah contoh sederhana menggunakan Streamlit dan BERTopic.")

# Input teks
user_input = st.text_area("Masukkan teks untuk topik modeling:")

# Tombol untuk memproses
if st.button("Proses Teks"):
    if user_input:
        # Membuat model BERTopic
        topic_model = BERTopic()
        
        # Melakukan topik modeling
        topics, probs = topic_model.fit_transform([user_input])
        
        # Menampilkan hasil
        st.write("Topik yang ditemukan:")
        st.write(topics)
    else:
        st.write("Silakan masukkan teks terlebih dahulu.")

# Pesan tambahan
st.write("Terima kasih telah menggunakan aplikasi ini!")
