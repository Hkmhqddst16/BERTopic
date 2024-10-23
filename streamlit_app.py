import streamlit as st
import pandas as pd
import WordCloud
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="BERTopic", page_icon="🤗")

df = pd.read_csv("bertopic_results.csv")
st.title("Topic modeling using BERTopic")

st.markdown("Mengambil 15 topik dan berikut 10 contoh dari isi topik :")
st.dataframe(df.sample(10))

st.subheader("Distribution of topics in the collection of articles.")
topic_distribution = df['topic'].value_counts().sort_index()
st.bar_chart(topic_distribution)

def contnet_topwords(topic):
    topic_df = df[df['topic'] == topic]
    text = ' '.join(topic_df['content'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    

st.subheader('kata teratas dan WordCloud dari topik')
topic_selected = st.selectbox('Select a Topic', sorted(df['topic'].unique()))

topic_df = df[df['topic'] == topic_selected]
all_content = ' '.join(topic_df['content'])

words = re.findall(r'\b\w+\b', all_content.lower())

word_counts = pd.Series(words).value_counts()
top_words = word_counts.head(10)
st.write(f"kata teratas dari topik {topic_selected}")
st.write(top_words)

top_articles = topic_df.head(10)  
st.write(f"Top 10 Topik {topic_selected}")
st.write(top_articles[['content', 'topic']])

st.write(f"Word Cloud topik {topic_selected}")
contnet_topwords(topic_selected)
