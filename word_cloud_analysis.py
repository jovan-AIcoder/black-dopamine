import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
# load dataset
df = pd.read_csv('dataset/raw_youtube_comments.csv')
text_data = df['text'].to_string()
# stopwords (Indonesian [ID])
custom_stopwords = set(STOPWORDS).union({
    'yang', 'yg', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'akan', 'bisa', 'jadi', 'kalo', 'kalau', 'bikin', 'ada',
    'gak', 'ga', 'ngga', 'nggak', 'biar', 'sama', 'buat', 'tetap', 'tapi', 'saya', 'gue', 'gw', 'aku', 'bang', 'gua', 'juga', 'udah', 'ya', 'sudah', 'kak', 'nya', 
    'utk', 'lg', 'lagi', 'si', 'nih', 'tuh', 'sih', 'bro', 'rumah', 'Terimakasih', 'Terima', 'kasih', 'thank', 'Thanks', 'klo', 'Alhamdulillah', 'thankyou', 'editor', 'min', 'Makasih', 'Bismillah', 'sy'
})
# generate wordcloud
wordcloud = WordCloud(
    width=1200,
    height=800,
    background_color='black',  
    colormap='PuRd',           
    stopwords=custom_stopwords,
    min_font_size=10,
    max_words=100
).generate(text_data)

# plot
plt.figure(figsize=(12, 8), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# save and show plot
output_image = 'wordcloud.png'
plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
print(f"Word Cloud is saved as: {output_image}")
plt.show()