# Imports
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from matplotlib import pyplot as plt

"""nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')"""

# Importing the spotify million song dataset
df = pd.read_csv("spotify_million_dataset.csv")

# Data Cleaning
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

df['clean_song'] = df['text'].apply(remove_punctuation)
df['clean_song'] = df['clean_song'].apply(remove_stopwords)
df['clean_song'] = df['clean_song'].apply(lemmatize_text)

# Data Prep
df['song_len'] = df['text'].str.len()
df['clean_song_len'] = df['clean_song'].str.len()

# Data Exploratory Analysis
""" Get top 20 artists  """
artist_df = (
    df['artist'].value_counts()
    .head(10)
    .rename_axis('values')
    .reset_index(name = 'counts')
)

print(artist_df.head())    # Check new df

""" Song Length Stats   """
print("Avg. song length: ", df['song_len'].mean())
print("Avg. song length - no stopwords: ", df['clean_song_len'].mean())
print("Max. song length: ", df['song_len'].max())
print("Max. song length - no stopwords: ", df['clean_song_len'].max())
print("Min. song length: ", df['song_len'].min())
print("Min. song length - no stopwords: ", df['clean_song_len'].min())



# Data Viz

#ax1 = artist_df.plot(x = 'counts', kind = 'hist')
#ax2 = df.plot.scatter(x = 'artist', y = )

df['artist'].hist()
plt.xlim(0, 10)
plt.ylim(0, 200)
plt.show()

