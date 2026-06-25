import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

text = "This is a sample text. It is used for testing the NLTK word frequency counting functionality."

tokens = word_tokenize(text)

words = [word.lower() for word in tokens if word.isalpha()]

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

freq_dist = FreqDist(filtered_words)

print("Word Frequencies:")
for word, count in freq_dist.items():
    print(f"{word}: {count}")

