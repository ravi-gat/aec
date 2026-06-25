import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

text = "The quick brown foxes are jumping over the lazy dogs."

tokens = word_tokenize(text)
words = [w.lower() for w in tokens if w.isalpha()]

stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w not in stop_words]

stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered_words]

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]

print("Filtered:", filtered_words)
print("Stemmed: ", stemmed)
print("Lemmatized:", lemmatized)
