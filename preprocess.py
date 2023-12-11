import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_df(dataFrame):
  lem = WordNetLemmatizer()
  stop_words = stopwords.words('english')
  dataFrame = dataFrame.lower()
  dataFrame = dataFrame.translate(str.maketrans('', '', string.punctuation))
  #dataFrame = re.sub('[^a-zA-Z]', ' ', dataFrame)
  dataFrame = ' '.join([word for word in nltk.word_tokenize(dataFrame) if word not in stop_words])
  dataFrame = ' '.join(lem.lemmatize(word) for word in dataFrame.split())
  return dataFrame