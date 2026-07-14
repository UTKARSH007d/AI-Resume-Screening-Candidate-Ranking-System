import re



from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

# Create these ONCE
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_html(text):
    text = re.sub(r"<.*?>", "", text)
    return text


def preprocess_text(text):
    # Check input type
    if not isinstance(text, str):
        raise TypeError("Input to preprocess_text() must be a string.")

    # Check empty string
    if text.strip() == "":
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML
    text = remove_html(text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    clean_tokens = []

    for word in tokens:
        if word not in stop_words:
            clean_tokens.append(lemmatizer.lemmatize(word))

    # Join tokens back into a string
    text = " ".join(clean_tokens)

    return text
