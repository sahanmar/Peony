import hashlib
import nltk

from typing import List, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


def create_hash(hash_args: List[Any]) -> str:
    sha = hashlib.sha256()
    sha.update(" ".join(hash_args).encode())
    return sha.hexdigest()


def stop_words_filter(tokens: List[str]) -> List[str]:
    stopset = set(stopwords.words("english"))
    return [token for token in tokens if token not in stopset]


def lemmatizer(tokens: List[str]):

    lemmatizer = WordNetLemmatizer()
    lemmatized_text: List[str] = []

    for token in tokens:
        if lemmatizer.lemmatize(token, pos="n") == token:
            if lemmatizer.lemmatize(token, pos="v") == token:
                lemmatized_text.append(lemmatizer.lemmatize(token, pos="a"))
            else:
                lemmatized_text.append(lemmatizer.lemmatize(token, pos="v"))
        else:
            lemmatized_text.append(lemmatizer.lemmatize(token, pos="n"))

    return lemmatized_text


def tokenizer(text: str) -> List[str]:
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text if isinstance(text, str) else " ")
    return [token.lower() for token in tokens if token.isalpha() == True]
