import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from spam_classifier import train_data

nltk.download('stopwords')
vectorizer = CountVectorizer()
classifier = MultinomialNB()
train_all = pd.DataFrame()
STOPWORDS = set(stopwords.words('english'))


def stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])


def remove_chinese_characters(text):
    if len(re.findall('[А-Яа-я]+', text)) > 0:
        return text

    return re.sub("([^\x00-\x7F])+", "", text)


def remove_single_character_words(text):
    return ' '.join([word for word in text.split() if len(word) > 1])


def prepare_train_en(df):
    # Предобработка текста

    # Удалим одну пустую строку
    df.dropna(inplace=True)

    # Удалим все китайские символы, а также все другие, которые не подходят под кодировку ASCII, если встретятся
    df.email = df.email.apply(remove_chinese_characters)

    # Удалим все односимвольные слова
    df.email = df.email.apply(remove_single_character_words)

    # Удалим слова NUMBER и URL
    df.email = df.email.str.replace('NUMBER', '')
    df.email = df.email.str.replace('URL', '')

    # Удалим шумовые слова
    df.email = df.email.apply(stopwords)

    # Удалим символы пунктуации
    df.email = df.email.str.replace('[^\\w\\s]', '')

    # Удалим символы '_', '_ '
    df.email = df.email.str.replace('_', '')
    df.email = df.email.str.replace('_ ', '')

    # Приведем все слова к нижнему регистру, удалим пробелы в начале и конце предложений, удалим лишние пробелы
    df.email = df.email.str.lower()
    df.email = df.email.str.strip()
    df.email = (df.email.str.split()).str.join(' ')

    return df


def prepare_message(text):
    text = text.lower()
    text = re.sub('[^\\w\\s]', '', text)
    text = text.strip()
    text = remove_single_character_words(text)
    text = stopwords(text)
    return text


def train():
    global vectorizer
    global classifier
    global train_all

    # Подготовим обучающую выборку сообщений на русском
    train_rus = pd.DataFrame(data=train_data)
    train_rus.columns = ['email', 'label']
    train_rus.label = train_rus.label.apply(lambda x: 1 if x == 'SPAM' else 0)

    train_en = pd.read_csv('spam_or_not_spam.csv')
    train_en = prepare_train_en(train_en)

    train_all = pd.concat([train_rus, train_en])

    X = train_all.email
    y = train_all.label

    counts = vectorizer.fit_transform(X)
    classifier.fit(counts, y)


def model_scoring():
    global vectorizer
    global classifier
    global train_all

    X = train_all.email
    y = train_all.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

    counts = vectorizer.fit_transform(X_train)
    classifier.fit(counts, y_train)

    y_pred = classifier.predict(vectorizer.transform(X_test))
    return round(f1_score(y_test, y_pred), 2)


def classify(email):
    global vectorizer
    global classifier

    email = prepare_message(email)
    predict = classifier.predict(vectorizer.transform([email]))

    if predict[0] == 1:
        return True
    else:
        return False


