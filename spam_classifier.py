import re
import pandas as pd

from math import log


# Определим переменные для хранения вероятности встретить спам и не встретить спам
pA = 0.0
pNotA = 0.0

# Определим словари, в которых будут храниться количества спам-слов и неспам-слов
spam_words = {}
not_spam_words = {}

# Класс сообщения
SPAM = 'SPAM'
NOT_SPAM = 'NOT_SPAM'


# Функция разбивки предложения на слова
def get_words(text):
    word_pattern = re.compile('\\w+')
    words = [word.lower() for word in word_pattern.findall(text)]

    return words


# Определяем функцию заполнения словарей
def calculate_word_frequencies(body, label):
    words = get_words(body)

    if label == SPAM:
        for word in words:
            if word not in spam_words:
                spam_words[word] = 1
            else:
                spam_words[word] += 1

    elif label == NOT_SPAM:
        for word in words:
            if word not in not_spam_words:
                not_spam_words[word] = 1
            else:
                not_spam_words[word] += 1

    # Обрабатываем проблему неизвестных слов
    for k in spam_words.keys():
        spam_words[k] += 1

    for k in not_spam_words.keys():
        not_spam_words[k] += 1

    return spam_words, not_spam_words


def train():
    # Подготовим словари:
    global spam_words
    global not_spam_words

    # Объявляем переменные вероятностей
    global pA
    global pNotA

    global train_data

    spam_words, not_spam_words = {}, {}
    pA, pNotA = 0.0, 0.0

    # Считаем вероятности
    pA = log(len(list(element for element in train_data if element[1] == SPAM)) / len(train_data))
    pNotA = log(len(list(element for element in train_data if element[1] == NOT_SPAM)) / len(train_data))

    # Заполняем словари
    for element in train_data:
        calculate_word_frequencies(*element)


def calculate_P_Bi_A(word, label):
    p = 0.0
    all_words = len(list(spam_words.keys()) + list(not_spam_words.keys()))

    if label == SPAM:
        # Обрабатываем проблему неизвестных слов
        # Для слова, которого нет в обучающей выборке притворимся, будто оно встречается один раз
        p = (1 if spam_words.get(word) is None else spam_words.get(word)) / (all_words + len(spam_words.keys()))

    if label == NOT_SPAM:
        # Здесь делаем тоже самое
        p = (1 if not_spam_words.get(word) is None else not_spam_words.get(word)) / (
                    all_words + len(not_spam_words.keys()))

    return log(p)


def calculate_P_B_A(text, label):
    amount = 0
    words = get_words(text)

    for word in words:
        amount += calculate_P_Bi_A(word, label)

    return amount


def classify(email):
    pBA_S = calculate_P_B_A(email, SPAM)
    pBA_NS = calculate_P_B_A(email, NOT_SPAM)

    if pA + pBA_S > pNotA + pBA_NS:
        return True
    else:
        return False


# Тестовые данные
train_data = [
    ['Купите новое чистящее средство', SPAM],
    ['Купи мою новую книгу', SPAM],
    ['Подари себе новый телефон', SPAM],
    ['Добро пожаловать и купите новый телевизор', SPAM],
    ['Привет давно не виделись', NOT_SPAM],
    ['Довезем до аэропорта из пригорода всего за 399 рублей', SPAM],
    ['Добро пожаловать в Мой Круг', NOT_SPAM],
    ['Я все еще жду документы', NOT_SPAM],
    ['Приглашаем на конференцию Data Science', NOT_SPAM],
    ['Потерял твой телефон напомни', NOT_SPAM],
    ['Порадуй своего питомца новым костюмом', SPAM]
]


