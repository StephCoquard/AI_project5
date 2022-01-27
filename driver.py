from os import listdir
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
# test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

train_path = "aclImdb/train/"
test_path = "imdb_te.csv"

stopwords = []


def init_stopwords():
    global stopwords
    sw = open("stopwords.en.txt", "r", encoding="utf8")
    stopwords = sw.read().split("\n")
    sw.close()


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    fo = open(name, "w", encoding='utf8')
    fo.write("row_number,text,polarity\n")

    row_number = 0
    for text in listdir(inpath + "pos"):
        fi = open(inpath + "pos/" + text, "r", encoding='utf8')
        fo.write(str(row_number) + "," + preprocess_row(fi.read()) + ",1" + "\n")
        row_number += 1

    for text in listdir(inpath + "neg"):
        fi = open(inpath + "neg/" + text, "r", encoding='utf8')
        fo.write(str(row_number) + "," + preprocess_row(fi.read()) + ",0" + "\n")
        row_number += 1


def preprocess_row(sentence):
    words = sentence
    words = words.replace("<br />", " ")
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    words = words.lower().translate(replace_punctuation)
    words = words.split()
    words = [word for word in words if word not in stopwords]
    words = ' '.join(words)

    return words


def init_classifier(vectorizer):
    transform = vectorizer.fit_transform(training['text'])
    c = SGDClassifier(loss="hinge", penalty="l1")
    c.fit(transform, training['polarity'])
    return c


def get_results(vectorizer):
    tt = count_vectorizer.transform(test_set['text'])
    return classifier.predict(tt)


if __name__ == "__main__":
    init_stopwords()
    imdb_data_preprocess(train_path)

    training = pd.read_csv("imdb_tr.csv")
    test_set = pd.read_csv(test_path, encoding="ISO-8859-1")
    test_set['text'] = test_set['text'].apply(preprocess_row)

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    count_vectorizer = CountVectorizer(stop_words=stopwords)
    classifier = init_classifier(count_vectorizer)

    unigram_fo = open("unigram.output.txt", "w")
    for result in get_results(count_vectorizer):
        unigram_fo.write(str(result) + "\n")

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''
    count_vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    classifier = init_classifier(count_vectorizer)

    bigram_fo = open("bigram.output.txt", "w")
    for result in get_results(count_vectorizer):
        bigram_fo.write(str(result) + "\n")

    '''train a SGD classifier using unigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigramtfidf.output.txt'''
    count_vectorizer = TfidfVectorizer(stop_words=stopwords)
    classifier = init_classifier(count_vectorizer)

    unigramtfidf_fo = open("unigramtfidf.output.txt", "w")
    for result in get_results(count_vectorizer):
        unigramtfidf_fo.write(str(result) + "\n")

    '''train a SGD classifier using bigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to bigramtfidf.output.txt'''
    count_vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    classifier = init_classifier(count_vectorizer)

    bigramtfidf_fo = open("bigramtfidf.output.txt", "w")
    for result in get_results(count_vectorizer):
        bigramtfidf_fo.write(str(result) + "\n")
