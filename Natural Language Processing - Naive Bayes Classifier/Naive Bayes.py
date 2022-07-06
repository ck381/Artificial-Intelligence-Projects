#do not change the code in this cell
#preliminary imports

#set up nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('movie_reviews')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews

#for setting up training and testing data
import random

#useful other tools
import re
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from itertools import zip_longest
from nltk.probability import FreqDist
from nltk.classify.api import ClassifierI
import random



# do not change the code in this cell
def split_data(data, ratio=0.7):  # when the second argument is not given, it defaults to 0.7
    """
    Given corpus generator and ratio:
     - partitions the corpus into training data and test data, where the proportion in train is ratio,

    :param data: A corpus generator.
    :param ratio: The proportion of training documents (default 0.7)
    :return: a pair (tuple) of lists where the first element of the
            pair is a list of the training data and the second is a list of the test data.
    """

    data = list(data)
    n = len(data)
    train_indices = random.sample(range(n), int(n * ratio))
    test_indices = list(set(range(n)) - set(train_indices))
    train = [data[i] for i in train_indices]
    test = [data[i] for i in test_indices]
    return (train, test)


def get_train_test_data():
    # get ids of positive and negative movie reviews
    pos_review_ids = movie_reviews.fileids('pos')
    neg_review_ids = movie_reviews.fileids('neg')

    # split positive and negative data into training and testing sets
    pos_train_ids, pos_test_ids = split_data(pos_review_ids)
    neg_train_ids, neg_test_ids = split_data(neg_review_ids)
    # add labels to the data and concatenate
    training = [(movie_reviews.words(f), 'pos') for f in pos_train_ids] + [(movie_reviews.words(f), 'neg') for f in
                                                                           neg_train_ids]
    testing = [(movie_reviews.words(f), 'pos') for f in pos_test_ids] + [(movie_reviews.words(f), 'neg') for f in
                                                                         neg_test_ids]

    return training, testing


random.seed(candidateno)
training_data, testing_data = get_train_test_data()
print("The amount of training data is {}".format(len(training_data)))
print("The amount of testing data is {}".format(len(testing_data)))
print("The representation of a single data item is below")
print(training_data[0])

import nltk
from nltk.corpus import stopwords
import re
import string
import itertools
from nltk.probability import FreqDist
from collections import Counter
import pandas as pd
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')
stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
              "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
              "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing",
              "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has",
              "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
              "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself",
              "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself",
              "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other",
              "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's",
              "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll",
              "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
              "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were",
              "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
              "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm",
              "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're",
              "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would",
              "able", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj",
              "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also",
              "although", "always", "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore", "anyone",
              "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "arent", "arise", "around",
              "aside", "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become",
              "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe",
              "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause",
              "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains",
              "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight",
              "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even",
              "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff",
              "fifth", "first", "five", "fix", "followed", "following", "follows", "former", "formerly", "forth",
              "found", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving",
              "go", "goes", "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby",
              "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however", "hundred",
              "id", "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed", "index",
              "information", "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg",
              "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least",
              "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking",
              "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime",
              "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much",
              "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary",
              "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "nobody", "non",
              "none", "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained",
              "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord", "others",
              "otherwise", "outside", "overall", "owing", "p", "page", "pages", "part", "particular", "particularly",
              "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially",
              "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides",
              "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really", "recent",
              "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research",
              "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say", "saying",
              "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
              "sent", "seven", "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows",
              "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "somebody", "somehow",
              "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry",
              "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially",
              "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends",
              "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered",
              "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've",
              "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru",
              "thus", "til", "tip", "together", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying",
              "ts", "twice", "two", "u", "un", "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups",
              "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value",
              "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed",
              "welcome", "went", "werent", "whatever", "what'll", "whats", "whence", "whenever", "whereafter",
              "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod",
              "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within",
              "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero",
              "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", "appropriate", "associated", "best",
              "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider",
              "considering", "corresponding", "course", "currently", "definitely", "described", "despite", "entirely",
              "exactly", "example", "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch",
              "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably",
              "reasonably", "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third",
              "thorough", "thoroughly", "three", "well", "wonder"]


# Filters sentences
def cleanText(training_data):
    cleaned_data = []

    for doc, label in training_data:
        doc = list(doc)
        doc = [w for w in doc if w not in stop_words]
        doc = [re.sub('[^A-Za-z0-9]+', '', w) for w in doc]
        doc = [word.lower() for word in doc if word.isalpha()]
        lemmatizer = WordNetLemmatizer()
        doc = [lemmatizer.lemmatize(word) for word in doc]
        doc = (doc, label)
        cleaned_data.append(doc)

    return cleaned_data


# Builds a word list from the input sentences
def create_word_list(sentences):
    return list(itertools.chain(*sentences))


# Creates a frequency distribution from a word list
def create_freqDist(words):
    words = " ".join(word for word in words)
    fdist = FreqDist()
    for word in word_tokenize(words):
        fdist[word.lower()] += 1

    return fdist


# Calculates words with the biggest frequency difference
def most_frequent_words(fdist_pos, fdist_neg, k):
    positive = {}
    negative = {}

    for key in fdist_pos.keys():
        pos_count = fdist_pos.get(key)
        neg_count = fdist_neg.get(key)
        if neg_count == None:
            neg_count = 0
            positive[key] = 0
        else:
            positive[key] = pos_count - neg_count

    positive = sorted(positive.items(), key=lambda x: x[1], reverse=True)
    positive = [x[0] for x in positive]

    for key in fdist_neg.keys():
        neg_count = fdist_neg.get(key)
        pos_count = fdist_pos.get(key)

        if pos_count == None:
            pos_count = 0
            negative[key] = 0
        else:
            negative[key] = neg_count - pos_count

    negative = sorted(negative.items(), key=lambda x: x[1], reverse=True)
    negative = [x[0] for x in negative]

    return positive[:k], negative[:k]


# Returns frequency distributions for positive and negative reviews from the input training data
def generate_frequency_distributions(training_data):
    cleaned_pos_reviews = []
    cleaned_neg_reviews = []

    data = cleanText(training_data)

    for (doc, label) in data:
        if label == 'pos':
            cleaned_pos_reviews.append(doc)

        else:
            cleaned_neg_reviews.append(doc)

    pos_words = create_word_list(cleaned_pos_reviews)
    neg_words = create_word_list(cleaned_neg_reviews)

    fdist_pos = create_freqDist(pos_words)
    fdist_neg = create_freqDist(neg_words)

    return fdist_pos, fdist_neg


def create_dataframe(positive_words, negative_words):
    d = {'positive': positive_words, 'negative': negative_words}
    df = pd.DataFrame(data=d)
    print(df)


fdist_pos, fdist_neg = generate_frequency_distributions(training_data)
positive_words, negative_words = most_frequent_words(fdist_pos, fdist_neg, 10)
create_dataframe(positive_words, negative_words)


class WordListClassifier():

    # param pos: Positive wordlist
    # param neg: Negative wordlist
    def __init__(self, pos, neg):
        self._pos = pos
        self._neg = neg

        # param words: tokenized sentence

    def classify(self, words):
        score = 0

        for word in words:
            if word in self._pos:
                score += 1
            if word in self._neg:
                score -= 1

        if score < 0:
            return "neg"
        elif score == 0:
            return random.choice(["pos", "neg"])
        else:
            return "pos"


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# Evaluation class for the word list classifier
class Evaluate:

    # Constructor
    def __init__(self, testing_data, classifier):
        self.testing_data = testing_data
        self.classifier = classifier
        self.predictions = []
        self.goldstandards = []

    # Cleans the test data using the same techniques used to clean the training data

    def cleanTestData(self):

        cleaned_data = []

        for doc, label in self.testing_data:
            doc = list(doc)
            doc = [w for w in doc if w not in stop_words]
            doc = [re.sub('[^A-Za-z0-9]+', '', w) for w in doc]
            doc = [word.lower() for word in doc if word.isalpha()]
            lemmatizer = WordNetLemmatizer()
            doc = [lemmatizer.lemmatize(word) for word in doc]
            doc = (doc, label)
            cleaned_data.append(doc)

        self.testing_data = cleaned_data

    def create_predictions(self):

        self.cleanTestData()

        correct = 0

        for (doc, label) in self.testing_data:
            doc = list(doc)
            prediction = self.classifier.classify(doc)
            self.predictions.append(prediction)
            self.goldstandards.append(label)
        for (prediction, goldstandard) in zip(self.predictions, self.goldstandards):
            if prediction == goldstandard:
                correct += 1

    # Produces a confusion matrix off the testing data predictions from the classifier
    def confusion_matrix(self):

        self.TP = 0  # True positives
        self.FP = 0  # False postives
        self.TN = 0  # True negatives
        self.FN = 0  # False negatives

        for (prediction, goldstandard) in zip(self.predictions, self.goldstandards):
            if goldstandard == 'pos':
                if prediction == 'pos':
                    self.TP += 1
                else:
                    self.FN += 1

            elif prediction == 'pos':
                self.FP += 1
            else:
                self.TN += 1

    # Calculates and returns classifer f1 score
    def f1_score(self, precision, recall):
        score = 2 * precision * recall / (precision + recall)
        score = round(score, 2)
        return score

    # Calculates and returns classifer accuracy
    def accuracy(self):
        accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        accuracy = round(accuracy, 2)
        return accuracy

    # Calculates and return classifier precision score
    def precision(self):
        precision = self.TP / (self.FP + self.TP)
        precision = round(precision, 2)
        return precision

        # Calculates and return classifier recall score

    def recall(self):
        recall = self.TP / (self.TP + self.FN)
        recall = round(recall, 2)
        return recall


# ----------------------------------------------------------- #

classifier = WordListClassifier(positive_words, negative_words)  # Build a classifier
evaluate = Evaluate(testing_data, classifier)  # Create an evaluation module
evaluate.create_predictions()  # Generate predictions
evaluate.confusion_matrix()  # Build a confusion matrix
precision = evaluate.precision()  # Calculate precision
recall = evaluate.recall()  # Calculate recall
accuracy = evaluate.accuracy()  # Calculate accuracy
f1_score = evaluate.f1_score(precision, recall)  # Calculate F1 Score

# ----------------------------------------------------------- #
# Print metrics

print('Accuracy = ' + str(accuracy))
print('Precision = ' + str(precision))
print('Recall = ' + str(recall))
print('F1 Score = ' + str(f1_score))

import math
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class NBClassifier:

    def __init__(self, training_data):
        self.labels = ["pos", "neg"]
        self.priors = None
        self.cond_probs = None
        self.training_data = training_data
        self.known_vocabulary = []

    # Cleans the training data
    def clean_training_data(self):

        cleaned_data = []

        for doc, label in self.training_data:
            stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any",
                          "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
                          "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't",
                          "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few",
                          "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
                          "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
                          "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll",
                          "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself",
                          "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or",
                          "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan",
                          "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such",
                          "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then",
                          "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
                          "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when",
                          "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn",
                          "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
                          "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm",
                          "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll",
                          "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's",
                          "who's", "why's", "would", "able", "abst", "accordance", "according", "accordingly", "across",
                          "act", "actually", "added", "adj", "affected", "affecting", "affects", "afterwards", "ah",
                          "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst",
                          "announce", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway",
                          "anyways", "anywhere", "apparently", "approximately", "arent", "arise", "around", "aside",
                          "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become",
                          "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind",
                          "believe", "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came",
                          "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes",
                          "contain", "containing", "contains", "couldnt", "date", "different", "done", "downwards",
                          "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere",
                          "end", "ending", "enough", "especially", "et", "etc", "even", "ever", "every", "everybody",
                          "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first",
                          "five", "fix", "followed", "following", "follows", "former", "formerly", "forth", "found",
                          "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives",
                          "giving", "go", "goes", "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence",
                          "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home",
                          "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance",
                          "important", "inc", "indeed", "index", "information", "instead", "invention", "inward", "itd",
                          "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l",
                          "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let",
                          "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd",
                          "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime",
                          "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr",
                          "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly",
                          "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new",
                          "next", "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos",
                          "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok", "okay",
                          "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall",
                          "owing", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps",
                          "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp",
                          "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud",
                          "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd",
                          "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless",
                          "regards", "related", "relatively", "research", "respectively", "resulted", "resulting",
                          "results", "right", "run", "said", "saw", "say", "saying", "says", "sec", "section", "see",
                          "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven",
                          "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows",
                          "significant", "significantly", "similar", "similarly", "since", "six", "slightly",
                          "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes",
                          "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify",
                          "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully",
                          "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th",
                          "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered",
                          "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon",
                          "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug",
                          "throughout", "thru", "thus", "til", "tip", "together", "took", "toward", "towards", "tried",
                          "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless",
                          "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully",
                          "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol",
                          "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent",
                          "whatever", "what'll", "whats", "whence", "whenever", "whereafter", "whereas", "whereby",
                          "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod", "whoever",
                          "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within",
                          "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre",
                          "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate",
                          "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly",
                          "concerning", "consequently", "consider", "considering", "corresponding", "course",
                          "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going",
                          "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated",
                          "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably",
                          "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough",
                          "thoroughly", "three", "well", "wonder"]
            doc = list(doc)
            doc = [w for w in doc if w not in stop_words]
            doc = [re.sub('[^A-Za-z0-9]+', '', w) for w in doc]
            doc = [word.lower() for word in doc if word.isalpha()]
            lemmatizer = WordNetLemmatizer()
            doc = [lemmatizer.lemmatize(word) for word in doc]
            doc = (doc, label)
            cleaned_data.append(doc)

        self.training_data = cleaned_data

    # Produces vocabulary set (no duplicates)
    def create_known_vocabulary(self):

        vocab = []
        for (doc, label) in self.training_data:
            vocab += list(doc)
            self.known_vocabulary = set(vocab)

    # Calculate prior probabilities
    def create_priors(self):

        priors = {}

        for (doc, label) in self.training_data:
            priors[label] = priors.get(label, 0) + 1
        total = sum(priors.values())
        for key, value in priors.items():
            priors[key] = value / total
        self.priors = priors

    # Calculates conditional probabilities
    def create_cond_probs(self):
        conditional = {}
        for (doc, label) in self.training_data:
            doc = list(doc)
            class_conditional = conditional.get(label, {})
            for item in doc:
                class_conditional[item] = class_conditional.get(item, 0) + 1
            conditional[label] = class_conditional

        for label, c_conditional in conditional.items():
            for item in self.known_vocabulary:
                c_conditional[item] = c_conditional.get(item, 0) + 1
            conditional[label] = c_conditional

        for label, d in conditional.items():
            total = sum(d.values())
            conditional[label] = {key: value / total for (key, value) in d.items()}

        self.cond_probs = conditional

    # Trains the naive bayes classifier
    def train_classifier(self):
        self.create_known_vocabulary()
        self.create_priors()
        self.create_cond_probs()

    # Classifies an unseen document
    def classify(self, doc):
        d_probs = {key: math.log(value) for (key, value) in self.priors.items()}
        for word in doc:
            if word in self.known_vocabulary:
                d_probs = {classlabel: sofar + math.log(self.cond_probs[classlabel].get(word, 0)) for
                           (classlabel, sofar) in d_probs.items()}
        h_prob = max(d_probs.values())
        classes = [c for c in d_probs.keys() if d_probs[c] == h_prob]
        return random.choice(classes)


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from collections import Counter
from sklearn.metrics import precision_recall_curve


# Compare word list classifier and NBClassifer
class Compare:

    def __init__(self, training_data, testing_data):

        self.word_list_classifier = None
        self.naive_bayes_classifier = None
        self.training_data = training_data
        self.testing_data = testing_data
        self.setup_classifiers()  # Intialsie the classifiers

    # Build and train the classifiers
    def setup_classifiers(self):

        # Word list classifier
        positive_words, negative_words = most_frequent_words(fdist_pos, fdist_neg, 10)
        self.word_list_classifier = WordListClassifier(positive_words, negative_words)

        # Naive bayes classifier
        self.naive_bayes_classifier = NBClassifier(self.training_data)
        self.naive_bayes_classifier.clean_training_data()
        self.naive_bayes_classifier.train_classifier()

    # Generate data for receiving operating characteristic graph from an evaluation module
    def generateROCData(self, evaluate):

        # Compute fpr, tpr, thresholds and roc auc
        goldstandards = []
        predictions = []
        for x in evaluate.goldstandards:
            if x == "pos":
                goldstandards.append(1)
            else:
                goldstandards.append(0)

        for x in evaluate.predictions:
            if x == "pos":
                predictions.append(1)
            else:
                predictions.append(0)
        fpr, tpr, thresholds = roc_curve(goldstandards, predictions)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, thresholds, roc_auc

    # Plots a receiving operating characteristic graph
    def plotROC(self, evaluateW, evaluateNB):

        # Plot ROC curve
        fpr, tpr, thresholds, roc_auc = self.generateROCData(evaluateW)
        plt.plot(fpr, tpr, label='Word List ROC curve (area = %0.3f)' % roc_auc)

        fpr, tpr, thresholds, roc_auc = self.generateROCData(evaluateNB)
        plt.plot(fpr, tpr, label='Naive Bayes ROC curve  (area = %0.3f)' % roc_auc)

        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    # Creates predictions and calculates evaluation metrics for each classifier
    def get_data(self):

        # Word list classifier
        evaluate = Evaluate(self.testing_data, self.word_list_classifier)
        evaluate.create_predictions()
        evaluate.confusion_matrix()
        word_list_accuracy = evaluate.accuracy()
        precision = evaluate.precision()
        recall = evaluate.recall()

        # Naive Bayes

        evaluateNB = Evaluate(testing_data, self.naive_bayes_classifier)
        evaluateNB.create_predictions()
        evaluateNB.confusion_matrix()
        naive_bayes_accuracy = evaluateNB.accuracy()
        precision = evaluateNB.precision()
        recall = evaluateNB.recall()

        # Plot data

        self.plotROC(evaluate, evaluateNB)

        # self.plotPR(evaluate,evaluateNB)

        self.plot_accuracy(word_list_accuracy, naive_bayes_accuracy)

        self.plotErrorRate(word_list_accuracy, naive_bayes_accuracy)

    # Plots an accuracy graph
    def plot_accuracy(self, word_list_accuracy, naive_bayes_accuracy):
        df = pd.DataFrame({'Classifier': [
            'Wordlist Classifier', 'Naive Bayes Classifier', ], 'Accuracy': [word_list_accuracy, naive_bayes_accuracy]})
        ax = df.plot.bar(x='Classifier', y='Accuracy', rot=0)
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Accuracy")

    # Plots error rate graph
    def plotErrorRate(self, word_list_accuracy, naive_bayes_accuracy):
        df = pd.DataFrame({'Classifier': ['Wordlist Classifier', 'Naive Bayes Classifier', ],
                           'Error Rate': [1 - word_list_accuracy, 1 - naive_bayes_accuracy]})
        ax = df.plot.bar(x='Classifier', y='Error Rate', rot=0)
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Error Rate")


classifier_compare = Compare(training_data, testing_data)
classifier_compare.get_data()

from sklearn.model_selection import KFold
import numpy as np


# This method generates training and validation datasets ensuring class balance between the two.
def generateTrainValidation(training):
    positive_reviews = [(doc, label) for doc, label in training if label == 'pos']
    negative_reviews = [(doc, label) for doc, label in training if label == 'neg']

    positive_train, positive_validation = positive_reviews[(int(len(positive_reviews) * 0.1)):], positive_reviews[:int(
        len(positive_reviews) * 0.1)]
    negative_train, negative_validation = negative_reviews[(int(len(negative_reviews) * 0.1)):], negative_reviews[:int(
        len(negative_reviews) * 0.1):]

    training_data = positive_train + negative_train
    validation_data = positive_validation + negative_validation

    return training_data, validation_data


# Confirms that each dataset has a balanced class distribution
def confirmClassBalance(testing_data, validation_data, training_data):
    testing_count = {}
    for (doc, label) in testing_data:
        if label in testing_count:
            testing_count[label] += 1
        else:
            testing_count[label] = 1

    validation_count = {}
    for (doc, label) in validation_data:
        if label in validation_count:
            validation_count[label] += 1
        else:
            validation_count[label] = 1

    training_count = {}
    for (doc, label) in training_data:
        if label in training_count:
            training_count[label] += 1
        else:
            training_count[label] = 1

    print('Testing data class balance = ' + str(testing_count))
    print('Validation data class balance = ' + str(validation_count))
    print('Training data class balance = ' + str(training_count))


# Plots evaluation metrics for model tested on validation dataset
def plotValidation(accuracy, precision, recall, f1_score):
    # ------------------- #
    # Accuracy
    accuracy = dict(accuracy)
    accuracy = sorted(accuracy.items(), key=lambda x: x[0])

    x = [int(value[1] * 100) for value in accuracy]
    y = [int(value[0]) for value in accuracy]

    plt.plot(y, x, color='r', label='Validation Accuracy')
    plt.xlabel('Word list Length', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    plt.show()

    # ------------------- #

    plt.plot(y, x, color='y', label='Validation Precision')
    plt.xlabel('Word list Length', fontsize=18)
    plt.ylabel('Precision', fontsize=16)
    plt.show()

    # ------------------- #
    # Recall

    recall = dict(recall)
    recall = sorted(recall.items(), key=lambda x: x[0])

    x = [int(value[1] * 100) for value in recall]
    y = [int(value[0]) for value in recall]

    plt.plot(y, x, color='g', label='Validation Recall')
    plt.xlabel('Word list Length', fontsize=18)
    plt.ylabel('Recall', fontsize=16)
    plt.show()

    # ------------------- #
    # F1 Score

    f1_score = dict(f1_score)
    f1_score = sorted(f1_score.items(), key=lambda x: x[0])

    x = [int(value[1] * 100) for value in f1_score]
    y = [int(value[0]) for value in f1_score]

    plt.plot(y, x, color='b', label='Validation F1 Score')
    plt.xlabel('Word list Length', fontsize=18)
    plt.ylabel('F1 Score', fontsize=16)
    plt.show()


# Hyperparameter investigation experiment (Testing word list length)
def experiment(testing_data, validation_data):
    freq_dist_pos, freq_dist_neg = generate_frequency_distributions(training_data)

    accuracy = {}
    precision = {}
    recall = {}
    f1_score = {}

    for k in range(10, 1510, 10):
        positive_words, negative_words = most_frequent_words(freq_dist_pos, freq_dist_neg, k)
        word_list_classifier = WordListClassifier(positive_words, negative_words)
        evaluate = Evaluate(validation_data, word_list_classifier)
        evaluate.create_predictions()
        evaluate.confusion_matrix()
        word_list_accuracy = evaluate.accuracy()
        word_list_precision = evaluate.precision()
        word_list_recall = evaluate.recall()
        word_list_f1_score = evaluate.f1_score(word_list_precision, word_list_recall)

        accuracy[k] = word_list_accuracy
        precision[k] = word_list_precision
        recall[k] = word_list_recall
        f1_score[k] = word_list_f1_score

    accuracy = sorted(accuracy.items(), key=lambda x: x[1], reverse=True)

    plotValidation(accuracy, precision, recall, f1_score)

    testBestModel(accuracy, testing_data)


# Selects the best performing model from the experiment and tests it on the testing dataset
def testBestModel(accuracy, testing_data):
    print('Testing best hyperparameter value on testing dataset')

    # Selects the best K value from the results

    k = int(next(iter(accuracy))[0])

    # Builds a word list classifier with this hyperparameter value

    freq_dist_pos, freq_dist_neg = generate_frequency_distributions(training_data)
    positive_words, negative_words = most_frequent_words(freq_dist_pos, freq_dist_neg, k)
    word_list_classifier = WordListClassifier(positive_words, negative_words)

    # Tests it on the the testing data and evaluates it

    evaluate = Evaluate(validation_data, word_list_classifier)
    evaluate.create_predictions()
    evaluate.confusion_matrix()
    word_list_accuracy = evaluate.accuracy()
    print('Test Accuracy = ' + str(word_list_accuracy))


training_data, testing_data = get_train_test_data()
training_data, validation_data = generateTrainValidation(training_data)
confirmClassBalance(testing_data, validation_data, training_data)
experiment(testing_data, validation_data)




