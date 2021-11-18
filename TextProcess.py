import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams


# https://medium.com/swlh/language-modelling-with-nltk-20eac7e70853
# https://becominghuman.ai/nlp-for-beginners-using-nltk-f58ec22005cd

def remove_break_lines(text):
    return text.lower().replace('\r', '').replace('\n', '').replace('       ', ' ').replace('      ', '')


def sentences_tokenize(text):
    return nltk.sent_tokenize(text)


def sentences_via_padding(sent_lst):
    sentences_list_padding = []
    for sentence in sent_lst:
        sentences_list_padding.append(list(pad_both_ends(nltk.word_tokenize(sentence), n=2)))
    return sentences_list_padding


def print_sentences(sent_lst):
    for sent in sent_lst:
        print(sent)


def read_txt_file(path):
    with open(path) as f:
        return f.read()


def get_num_of_total_token(sent_lst):
    num_of_token = 0
    for s in sent_lst:
        num_of_token += len(s)
    return num_of_token


def get_word_freq_dict(sent_lst):
    word_freq_dict = {}
    for s in sent_lst:
        for w in s:
            if word_freq_dict.keys().__contains__(w):
                word_freq_dict[w] = word_freq_dict[w] + 1
            else:
                word_freq_dict[w] = 1

    word_freq_dict = {k: v for k, v in sorted(word_freq_dict.items(), key=lambda item: item[1])}
    return word_freq_dict


def calculete_unigram_prob_dict(freq_dict, num_of_token):
    unigram_prob_dict = {}
    for w in freq_dict.keys():
        prob = freq_dict[w] / num_of_token
        unigram_prob_dict[w] = prob
    return unigram_prob_dict


def get_bigrams_list(sent_lst):
    bigrams_lst = []
    for padding_sent in sent_lst:
        bigrams_lst.append(list(bigrams(padding_sent)))
    return bigrams_lst


def create_bigram_freq_dict(bigram_lst):
    bigram_freq_dict = {}
    for b_list in bigram_lst:
        for b in b_list:
            if bigram_freq_dict.keys().__contains__(b):
                bigram_freq_dict[b] = bigram_freq_dict[b] + 1
            else:
                bigram_freq_dict[b] = 1
    bigram_freq_dict = {k: v for k, v in sorted(bigram_freq_dict.items(), key=lambda item: item[1])}
    return bigram_freq_dict


def calculate_bigram_prob_dict(bigram_freq_dict, word_freq_dict):
    bigram_prob_dict = {}
    for b in bigram_freq_dict.keys():
        prob = bigram_freq_dict[b] / word_freq_dict[b[0]]
        bigram_prob_dict[b] = prob
    return bigram_prob_dict


def calculate_bigram_smooting_prob_dict(bigram_freq_dict, word_freq_dict):
    bigram_prob_dict = {}
    for b in bigram_freq_dict.keys():
        prob = (bigram_freq_dict[b] + 0.5) / (word_freq_dict[b[0]] + len(word_freq_dict))
        bigram_prob_dict[b] = prob
    return bigram_prob_dict


def replace_unk_less_freq_words(sent_lst):
    freq_dict = get_word_freq_dict(sent_lst)
    for sent in sent_lst:
        for less_freq_word in list(freq_dict)[:3]:
            if sent.__contains__(less_freq_word):
                index = sent.index(less_freq_word)
                sent[index] = "unk"
    return sent_lst


def print_bigrams(bigram_lst):
    for bi in bigram_lst:
        print(bi)


class TextProcessing(object):
    read_text = ""

    def __init__(self, filepath, filename):
        self.path = filepath
        self.file_name = filename

    # Print methods
    def print_sentences(self):
        for sent in sentences_tokenize(self.read_text):
            print(sent)

    def print_num_of_sent(self):
        return len(sentences_tokenize(self.read_text))

    def print_num_of_total_token(self):
        return get_num_of_total_token(sentences_via_padding(sentences_tokenize(self.read_text)))

    def print_word_freq_dict(self):
        freq_dict = get_word_freq_dict(sentences_via_padding(sentences_tokenize(self.read_text)))
        for word in freq_dict.keys():
            print("(\"" + str(word) + "\": " + str(freq_dict[word]) + ")" + ",", end=" ")

    def print_n_unigram_via_prob(self, n):
        freq_dict = get_word_freq_dict(sentences_via_padding(sentences_tokenize(self.read_text)))
        num_of_token = get_num_of_total_token(sentences_via_padding(sentences_tokenize(self.read_text)))
        unigram_prob_dict = calculete_unigram_prob_dict(freq_dict, num_of_token)

        for b in list(unigram_prob_dict.keys().__reversed__())[:n]:
            print("(\"" + str(b) + "\")      " + str(freq_dict[b]) + "        " + str(unigram_prob_dict[b]), end=" ")
            print("\n")

    def print_n_bigrams_via_prob(self, n):
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(sentences_via_padding(sentences_tokenize(self.read_text))))
        freq_dict = get_word_freq_dict(sentences_via_padding(sentences_tokenize(self.read_text)))
        bigram_prob_dict_smooting = calculate_bigram_smooting_prob_dict(bigram_dict, freq_dict)

        for b in list(bigram_dict.keys().__reversed__())[:10]:
            print("(" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(bigram_prob_dict_smooting[b]), end=" ")
            print("\n")

    def print_n_bigrams_via_prob_unk(self, n):
        unk_sent_lst = replace_unk_less_freq_words(sentences_via_padding(sentences_tokenize(self.read_text)))
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(unk_sent_lst))
        freq_dict = get_word_freq_dict(unk_sent_lst)
        bigram_prob_dict = calculate_bigram_prob_dict(bigram_dict, freq_dict)

        for b in list(bigram_dict.keys().__reversed__())[:10]:
            print("(" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(bigram_prob_dict[b]), end=" ")
            print("\n")

    def show_results(self):
        print("File name: " + self.file_name)
        self.read_text = read_txt_file(self.path)
        self.read_text = remove_break_lines(self.read_text)
        print("Number of Sentences in Test File:" + str(self.print_num_of_sent()))
        print("Number of Total Tokens: " + str(self.print_num_of_total_token()))
        self.print_word_freq_dict()
        print("\nTop 10 Unigrams with Highest Frequencies:")
        self.print_n_unigram_via_prob(10)
        print("\nTop 10 Bigrams with Highest Frequencies:")
        self.print_n_bigrams_via_prob(10)
        print("\nAfter UNK addition and Smoothing Operations:")
        print("\nTop 10 Bigrams with Highest Frequencies:")
        self.print_n_bigrams_via_prob_unk(10)

