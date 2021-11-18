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
            if sent_lst.__contains__(w):
                word_freq_dict[w] = word_freq_dict[w] + 1
            else:
                word_freq_dict[w] = 1

    word_freq_dict = {k: v for k, v in sorted(word_freq_dict.items(), key=lambda item: item[1])}
    return word_freq_dict


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

    def show_results(self):
        print("File name: " + self.file_name)
        self.read_text = read_txt_file(self.path)
        self.read_text = remove_break_lines(self.read_text)
        print("Number of Sentences in Test File:" + str(self.print_num_of_sent()))
        print("Number of Total Tokens: " + str(self.print_num_of_total_token()))
        self.print_word_freq_dict()


