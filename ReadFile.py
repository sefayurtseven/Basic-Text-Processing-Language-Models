import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams

# https://medium.com/swlh/language-modelling-with-nltk-20eac7e70853
# https://becominghuman.ai/nlp-for-beginners-using-nltk-f58ec22005cd

class FileReader(object):
    text = ""
    file_name = ""
    sentences_list = []
    sentences_list_padding = []
    word_list = []
    freq_dict = {}
    bigrams_lst = []
    bigram_dict = {}
    unigram_prob_dict = {}
    bigram_prob_dict = {}
    num_of_token = 0

    def __init__(self, filepath, filename):
        self.path = filepath
        self.file_name = filename
        self.read_txt_file()
        self.sent_tokenize()
        self.sentences_via_padding()
        self.word_tokenize()
        self.get_word_frequency()
        self.get_bigrams_list()
        self.create_bigram_dict()
        self.create_bigram_prob_dict()
        self.create_unigram_prob_dict()


    def read_txt_file(self):
        with open(self.path) as f:
            contents = f.read()
            self.text = contents.lower().replace('\r', '').replace('\n', '').replace('       ', ' ').replace('      ','')

    def sent_tokenize(self):
        self.sentences_list = nltk.sent_tokenize(self.text)

    def word_tokenize(self):
        for sentences in self.sentences_list_padding:
            self.word_list.append(sentences)

    def print_sentences(self):
        print("--------- * Sentences * ---------")
        for sentences in self.sentences_list:
            print(sentences)

    def print_sentences_via_padding(self):
        print("--------- * Sentences Via Padding * ---------")
        for sentences in self.sentences_list_padding:
            print(sentences)

    def print_bigram_list(self):
        for b in self.bigrams_lst:
            print(b)

    def sentences_via_padding(self):
        for sentence in self.sentences_list:
            self.sentences_list_padding.append(list(pad_both_ends(nltk.word_tokenize(sentence), n=2)))

    def get_word_frequency(self):
        for w_list in self.word_list:
            self.num_of_token += len(w_list)
            for w in w_list:
                if self.freq_dict.__contains__(w):
                    self.freq_dict[w] = self.freq_dict[w] + 1
                else:
                    self.freq_dict[w] = 1
        self.freq_dict = {k: v for k, v in sorted(self.freq_dict.items(), key=lambda item: item[1])}

    def get_bigrams_list(self):
        for padding_sent in self.sentences_list_padding:
            self.bigrams_lst.append(list(bigrams(padding_sent)))

    def get_sentences_num(self):
        return len(self.sentences_list)

    def get_token_num(self):
        return self.num_of_token

    def create_bigram_dict(self):
        for b_list in self.bigrams_lst:
            for b in b_list:
                if self.bigram_dict.__contains__(b):
                    self.bigram_dict[b] = self.bigram_dict[b] + 1
                else:
                    self.bigram_dict[b] = 1
        self.bigram_dict = {k: v for k, v in sorted(self.bigram_dict.items(), key=lambda item: item[1])}

    def create_bigram_prob_dict(self):
        for b in self.bigram_dict.keys():
            prob = self.bigram_dict[b] / self.freq_dict[b[0]]
            self.bigram_prob_dict[b] = prob

    def create_unigram_prob_dict(self):
        for w in self.freq_dict.keys():
            prob = self.freq_dict[w] / self.num_of_token
            self.unigram_prob_dict[w] = prob


    def show_results(self):
        self.print_sentences_via_padding()
        print("Number of Sentences in " + self.file_name + ": " + str(self.get_sentences_num()))
        print("Number of Total Tokens: " + str(self.get_token_num()))
        print("Number of Unique Words (Vocabulary Size): ")
        for word in self.freq_dict.keys():
            print("(\"" + str(word) + "\": " + str(self.freq_dict[word]) + ")" + ",", end=" ")
        print("\nTop 10 Unigrams with Highest Frequencies:")
        for b in list(self.unigram_prob_dict.keys().__reversed__())[:10]:
            print("(\"" + str(b) + "\")      " + str(self.freq_dict[b]) + "        " + str(self.unigram_prob_dict[b]), end=" ")
            print("\n")
        print("\nTop 10 Unigrams with Highest Frequencies:")
        for b in list(self.bigram_dict.keys().__reversed__())[:10]:
            print("(" + str(b[0]) + "," + str(b[1]) + ")      " + str(self.bigram_dict[b]) + "        " + str(self.bigram_prob_dict[b]), end=" ")
            print("\n")

