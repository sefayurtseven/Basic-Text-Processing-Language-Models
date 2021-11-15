import nltk
# https://medium.com/swlh/language-modelling-with-nltk-20eac7e70853
# https://becominghuman.ai/nlp-for-beginners-using-nltk-f58ec22005cd

class FileReader(object):
    text = ""
    file_name = ""
    sentences_list = []
    word_list = []
    freq_list = []
    bigrams = []

    def __init__(self, filepath, filename):
        self.path = filepath
        self.file_name = filename
        self.read_txt_file()
        self.sent_tokenize()
        self.word_tokenize()
        self.get_word_frequency()
        self.get_bigrams_list()

    def read_txt_file(self):
        with open(self.path) as f:
            contents = f.read()
            self.text = contents.lower().replace('\r', '').replace('\n', '').replace('       ', ' ').replace('      ',
                                                                                                             '')

    def sent_tokenize(self):
        self.sentences_list = nltk.sent_tokenize(self.text)

    def word_tokenize(self):
        self.word_list = nltk.word_tokenize(self.text)

    def print_sentences(self):
        for sentences in self.sentences_list:
            print(sentences)

    def get_word_frequency(self):
        self.freq_list = nltk.FreqDist(self.word_list)

    def get_bigrams_list(self):
        self.bigrams = nltk.ngrams(self.word_list, 2)

    def get_sentences_num(self):
        return len(self.sentences_list)

    def get_token_num(self):
        return len(self.word_list)

    def show_results(self):
        print("Number of Sentences in " + self.file_name + ": " + str(self.get_sentences_num()))
        print("Number of Total Tokens: " + str(self.get_token_num()))
        print("Number of Unique Words (Vocabulary Size): ")
        for word in self.freq_list.keys():
            print(str(word) + ": " + str(self.freq_list[word]) + "|", end=" ")
