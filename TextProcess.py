import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams
from fpdf import FPDF

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


def get__num_of_unique_words(sent_lst):
    freq_dict = get_word_freq_dict(sent_lst)
    return len(freq_dict.keys())


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


def calculate_sentences_prbo(sentences, text, isPadding):
    sent_lst = []
    if isPadding:
        sent_lst = sentences_via_padding(sentences_tokenize(text))
        sentences = list(pad_both_ends(nltk.word_tokenize(sentences.lower()), n=2))
    else:
        sent_lst = sentences_tokenize(text)
    freq_dict = get_word_freq_dict(sent_lst)
    for less_freq_word in list(freq_dict)[:3]:
        if sentences.__contains__(less_freq_word):
            index = sentences.index(less_freq_word)
            sentences[index] = "unk"

    unk_sent_lst = replace_unk_less_freq_words(sent_lst)
    bigram_dict = create_bigram_freq_dict(get_bigrams_list(unk_sent_lst))
    freq_dict_unk = get_word_freq_dict(unk_sent_lst)
    bigram_prob_dict = calculate_bigram_prob_dict(bigram_dict, freq_dict_unk)

    sent_bigram_lst = list(bigrams(sentences))
    prob = 1
    for bi in sent_bigram_lst:
        if bigram_prob_dict.keys().__contains__(bi):
            prob = prob * bigram_prob_dict[bi]
        else:
            prob = 0
    return prob

def print_bigrams(bigram_lst):
    for bi in bigram_lst:
        print(bi)


class TextProcessing(object):
    read_text = ""

    def __init__(self, filepath, filename):
        self.path = filepath
        self.file_name = filename
        self.read_text = read_txt_file(self.path)
        self.read_text = remove_break_lines(self.read_text)

    # Print methods
    def print_sentences(self):
        for sent in sentences_tokenize(self.read_text):
            print(sent)

    def print_num_of_sent(self):
        return len(sentences_tokenize(self.read_text))

    def print_num_of_total_token(self, is_padding):
        num_of_token = 0
        if is_padding:
            num_of_token = get_num_of_total_token(sentences_via_padding(sentences_tokenize(self.read_text)))
        else:
            num_of_token = get_num_of_total_token(sentences_tokenize(self.read_text))
        return num_of_token

    def print_num_of_unique_words(self, is_padding):
        num_of_token = 0
        if is_padding:
            num_of_token = get__num_of_unique_words(sentences_via_padding(sentences_tokenize(self.read_text)))
        else:
            num_of_token = get__num_of_unique_words(sentences_tokenize(self.read_text))
        return num_of_token


    def print_word_freq_dict(self):
        freq_dict = get_word_freq_dict(sentences_via_padding(sentences_tokenize(self.read_text)))
        for word in freq_dict.keys():
            print("(\"" + str(word) + "\": " + str(freq_dict[word]) + ")" + ",", end=" ")

    def print_n_unigram_via_prob(self, n, is_padding):
        sent_lst = []
        if is_padding:
            sent_lst = sentences_via_padding(sentences_tokenize(self.read_text))
        else:
            sent_lst = sentences_tokenize(self.read_text)
        freq_dict = get_word_freq_dict(sent_lst)
        num_of_token = get_num_of_total_token(sent_lst)
        unigram_prob_dict = calculete_unigram_prob_dict(freq_dict, num_of_token)
        return unigram_prob_dict
        # for b in list(unigram_prob_dict.keys().__reversed__())[:n]:
        #     print("(\"" + str(b) + "\")      " + str(freq_dict[b]) + "        " + str(unigram_prob_dict[b]), end=" ")
        #     print("\n")

    def print_n_bigrams_via_prob(self, n, is_padding):
        sent_lst = []
        if is_padding:
            sent_lst = sentences_via_padding(sentences_tokenize(self.read_text))
        else:
            sent_lst = sentences_tokenize(self.read_text)
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(sent_lst))
        freq_dict = get_word_freq_dict(sent_lst)
        bigram_prob_dict = calculate_bigram_prob_dict(bigram_dict, freq_dict)
        return bigram_prob_dict
        # for b in list(bigram_dict.keys().__reversed__())[:10]:
        #     print("(" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(bigram_prob_dict_smooting[b]), end=" ")
        #     print("\n")

    def print_n_bigrams_via_prob_unk(self, n):
        unk_sent_lst = replace_unk_less_freq_words(sentences_via_padding(sentences_tokenize(self.read_text)))
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(unk_sent_lst))
        freq_dict = get_word_freq_dict(unk_sent_lst)
        bigram_prob_dict_smooting = calculate_bigram_smooting_prob_dict(bigram_dict, freq_dict)

        for b in list(bigram_prob_dict_smooting.keys().__reversed__())[:10]:
            print("(" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(bigram_prob_dict_smooting[b]), end=" ")
            print("\n")

    def show_results(self):
        print("File name: " + self.file_name)

        print("Number of Sentences in Test File:" + str(self.print_num_of_sent()))
        print("Number of Total Tokens: " + str(self.print_num_of_total_token(True)))
        self.print_word_freq_dict()
        print("\nTop 10 Unigrams with Highest Frequencies:")
        self.print_n_unigram_via_prob(10)
        print("\nTop 10 Bigrams with Highest Frequencies:")
        self.print_n_bigrams_via_prob(10)
        print("\nAfter UNK addition and Smoothing Operations:")
        print("\nTop 10 Bigrams with Highest Frequencies:")
        self.print_n_bigrams_via_prob_unk(10)
        while True:
            print("Enter the sentences(hint: if press 'E' means exit, 'P' means print): ", end=" ")
            string = str(input())
            if string == "E":
                break;
            else:
                # output
                prob = calculate_sentences_prbo(string, self.read_text, True)
                print(string + ": " + str(prob))

    def write_pdf_file(self, is_padding):
        sent_lst = []
        if is_padding:
            sent_lst = sentences_via_padding(sentences_tokenize(self.read_text))
        else:
            sent_lst = sentences_tokenize(self.read_text)

        fn = str(self.file_name).split(".")
        pdf=FPDF('P', 'mm', 'A4')
        pdf.add_page()
        pdf.set_font("Arial", size=15)
        pdf.cell(200, 10, txt=fn[0], ln=1, align='C')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=10)
        for x in sentences_tokenize(self.read_text):
            pdf.cell(0, 5, txt=x, border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        pdf.cell(0, 5, txt="  - Number of Sentences in Test File: " + str(self.print_num_of_sent()), border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - Number of Total Tokens: " + str(self.print_num_of_total_token(is_padding)), border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - Number of Unique Words (Vocabulary Size): " + str(self.print_num_of_unique_words(is_padding)), border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - Top 10 Unigrams with Highest Frequencies: ", border=0, ln=2, align='L')
        freq_dict = get_word_freq_dict(sent_lst)
        unigram_prob_dict = self.print_n_unigram_via_prob(10, is_padding)
        for u in list(unigram_prob_dict.keys().__reversed__())[:10]:
            pdf.cell(0, 5, txt="        (\"" + str(u) + "\")      " + str(freq_dict[u]) + "        " + str(unigram_prob_dict[u]), border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - Top 10 Bigrams with Highest Frequencies: ", border=0, ln=2, align='L')
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(sent_lst))
        bigram_prob_dict = calculate_bigram_prob_dict(bigram_dict, freq_dict)
        for b in list(bigram_prob_dict.keys().__reversed__())[:10]:
            pdf.cell(0, 5, txt="        (" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(bigram_prob_dict[b]), border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - After UNK addition and Smoothing Operations: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="    - Top 10 Bigrams with Highest Frequencies: ", border=0, ln=2, align='L')
        bigram_prob_dict_smooting = calculate_bigram_smooting_prob_dict(bigram_dict, freq_dict)
        for b in list(bigram_prob_dict_smooting.keys().__reversed__())[:10]:
            pdf.cell(0, 5, txt="        (" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(bigram_prob_dict_smooting[b]), border=0, ln=2, align='L')
        unk_sent_lst = replace_unk_less_freq_words(sent_lst)
        bigram_dict_unk = create_bigram_freq_dict(get_bigrams_list(unk_sent_lst))
        freq_dict_unk = get_word_freq_dict(unk_sent_lst)
        bigram_prob_dict_smooting_unk = calculate_bigram_smooting_prob_dict(bigram_dict_unk, freq_dict_unk)
        pdf.cell(0, 5, txt="    - Top 10 Bigrams with UNK: ", border=0, ln=2, align='L')
        for b in list(bigram_prob_dict_smooting_unk.keys().__reversed__())[:10]:
            pdf.cell(0, 5, txt="        (" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(bigram_prob_dict_smooting_unk[b]), border=0, ln=2, align='L')
        while True:
            print("Enter the sentences(hint: if press 'E' means exit): ", end=" ")
            string = str(input())
            if string == "E":
                break;
            else:
                prob = calculate_sentences_prbo(string, self.read_text, True)
                pdf.cell(0, 5, txt="    -" + string + ": " + str(prob), border=0, ln=2, align='L')
        pdf.output(fn[0]+".pdf")