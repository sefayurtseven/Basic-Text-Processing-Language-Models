import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams
from fpdf import FPDF


# read the txt file
def read_txt_file(path):
    with open(path) as f:
        return f.read()


# remove linebreaks and spaces in text
def remove_break_lines(text):
    return text.lower().replace('\r', '').replace('\n', '').replace('       ', ' ').replace('      ', '')


# get sentences in text
def sentences_tokenize(text):
    return text.lstrip().split(".")[:-1]


# apply padding on sentences
def sentences_via_padding(sent_lst):
    sentences_list_padding = []
    for sentence in sent_lst:
        sentences_list_padding.append(list(pad_both_ends(nltk.word_tokenize(sentence), n=2)))
    return sentences_list_padding


# get token frequency dictionary
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


# get total number of token
def get_num_of_total_token(sent_lst):
    num_of_token = 0
    for s in sent_lst:
        num_of_token += len(s)
    return num_of_token


# get number of unique token
def get_num_of_unique_words(sent_lst):
    freq_dict = get_word_freq_dict(sent_lst)
    return len(freq_dict.keys())


# get unigram probability dictionary
def get_unigram_prob_dict(freq_dict, num_of_token):
    unigram_prob_dict = {}
    for w in freq_dict.keys():
        prob = freq_dict[w] / num_of_token
        unigram_prob_dict[w] = prob
    return unigram_prob_dict


# get bigram list
def get_bigrams_list(sent_lst):
    bigrams_lst = []
    for padding_sent in sent_lst:
        bigrams_lst.append(list(bigrams(padding_sent)))
    return bigrams_lst


# get bigram frequency dictionary
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


# get bigram probability dictionary
def get_bigram_prob_dict(bigram_freq_dict, word_freq_dict):
    bigram_prob_dict = {}
    for b in bigram_freq_dict.keys():
        prob = bigram_freq_dict[b] / word_freq_dict[b[0]]
        bigram_prob_dict[b] = prob
    return bigram_prob_dict


# get bigram probability with smoothing
def get_bigram_smooting_prob_dict(bigram_freq_dict, word_freq_dict):
    bigram_prob_dict = {}
    for b in bigram_freq_dict.keys():
        prob = (bigram_freq_dict[b] + 0.5) / (word_freq_dict[b[0]] + len(word_freq_dict))
        bigram_prob_dict[b] = prob
    return bigram_prob_dict


# replace less frequency wort with unk
def replace_unk_less_freq_words(sent_lst):
    freq_dict = get_word_freq_dict(sent_lst)
    for sent in sent_lst:
        for less_freq_word in list(freq_dict)[:3]:
            if sent.__contains__(less_freq_word):
                index = sent.index(less_freq_word)
                sent[index] = "unk"
    return sent_lst


# get sentence bigram probability
def get_sentences_prob(sentences, text, is_padding):
    sent_lst = []
    if is_padding:
        sent_lst = sentences_via_padding(sentences_tokenize(text))
        sentences = list(pad_both_ends(nltk.word_tokenize(sentences.lower()), n=2))
    else:
        sent_lst = sentences_tokenize(text)
    freq_dict = get_word_freq_dict(sent_lst)
    for less_freq_word in list(freq_dict)[:3]:
        if sentences.__contains__(less_freq_word):
            index = sentences.index(less_freq_word)
            sentences[index] = "unk"
    bigram_freq_dict = {}
    word_freq_dict = {}
    bigrams_lst = list(bigrams(sentences))
    for b in bigrams_lst:
        if bigram_freq_dict.keys().__contains__(b):
            bigram_freq_dict[b] = bigram_freq_dict[b] + 1
        else:
            bigram_freq_dict[b] = 1
    for w in sentences:
        if word_freq_dict.keys().__contains__(w):
            word_freq_dict[w] = word_freq_dict[w] + 1
        else:
            word_freq_dict[w] = 1

    bigram_prob_dict = get_bigram_smooting_prob_dict(bigram_freq_dict, word_freq_dict)

    prob = 1
    for bi in bigrams_lst:
        if bigram_prob_dict.keys().__contains__(bi):
            prob = prob * bigram_prob_dict[bi]
        else:
            prob = 0
    return prob


class TextProcessing(object):
    read_text = ""
    processed_sentences_list = []

    def __init__(self, filepath, filename):
        self.path = filepath
        self.file_name = filename
        self.process_text()
        self.processed_sentences_list = self.apply_padding_on_sentences_lst()

    # Processing
    def process_text(self):
        self.read_text = read_txt_file(self.path)
        self.read_text = remove_break_lines(self.read_text)

    # Apply padding
    def apply_padding_on_sentences_lst(self, is_padding=True):
        if is_padding:
            return sentences_via_padding(sentences_tokenize(self.read_text))
        else:
            return sentences_tokenize(self.read_text)

    # pdf writer
    def write_pdf_file(self, is_padding):
        if not is_padding:
            self.processed_sentences_list = sentences_tokenize(self.read_text)

        fn = str(self.file_name).split(".")
        pdf = FPDF('P', 'mm', 'A4')
        pdf.add_page()
        pdf.set_font("Arial", size=15)
        pdf.cell(200, 10, txt=fn[0], ln=1, align='C')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')

        pdf.cell(0, 5, txt="Results: ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        pdf.cell(0, 5, txt="  - Number of Sentences in Test File: " + str(len(sentences_tokenize(self.read_text))), border=0, ln=2,
                 align='L')
        pdf.cell(0, 5, txt="  - Number of Total Tokens: " + str(get_num_of_total_token(self.processed_sentences_list)), border=0,
                 ln=2, align='L')
        pdf.cell(0, 5,
                 txt="  - Number of Unique Words (Vocabulary Size): " + str(get_num_of_unique_words(self.processed_sentences_list)),
                 border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - Top 10 Unigrams with Highest Frequencies: ", border=0, ln=2, align='L')
        freq_dict = get_word_freq_dict(self.processed_sentences_list)
        unigram_prob_dict = get_unigram_prob_dict(freq_dict, get_num_of_total_token(self.processed_sentences_list))
        for u in list(unigram_prob_dict.keys().__reversed__())[:10]:
            pdf.cell(0, 5, txt="        (\"" + str(u) + "\")      " + str(freq_dict[u]) + "        " + str(
                unigram_prob_dict[u]), border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - Top 10 Bigrams with Highest Frequencies: ", border=0, ln=2, align='L')
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(self.processed_sentences_list))
        bigram_prob_dict = get_bigram_prob_dict(bigram_dict, freq_dict)
        for b in list(bigram_prob_dict.keys().__reversed__())[:10]:
            pdf.cell(0, 5,
                     txt="        (" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(
                         bigram_prob_dict[b]), border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="  - After UNK addition and Smoothing Operations: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="    - Top 10 Bigrams with Highest Frequencies: ", border=0, ln=2, align='L')
        bigram_prob_dict_smooting = get_bigram_smooting_prob_dict(bigram_dict, freq_dict)
        for b in list(bigram_prob_dict_smooting.keys().__reversed__())[:10]:
            pdf.cell(0, 5,
                     txt="        (" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(
                         bigram_prob_dict_smooting[b]), border=0, ln=2, align='L')
        unk_sent_lst = replace_unk_less_freq_words(self.processed_sentences_list)
        bigram_dict_unk = create_bigram_freq_dict(get_bigrams_list(unk_sent_lst))
        freq_dict_unk = get_word_freq_dict(unk_sent_lst)
        bigram_prob_dict_smooting_unk = get_bigram_smooting_prob_dict(bigram_dict_unk, freq_dict_unk)
        pdf.cell(0, 5, txt="    - Top 10 Bigrams with UNK: ", border=0, ln=2, align='L')
        for b in list(bigram_prob_dict_smooting_unk.keys().__reversed__())[:10]:
            pdf.cell(0, 5,
                     txt="        (" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_prob_dict_smooting_unk[b]) + "        " + str(
                         bigram_prob_dict_smooting_unk[b]), border=0, ln=2, align='L')
        while True:
            print("Enter the sentences(hint: if press 'E' means exit): ", end=" ")
            string = str(input())
            if string == "E":
                break;
            else:
                prob = get_sentences_prob(string, self.read_text, True)
                pdf.cell(0, 5, txt="    -" + string + "    (ItsComputeProbability: " + str(prob) + ")", border=0, ln=2, align='L')
        pdf.cell(0, 20, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, txt="Sentences: ", border=0, ln=2, align='L')
        for x in sentences_tokenize(self.read_text):
            pdf.cell(0, 5, txt=" - " + x, border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.ln()
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, txt="Frequency List: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        line_counter = 0;
        for word in freq_dict.keys():
            if line_counter == 8:
                pdf.ln()
                pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
                line_counter = 0
            pdf.cell(0, 0, txt="(\"" + str(word) + "\": " + str(freq_dict[word]) + ")" + ",\n", border=0, ln=2, align='L')
            pdf.cell(23)
            line_counter += 1
        pdf.set_font("Arial", size=10)
        pdf.ln()
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="Bigram Frequency List: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        line_counter = 0;
        for b in bigram_dict.keys():
            if line_counter == 5:
                pdf.ln()
                pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
                line_counter = 0
            pdf.cell(0, 0,
                     txt="(" + str(b[1]) + "," + str(b[0]) + ") =" + str(bigram_dict[b]) , border=0, ln=2, align='L')
            pdf.cell(30)
            line_counter += 1
        pdf.set_font("Arial", size=10)
        pdf.ln()
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="Unigram Probability List: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        line_counter = 0;
        for u in unigram_prob_dict.keys():
            if line_counter == 5:
                pdf.ln()
                pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
                line_counter = 0
            pdf.cell(0, 0, txt="(\"" + str(u) + "\")= "+str(round(unigram_prob_dict[u], 5)), border=0, ln=2, align='L')
            pdf.cell(50)
            line_counter += 1

        pdf.set_font("Arial", size=10)
        pdf.ln()
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="Bigram Probability List: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        line_counter = 0;
        for b in bigram_prob_dict.keys():
            if line_counter == 5:
                pdf.ln()
                pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
                line_counter = 0
            pdf.cell(0, 0, txt="(" + str(b[1]) + "," + str(b[0]) + ")= " + str(round(bigram_prob_dict[b], 3)), border=0, ln=2, align='L')
            pdf.cell(50)
            line_counter += 1

        pdf.ln()
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="Sentences After Replace With UNK: ", border=0, ln=2, align='L')
        for word_lst in unk_sent_lst:
            sent = "";
            for w in word_lst:
                sent += w
                sent += " "
            pdf.cell(0, 5, txt=" - " + sent, border=0, ln=2, align='L')

        pdf.ln()
        pdf.cell(0, 20, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, txt="Frequency List After Replace with UNK: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        line_counter = 0;
        for word in freq_dict_unk.keys():
            if line_counter == 8:
                pdf.ln()
                pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
                line_counter = 0
            pdf.cell(0, 0, txt="(\"" + str(word) + "\": " + str(freq_dict_unk[word]) + ")" + ",\n", border=0, ln=2, align='L')
            pdf.cell(23)
            line_counter += 1
        pdf.set_font("Arial", size=10)
        pdf.ln()
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="Bigram Frequency List After Replace with UNK: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        line_counter = 0;
        for b in bigram_dict_unk.keys():
            if line_counter == 5:
                pdf.ln()
                pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
                line_counter = 0
            pdf.cell(0, 0,
                     txt="(" + str(b[1]) + "," + str(b[0]) + ") =" + str(bigram_dict_unk[b]) , border=0, ln=2, align='L')
            pdf.cell(30)
            line_counter += 1

        pdf.set_font("Arial", size=10)
        pdf.ln()
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt="Bigram Probability List After Replace with UNK and Smoothing: ", border=0, ln=2, align='L')
        pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
        pdf.set_font("Arial", size=9)
        line_counter = 0;
        for b in bigram_prob_dict_smooting_unk.keys():
            if line_counter == 5:
                pdf.ln()
                pdf.cell(0, 5, txt=" ", border=0, ln=2, align='L')
                line_counter = 0
            pdf.cell(0, 0, txt="(" + str(b[1]) + "," + str(b[0]) + ")= " + str(round(bigram_prob_dict_smooting_unk[b], 3)), border=0, ln=2, align='L')
            pdf.cell(50)
            line_counter += 1
        pdf.output(fn[0] + ".pdf")

    # Print methods
    def print_sentences(self):
        for sent in sentences_tokenize(self.read_text):
            print(sent)

    def print_num_of_sent(self):
        print(len(sentences_tokenize(self.read_text)))

    def print_num_of_total_token(self):
        print(get_num_of_total_token(self.processed_sentences_list))

    def print_num_of_unique_words(self):
        return print(get_num_of_unique_words(self.processed_sentences_list))

    def print_word_freq_dict(self):
        freq_dict = get_word_freq_dict(self.processed_sentences_list)
        for word in freq_dict.keys():
            print("(\"" + str(word) + "\": " + str(freq_dict[word]) + ")" + ",", end=" ")

    def print_n_unigram_via_prob(self, n):
        freq_dict = get_word_freq_dict(self.processed_sentences_list)
        num_of_token = get_num_of_total_token(self.processed_sentences_list)
        unigram_prob_dict = get_unigram_prob_dict(freq_dict, num_of_token)

        for b in list(unigram_prob_dict.keys().__reversed__())[:n]:
            print("(\"" + str(b) + "\")      " + str(freq_dict[b]) + "        " + str(unigram_prob_dict[b]), end=" ")
            print("\n")

    def print_n_bigrams_via_prob(self, n):
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(self.processed_sentences_list))
        freq_dict = get_word_freq_dict(self.processed_sentences_list)
        bigram_prob_dict_smooting = get_bigram_smooting_prob_dict(bigram_dict, freq_dict)

        for b in list(bigram_dict.keys().__reversed__())[:10]:
            print("(" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(
                bigram_prob_dict_smooting[b]), end=" ")
            print("\n")

    def print_n_bigrams_via_prob_unk(self, n):
        unk_sent_lst = replace_unk_less_freq_words(self.processed_sentences_list)
        bigram_dict = create_bigram_freq_dict(get_bigrams_list(unk_sent_lst))
        freq_dict = get_word_freq_dict(unk_sent_lst)
        bigram_prob_dict_smooting = get_bigram_smooting_prob_dict(bigram_dict, freq_dict)

        for b in list(bigram_prob_dict_smooting.keys().__reversed__())[:10]:
            print("(" + str(b[1]) + "," + str(b[0]) + ")      " + str(bigram_dict[b]) + "        " + str(
                bigram_prob_dict_smooting[b]), end=" ")
            print("\n")


