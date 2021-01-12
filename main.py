from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
def get_top_x_words(corpus, x=None):
    vec = CountVectorizer()
    #row = document in corpus
    #column = word in document
    word_matrix = vec.fit_transform(corpus)
    #determine the number of times a word occures in the matrix
    #by counting the number of elements in each column of the matrix
    sum_words = word_matrix.sum(axis=0)
    #convert vector into tuple
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    #list word frequnecies from greatest to least
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:x]

if __name__ == '__main__':
    #split document by line, where each line is an element in array corpus
    corpus = [line.replace("\n", "") for line in open("/Users/shellyschwartz/Downloads/iron_man.txt")]
    top_words = get_top_x_words(corpus)
    rank = []
    freqs = []
    x = 0
    #loop through list of words and their respective frequencies
    for word, freq in top_words:
       x = x + 1
       rank.append(x)
       freqs.append(freq)

    print(freqs)
    plt.plot(freqs, rank)
    #change to logorithmic scale to observe proportionality between word frequency and rank
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("word frequencies")
    plt.ylabel("word rank")
    plt.show()

