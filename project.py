from  flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

dict_pos = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
def pos(token):
    newlist = []
    temp = pos_tag(token)
    for word, tag in temp:
        newlist.append(tuple([word, dict_pos.get(tag[0])]))
    return newlist

lemmatization = WordNetLemmatizer()
def find_lemma(pair_list):
    string = ""
    for word, tag in pair_list:
        if not tag:
            string = string + " " + word
        else:
            lemma = lemmatization.lemmatize(word, pos = tag)
            string = string + " " + lemma
    return string


app = Flask(__name__)
@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/button_yes', methods = ['GET'])
def btn_yes():
    return render_template('form.html')

@app.route('/button_no', methods = ['GET'])
def btn_no():
    return render_template('sorry.html')

@app.route('/read-more', methods = ['GET'])
def read_more():
    return render_template('read-more.html')

@app.route('/predict', methods = ['POST'])
def run_model():
    input_text = [ str(x) for x in request.form.values()]
    i = input_text[0]

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(i)
    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    pos_list = pos(filtered_sentence)
    lemma_str = find_lemma(pos_list)
    vs = analyzer.polarity_scores(lemma_str)
    print(type(vs))
    neg_score = vs['neg']
    pos_score = vs['pos']
    neu_score = vs['neu']
    
    return render_template('form.html', prediction_text = "Positive Score = {} Negative Score = {} neutral = {}".format(pos_score, neg_score, neu_score))


if __name__ == '__main__':
    app.run(port = 3000, debug = True)
    

