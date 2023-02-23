from  flask import Flask, render_template, request
import tensorflow as tf
model = tf.keras.models.load_model( 'model/sentiment')

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
    max_num_words = 5000
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_num_words, filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    tokenizer.fit_on_texts(i)
    token_input = tokenizer.texts_to_sequences(i)
    token_input = tf.keras.preprocessing.sequence.pad_sequences(token_input, maxlen = max_num_words, padding = 'post')

    out  = model.predict(token_input)
    out_score = out[0]
    neg_score = round(out_score[1], 3)
    pos_score = round(out_score[2], 3)
    neu_score = round(out_score[3], 3)
    
    
    return render_template('form.html', prediction_text = "Positive Score = {} Negative Score = {} neutral = {}".format(pos_score, neg_score, neu_score))


if __name__ == '__main__':
    app.run(port = 3000, debug = True)
    

