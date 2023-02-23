import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import matplotlib.pyplot as plt
import pickle

# Loading json data
with open('data_full.json') as file:
  data = json.loads(file.read())

# Loading out-of-scope intent data
val_oos = np.array(data['oos_val'])
train_oos = np.array(data['oos_train'])
test_oos = np.array(data['oos_test'])

# Loading other intents data
val_others = np.array(data['val'])
train_others = np.array(data['train'])
test_others = np.array(data['test'])

# Merging out-of-scope and other intent data
val = np.concatenate([val_oos,val_others])
train = np.concatenate([train_oos,train_others])
test = np.concatenate([test_oos,test_others])
data = np.concatenate([train,test,val])
data = data.T #transpose

text = data[0]
labels = data[1]

from sklearn.model_selection import train_test_split
train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)


# from tensorflow.keras.preprocessing.sequence import pad_sequences
max_num_words = 40000
classes = np.unique(labels) #Finds unique elements from the array labels

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(train_txt)
word_index = tokenizer.word_index

ls=[]
for c in train_txt:
    ls.append(len(c.split()))
maxLen=int(np.percentile(ls, 98))
train_sequences = tokenizer.texts_to_sequences(train_txt)
test_sequences = tokenizer.texts_to_sequences(test_txt)

test_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=maxLen, padding = 'post')
train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=maxLen, padding = 'post')

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(classes)

#Now it is converting the interget label_encoder to onehot_encoder (Basically encoding in 0 and 1 only)
#to do that it first need to transform, reshape the array before passing it through the onehotencoder
# First we did for the classes, then for the test and training dataset

onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit(integer_encoded)

# Now doing it for training data
train_label_encoded = label_encoder.transform(train_label)
train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
train_label = onehot_encoder.transform(train_label_encoded)

#now doing it for testing data
test_labels_encoded = label_encoder.transform(test_labels)
test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
test_labels = onehot_encoder.transform(test_labels_encoded)

embeddings_index={}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
num_words = min(max_num_words, len(word_index))+1
embedding_dim=len(embeddings_index['the'])
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_num_words:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, Embedding
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(classes.shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
history = model.fit(train_sequences, train_label, epochs = 1,
          batch_size = 1024, shuffle=True,
          validation_data=[test_sequences, test_labels])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


model.save('model/intent')
with open('model/classes.pkl','wb') as file:
   pickle.dump(classes,file)

with open('model/tokenizer.pkl','wb') as file:
   pickle.dump(tokenizer,file)

with open('model/label_encoder.pkl','wb') as file:
   pickle.dump(label_encoder,file)

model = tf.keras.models.load_model('model/intent')

with open('model/classes.pkl','rb') as file:
    classes = pickle.load(file)
with open('model/tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)
with open('model/label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = tf.keras.preprocessing.sequence.pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
print(nlu.get_intent("Please set a reminder"))