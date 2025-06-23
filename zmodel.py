import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
import pickle
from tensorflow.keras.callbacks import EarlyStopping



with open('LSTM DATA.txt',encoding='utf-8') as file:
    data = file.read()

tokeniser = Tokenizer()
tokeniser.fit_on_texts([data])
word_index = tokeniser.word_index
print(len(tokeniser.word_index))


input_sentense = []
for sentenses in data.split('\n'):
  """print(sentenses)
  print(tokeniser.texts_to_sequences([sentenses])[0])"""
  tokanised_sentenses = tokeniser.texts_to_sequences([sentenses])[0]

  for i in range(1,len(tokanised_sentenses)):
    input_sentense.append(tokanised_sentenses[:i+1])

max_len = max([len(x) for x in input_sentense])
print(max_len)

padded_input_sequences = pad_sequences(input_sentense,max_len,padding='pre')

x = padded_input_sequences[:,:-1]
y = padded_input_sequences[:,-1]

y = to_categorical(y,num_classes=len(tokeniser.word_index)+1)

model = Sequential()
model.add(Embedding(len(tokeniser.word_index)+1,320,input_length=max_len-1))
model.add(LSTM(200))
#model.add(Dropout(0.1))
model.add(Dense(len(tokeniser.word_index)+1,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
early_stop = EarlyStopping(monitor='loss', patience=2,restore_best_weights=True)
history = model.fit(x,y,epochs=20,callbacks=[early_stop])
print(history)

#model.save('Nest_word_predictor1.keras')
pickle.dump(model, open("Nest_word_predictor1.pkl", "wb"))
#pickle.load(open("model.pkl", "rb"))


# Optionally save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokeniser, f)




