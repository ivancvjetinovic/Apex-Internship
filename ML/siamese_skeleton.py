import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Model
from keras.layers import Dense, Embedding, Flatten, LSTM, Input, Lambda, Layer
from keras.utils import pad_sequences
from keras.losses import cosine_similarity
from keras import backend as K
from sklearn.model_selection import train_test_split

# implement attention mechanism
class AttentionMechanism(Layer):
	def __init__(self):
		super(AttentionMechanism, self).__init__()
	def call():
		return None

def contrastive_loss(y, preds, margin=1):
	y = tf.cast(y, preds.dtype)
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	return loss

# pad + tokenize data

# finish implementing the below
candidate = Input()
job = Input()

model = Sequential()
model.add(Embedding())
model.add(LSTM())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

candidate_encoding = model(candidate)
job_encoding = model(job)
distance = Lambda(cosine_similarity)([candidate_encoding, job_encoding])
siamese_model = Model([candidate, job], distance)
siamese_model.compile(loss=contrastive_loss, optimizer='adam')

# train the model below
# separate data into train, test, and validations
X = ""
y = ""
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
siamese_model.fit(x_train, y_train, epochs=10, batch_size=16)
