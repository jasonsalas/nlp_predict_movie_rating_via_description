from flask import Flask, redirect, url_for, request, render_template
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

app = Flask(__name__, template_folder='templates')

tokenizer = pickle.load(open('../tokenizer.pkl', 'rb'))
model = load_model('../models/movie_description_classifier.h5')
graph = tf.get_default_graph()

def get_prediction(sequence):
	# dictionary of key-value mappings for MPAA ratings
	mpaa_ratings = { 0:'G', 1:'NC-17', 2:'NR', 3:'PG', 4:'PG-13', 5:'R' }
	sequence = tokenizer.texts_to_sequences(sequence)

	flat_list = []
	for sublist in sequence:
	    for item in sublist:
	        flat_list.append(item)

	sequence = pad_sequences([flat_list], maxlen=400)

	global graph
	with graph.as_default():
		prediction = model.predict_classes(sequence)

	return mpaa_ratings[prediction[0]]


@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		data = request.form.get('text')
		predicted_rating = get_prediction(data)

		return render_template('index.html', rating=predicted_rating)

	return render_template('index.html')


if __name__ == '__main__':
	app.run(port=5309, debug=True)