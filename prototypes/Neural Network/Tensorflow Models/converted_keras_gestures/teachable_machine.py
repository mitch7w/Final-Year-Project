from keras.models import load_model

# Load the model
model = load_model('keras_model.h5')
print(model.summary())