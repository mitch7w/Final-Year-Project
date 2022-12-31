from keras.models import load_model

# Load the model
model = load_model('keras_model.h5', compile=False)
print(model.summary())

# for layer in model.layers:
#     print(layer.output_shape)

# first_layer_weights = model.layers[0].get_weights()[0]
# first_layer_biases  = model.layers[0].get_weights()[1]
# second_layer_weights = model.layers[1].get_weights()[0]
# second_layer_biases  = model.layers[1].get_weights()[1]
# print(first_layer_weights.shape) # 3 x 3 x 3 x 16 = 432
# print(second_layer_weights.shape)

# 224 x 224 x 3 = 150528
# 1280 * 100 = 128000

# (410208-1280)/1280 = 319.475 input shape
# (128300-2)/2 = 64149