from tensorflow.keras.utils import plot_model, model_to_dot
from tensorflow.keras.models import load_model
import graphviz
model = load_model('app/models/neural_network_model.h5')
print(model.summary())
plot_model(model, to_file='model_architecture2.png', show_shapes=True, show_layer_names=True, dpi=200)