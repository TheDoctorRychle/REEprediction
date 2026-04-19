from model.mlp import MLP
from train.train import train_model
from utils.helpers import generate_dummy_data, plot_predictions
import config as cfg

X, y = generate_dummy_data(200)

model = MLP(input_size=cfg.INPUT_SIZE, hidden_layers=cfg.HIDDEN_LAYERS, output_size=cfg.OUTPUT_SIZE)

train_model(model, X, y, epochs=cfg.EPOCHS, lr=cfg.LEARNING_RATE)

y_pred = model.forward_propagation(X)
plot_predictions(y, y_pred)