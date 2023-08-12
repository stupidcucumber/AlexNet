from src.model.model import instantiate_model

alexnet = instantiate_model(input_shape=(256, 256, 3))

print(alexnet.summary())