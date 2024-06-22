from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, Sequential
import numpy as np
from tensorflow import keras

def read_dataset(n_samples):
    ret = []
    for i in range(n_samples):
        ret.append(np.load(f'data/raw/sample_{i}/boundary.npy', 'r'))
    return np.array(ret)

def normalize(X: np.array):
    X = X.reshape((X.shape[0] * int(X.shape[1] / 2), 2))
    min_val = np.min(X, axis=0).reshape((2, 1))
    max_val = np.max(X, axis=0).reshape((2, 1))
    X = (X.T - min_val) / (max_val - min_val)
    X = X.T
    X = X.reshape(500, 360, 2)
    return X, min_val, max_val

def denormalize_one(X: np.array, min, max):
    X = X.reshape((X.shape[0] * X.shape[1], 2))
    X = (X.T * (max - min)) + min
    X = X.T
    X = X.reshape(1, 720)
    return X

def main():
    X = np.load('data_registered.npy')
    X, min_val, max_val = normalize(X)
    encoder = Sequential([
        Conv1D(256, 5, activation='relu', padding='same'),
        MaxPooling1D(2, padding='same'),
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(2, padding='same'),
        Conv1D(64, 5, activation='relu', padding='same'),
        MaxPooling1D(2, padding='same'),
        Conv1D(32, 5, activation='relu', padding='same'),
        MaxPooling1D(3, padding='same'), 
        Conv1D(16, 5, activation='relu', padding='same'),
        MaxPooling1D(3, padding='same'),
    ])

    decoder = Sequential([
        Conv1D(16, 5, activation='relu', padding='same'),
        UpSampling1D(3),
        Conv1D(32, 5, activation='relu', padding='same'),
        UpSampling1D(3),
        Conv1D(64, 5, activation='relu', padding='same'),
        UpSampling1D(2),
        Conv1D(128, 5, activation='relu', padding='same'),
        UpSampling1D(2),
        Conv1D(256, 5, activation='relu', padding='same'),
        UpSampling1D(2),
        Conv1D(2, 3, activation='linear', padding='same') 
    ])
    input = keras.Input(shape=(360, 2))
    latent_vector = encoder(input)
    output = decoder(latent_vector)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(
         optimizer="adam",
         loss=keras.losses.MeanSquaredError(),
    )
    model.fit(X, X, epochs=50)
    test_sample = X[np.random.randint(0, X.shape[0])]
    pred = model.predict(np.array([test_sample]))
    pred = denormalize_one(pred, min_val, max_val)
    test_sample = denormalize_one(np.array([test_sample]), min_val, max_val)
    np.save('generated_data/pred_conv.npy', pred)
    np.save('generated_data/test_sample_conv.npy', test_sample)

if __name__ == "__main__":
    main()