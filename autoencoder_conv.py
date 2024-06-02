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

def main():
    LATENT_SIZE = 10
    X = read_dataset(500)[:, :, :2]
    # X = X.reshape((500, 720))
    min_val = np.min(X, axis=1).reshape((X.shape[0], 1, 2))
    max_val = np.max(X, axis=1).reshape((X.shape[0], 1, 2))
    X = (X - min_val) / (max_val - min_val) # [0, 1]

    encoder = Sequential([
        Conv1D(16, 5, activation='relu', padding='same'),
        MaxPooling1D(2, padding='same'),
        Conv1D(32, 5, activation='relu', padding='same'),
        MaxPooling1D(2, padding='same'),
        Conv1D(64, 5, activation='relu', padding='same'),
        MaxPooling1D(2, padding='same'),
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(3, padding='same'), 
        Conv1D(256, 5, activation='relu', padding='same'),
        MaxPooling1D(3, padding='same'),
    ])

    decoder = Sequential([
        Conv1D(256, 5, activation='relu', padding='same'),
        UpSampling1D(3),
        Conv1D(128, 5, activation='relu', padding='same'),
        UpSampling1D(3),
        Conv1D(64, 5, activation='relu', padding='same'),
        UpSampling1D(2),
        Conv1D(32, 5, activation='relu', padding='same'),
        UpSampling1D(2),
        Conv1D(16, 5, activation='relu', padding='same'),
        UpSampling1D(2),
        Conv1D(X.shape[2], 3, activation='linear', padding='same') 
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
    pred = pred[0] * (max_val - min_val) + min_val
    np.save('generated_data/pred.npy', pred)
    np.save('generated_data/test_sample.npy', test_sample)

if __name__ == "__main__":
    main()