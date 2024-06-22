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
    X = np.load('data_registered.npy')
    min_val = np.min(X, axis=1).reshape((X.shape[0], 1))
    max_val = np.max(X, axis=1).reshape((X.shape[0], 1))
    X = (X - min_val) / (max_val - min_val) # [0, 1]

    encoder = Sequential([
        Dense(500, activation='relu', use_bias=True),
        Dense(200, activation='relu', use_bias=True),
        Dense(100, activation='relu', use_bias=True),
        Dense(50, activation='relu', use_bias=True),
        Dense(25, activation='relu', use_bias=True),
        Dense(10, activation='relu', use_bias=True),
    ])
    decoder = Sequential([
        Dense(10, activation='relu', use_bias=True),
        Dense(25, activation='relu', use_bias=True),
        Dense(50, activation='relu', use_bias=True),
        Dense(100, activation='relu', use_bias=True),
        Dense(200, activation='relu', use_bias=True),
        Dense(500, activation='relu', use_bias=True),
        Dense(X.shape[1], activation='relu', use_bias=True),
    ])
    input = keras.Input(shape=(720,))
    latent_vector = encoder(input)
    output = decoder(latent_vector)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(
         optimizer="adam",
         loss=keras.losses.MeanSquaredError(),
    )
    model.fit(X, X, epochs=10, shuffle=True)
    test_sample_index = np.random.randint(0, X.shape[0])
    test_sample = X[test_sample_index]
    
    pred = model.predict(np.array([test_sample]))
    
    # scale back both sample and result
    pred = pred[0] * (max_val[test_sample_index] - min_val[test_sample_index]) + min_val[test_sample_index]
    test_sample = test_sample * (max_val[test_sample_index] - min_val[test_sample_index]) + min_val[test_sample_index]
    np.save('generated_data/pred_dense.npy', pred)
    np.save('generated_data/test_sample_dense.npy', test_sample)

if __name__ == "__main__":
    main()