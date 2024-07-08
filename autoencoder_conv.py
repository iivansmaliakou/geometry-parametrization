from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

def read_dataset(n_samples):
    ret = []
    for i in range(n_samples):
        ret.append(np.load(f'data/raw/sample_{i}/boundary.npy', 'r'))
    return np.array(ret)

def normalize(X: np.array):
    X = X.reshape((X.shape[0] * X.shape[1], 2))
    min_val = np.min(X, axis=0).reshape(1, 2)
    max_val = np.max(X, axis=0).reshape(1, 2)
    X = (X.T - min_val.T) / (max_val - min_val).T
    X = X.T
    X = X.reshape((500, 360, 2))
    return X, min_val, max_val

def denormalize(X: np.array, min, max):
    X = (X * (max - min)) + min
    return X

def main():
    X = np.load('reg_boundary.npy')
    X, min_val, max_val = normalize(X)
    train_size = int(X.shape[0] * 0.8)
    test_size = X.shape[0] - train_size
    X_train_idx = np.random.choice([True, False], size=X.shape[0],
                       replace=True, p=[0.8, 0.2])
    X_train = X[X_train_idx]
    X_test = X[np.invert(X_train_idx)]

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
        Conv1D(X.shape[-1], 3, activation='linear', padding='same') 
    ])
    input = keras.Input(shape=(360, 2))
    latent_vector = encoder(input)
    output = decoder(latent_vector)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(
         optimizer="adam",
         loss=keras.losses.MeanSquaredError(),
    )
    print(model.summary())
    model.fit(X_train, X_train, epochs=40)
    test_sample = X_test[np.random.randint(0, X_test.shape[0])]
    pred = model.predict(np.array([test_sample]))
    pred = denormalize(pred, min_val, max_val)
    test_sample = denormalize(np.array([test_sample]), min_val, max_val)
    np.save('generated_data/pred_conv.npy', pred[0])
    np.save('generated_data/test_sample_conv.npy', test_sample[0])
    
    # calculate loss
    X_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(X_test, X_test_pred)
    print(f'mse is {mse_test}')
if __name__ == "__main__":
    main()