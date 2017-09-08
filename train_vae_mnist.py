import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.datasets import mnist
from keras.optimizers import RMSprop
from vae import VariationalAutoEncoder

# set parameters
batch_size = 100
latent_dim = 2
nr_epochs = 30
layers = [256, 128]
optimizer = RMSprop(lr=1e-3)

# get MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Train model
original_dim = x_train.shape[1]
vae_obj = VariationalAutoEncoder(original_dim, layers, activation='relu', optimizer=optimizer, dropout=0.25)
vae = vae_obj.compile()
vae.summary()
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=nr_epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test), verbose=2)

# get the model that projects inputs on the latent space
encoder = vae_obj.encoder

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# get a digit generator that can sample from the learned distribution
generator = vae_obj.decoder

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
