import math
import argparse
import numpy as np
import tensorflow as tf
from keras import layers
import keras_tuner as kt
from fid_score import FID
from models.WGAN_GP import WGAN_GP
from image_processing import load_images
from models.helper_functions import gen_and_show_images, save_imgs


def make_generator_model(noise_dim, g_kernel_sizes, num_hidden_layers, img_size=(32, 32, 3)):
    """
        Build generator model with given hyperparameters
    """
    model = tf.keras.Sequential(name="Generator")

    # chooses between 8,4,2 if img_size has 32x32 px or 16,8,4 if 64x64 px, for 2, 3, 4 hidden layers
    units_row_col = int(2 ** (math.log2(img_size[0])-num_hidden_layers))

    # 256, 512, 1024 for 2,3,4 hidden layers
    units_size = 2 ** (6 + num_hidden_layers)

    model.add(layers.Dense(units=units_row_col * units_row_col * units_size, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((units_row_col, units_row_col, units_size)))

    for i in range(int(num_hidden_layers)):
        model.add(
            layers.Conv2DTranspose(
                filters=units_size / (2 ** (i + 1)),
                kernel_size=g_kernel_sizes[i],
                strides=(2, 2), padding='same', use_bias=False
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(img_size[-1], (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.Activation('tanh'))
    assert model.output_shape == (None, img_size[0], img_size[1], img_size[2])

    return model


def make_discriminator_model(dropout_rate, kernel_sizes, dropout_hidden, dropout_last, num_hidden_layers):
    """
    Build discriminator model with given hyperparameters
    """
    model = tf.keras.Sequential(name="Discriminator")
    model.add(layers.InputLayer(input_shape=IMG_SHAPE))

    for i in range(num_hidden_layers):
        model.add(
            layers.Conv2D(
                filters=2**(6+i),  # starts at 64, then increases with power of two
                kernel_size=kernel_sizes[i],  # chooses between (4,4) and (5,5)
                strides=(2, 2), padding='same'
            )
        )
        model.add(layers.LeakyReLU())
        if dropout_hidden[i]:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())
    if dropout_last:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))

    return model


class HyperWGAN(kt.HyperModel):
    """
    Serves as optimization class
    """
    def build(self, hp):
        """ build is called for each optimization trial, new hyperaprameters are chosen here"""
        noise_size = hp.Choice("noise_size", [128, 256, 512])  # 128, 256, 512
        # g_lr = 0.0002, d_lr = 0.0002, beta_1 = 0.5, beta_2 = 0.9
        num_hidden_layers = hp.Choice("d_num_layers", [2, 3, 4])
        g_kernel_sizes = [hp.Choice(f'g_kernel_size_{i}', [4, 5]) for i in range(num_hidden_layers)]
        d_kernel_sizes = [hp.Choice(f'd_kernel_size_{i}', [4, 5]) for i in range(num_hidden_layers)]
        d_dropout_hidden = [hp.Boolean(f'dropout_{i}') for i in range(num_hidden_layers)]

        discriminator = make_discriminator_model(dropout_rate=hp.Choice(f"d_dropout", [0.2, 0.3, 0.4]),
                                                 num_hidden_layers=num_hidden_layers,
                                                 kernel_sizes=d_kernel_sizes,
                                                 dropout_last=hp.Boolean('dropout last'),
                                                 dropout_hidden=d_dropout_hidden)
        generator = make_generator_model(noise_dim=noise_size, g_kernel_sizes=g_kernel_sizes,
                                         num_hidden_layers=num_hidden_layers, img_size=IMG_SHAPE)

        wgan_model = WGAN_GP(discriminator, generator, noise_size)

        # According to original paper these parameters are advised
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

        wgan_model.compile(discriminator_optimizer, generator_optimizer)

        return wgan_model

    def fit(self, hp, model, *args, **kwargs):
        train_data_len = len(kwargs['x'])
        print(f"train data len: {train_data_len}")

        model.generator.summary()
        model.discriminator.summary()

        model.fit(batch_size=hp.Choice("batch_size", [32, 64, 128, 256]), *args, **kwargs)

        random_noise = tf.random.normal(shape=(train_data_len, model.noise_dim))
        generate_imgs = model.generator.predict(random_noise)

        fid_score = fid.calculate_fid(generate_imgs)

        imgs = model.generator.predict(tf.random.normal(shape=(16, model.noise_dim)))
        save_imgs(title="Imgs", file_name=f"imgs_with_{fid_score:.4f}.png", col_cnt=4, row_cnt=4,
                  generated_imgs=imgs, output_path='./')

        return fid_score


def load_best_model(project_name):
    """ Loads best hyperparameters """
    tuner = kt.BayesianOptimization(hypermodel=HyperWGAN(), max_trials=10, overwrite=False,
                                    directory='./', project_name=project_name)

    trials = tuner.oracle.trials

    # Print out the ID and the score of all trials
    for trial_id, trial in trials.items():
        print(trial_id, trial.score)

    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)

    return {'generator': model.generator, "discriminator": model.discriminator, "noise_dim": model.noise_dim}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trials', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default="100")
    parser.add_argument('-in_dir', '--input_data_dir', type=str, default="./datasets/")
    parser.add_argument('-data_bin', '--data_binary_file_name', type=str, default="swords_images.bin")
    parser.add_argument('-p_name', '--project_name', type=str, default="Swords_optimization")
    parser.add_argument('-out_dir', '--output_directory', type=str, default="./")
    args = vars(parser.parse_args())

    trials = args['trials']
    output_dir = args['output_directory']
    epochs = args['epochs']

    IMG_SHAPE = (32, 32, 3)
    train_images = load_images(args['input_data_dir'], IMG_SHAPE, args['data_binary_file_name'])
    print(f"images loaded len: {len(train_images)}")

    fid = FID(train_images)

    tuner = kt.BayesianOptimization(hypermodel=HyperWGAN(), max_trials=trials, overwrite=False,
                                    directory=output_dir, project_name='Swords_optimization')

    tuner.search(x=train_images, epochs=epochs)
    tuner.search_space_summary()
    tuner.results_summary()

    best_model = tuner.get_best_models()[0]

    # generate images with best model, waits for input (until "q" is passed)
    gen_and_show_images(16, best_model.noise_dim, best_model.generator)

