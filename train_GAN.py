import os
import argparse
import tensorflow as tf
from keras import layers
from fid_score import FID
from models.GAN import GAN
from models.WGAN_GP import WGAN_GP
from image_processing import load_images
from optimize_WGAN import load_best_model
from models.train_callback import TrainCallback
from models.helper_functions import save_fid, save_loss, create_gif, gen_and_show_images, save_imgs


def make_generator_model(noise_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(2 * 2 * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((2, 2, 256)))
    assert model.output_shape == (None, 2, 2, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model


def make_discriminator_model(img_shape):
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


IMG_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 20  # 300
NOISE_DIMENSION = 32
num_examples_to_generate = 16
save_rate = 20  # 50
save_img_rate = 5  # 10
save_fid_rate = 5  # 50


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', '--input_data_dir', type=str, default="./datasets/")
    parser.add_argument('-data_bin', '--data_binary_file_name', type=str, default="swords_images.bin")
    parser.add_argument('-out_dir', '--output_directory', type=str, default='./train_output/')
    parser.add_argument('-load_optimized', '--load_optimized_hyperparameters', type=bool, default=False)
    parser.add_argument('-opt_dir', '--optimized_model_dir', type=str, default="./Swords_optimization/")
    parser.add_argument('-use_wgan', '--use_wgan_model', type=bool, default=True)
    parser.add_argument('-train', '--train', type=bool, default=True)
    args = vars(parser.parse_args())

    output_img_path = f"{args['output_directory']}/output_imgs/"
    checkpoint_dir = f"{args['output_directory']}/checkpoints/"
    optimized_model_dir = args['optimized_model_dir']
    input_dir = args['input_data_dir']
    data_bin_file_name = args['data_binary_file_name']

    if not os.path.isdir(args['output_directory']):
        os.mkdir(args['output_directory'])

    if not os.path.isdir(output_img_path):
        os.mkdir(output_img_path)

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Load optimized hyperparameters and starts new training
    if args['load_optimized_hyperparameters']:
        loaded_models = load_best_model(optimized_model_dir)
        generator = loaded_models['generator']
        discriminator = loaded_models['discriminator']
        noise_dim = loaded_models['noise_dim']
    else:
        discriminator = make_discriminator_model(IMG_SHAPE)
        generator = make_generator_model(NOISE_DIMENSION)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    if args['train']:
        train_images = load_images(input_dir, IMG_SHAPE, data_bin_file_name)
        print(f"images loaded len: {len(train_images)}")

        observed_noise = tf.random.normal(shape=(num_examples_to_generate, NOISE_DIMENSION))
        fid = FID(train_images)

        callback = TrainCallback(noise=observed_noise, output_path=output_img_path, save_rate=save_rate,
                                 img_save_rate=save_img_rate, checkpoint=checkpoint, prefix=checkpoint_prefix, fid=fid,
                                 fid_save_rate=save_fid_rate)
        model = None
        if args['use_wgan_model']:
            model = WGAN_GP(discriminator, generator, NOISE_DIMENSION)
            model.compile(generator_optimizer, discriminator_optimizer)
        else:
            model = GAN(discriminator, generator, NOISE_DIMENSION)
            model.compile(generator_optimizer, discriminator_optimizer,
                          tf.keras.losses.BinaryCrossentropy(from_logits=True))

        model.fit(x=train_images, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback)

        save_loss(EPOCHS, callback.batch_loss, output_img_path + f"/loss_at_{EPOCHS}.png",
                  x_label='epoch', y_label='loss', title='Training loss')
        save_fid(EPOCHS, callback.fid_score, output_img_path + f"/fid_score_at_{EPOCHS}.png",
                 x_label='epoch', y_label='fid', title='FID score')
        create_gif(output_img_path)
    else:
        # loads trained model than generate and shows images
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint.restore(latest_checkpoint)
        gen_and_show_images(num_examples_to_generate, NOISE_DIMENSION, generator)

