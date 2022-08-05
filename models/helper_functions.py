import math
import glob
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def save_loss(epochs, loss, file_name, x_label, y_label, title):
    plt.figure()
    plt.axis('on')
    plt.clf()

    x_epochs = np.linspace(0, epochs, len(loss[0]))

    plt.plot(x_epochs, loss[1], 'r', label='gen loss')
    plt.plot(x_epochs, loss[0], 'g', label='disc loss')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.savefig(file_name)
    plt.close('all')


def save_fid(epochs, fid_score, file_name, x_label, y_label, title):
    plt.figure()
    plt.axis('on')
    plt.clf()

    x_epochs = np.linspace(0, epochs, len(fid_score))

    plt.plot(x_epochs, fid_score, 'b', label='fid score')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.savefig(file_name)
    plt.close('all')


def create_gif(output_img_path):
    anim_file = output_img_path + '/dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(output_img_path + '/image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def save_imgs(title, file_name, row_cnt, col_cnt, generated_imgs, output_path):
    fig = plt.figure(figsize=(col_cnt, row_cnt))

    for i in range(generated_imgs.shape[0]):
        plt.subplot(row_cnt, col_cnt, i + 1)
        plt.imshow(generated_imgs[i] * 0.5 + 0.5, cmap='gray' if generated_imgs[i].shape[-1] == 1 else None)
        plt.axis('off')

    plt.title(title, pad=180)
    plt.savefig(output_path + f'/{file_name}')
    # plt.show()
    plt.close(fig)


def gen_and_show_images(images_count, noise_dim, generator):

    fig_size = math.sqrt(images_count)
    rows = fig_size
    rows = int(rows + 1) if rows - int(rows) != 0.0 else int(rows)
    figure = plt.figure(figsize=(rows, int(fig_size)))

    while input() != 'q':
        random_input = tf.random.normal([images_count, noise_dim], mean=0.0, stddev=1.0)
        images = generator(random_input, training=False)

        for i in range(images.shape[0]):
            plt.subplot(rows, int(fig_size), i + 1)
            plt.imshow(images[i] * 0.5 + 0.5, cmap='gray' if images[i].shape[-1] == 1 else None)
            plt.axis('off')

        plt.show()