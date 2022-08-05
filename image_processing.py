import os
import numpy as np
from PIL import Image
from pathlib import Path


def load_images(path, image_shape=(32, 32, 3), bin_file=''):
    r, c, ch = image_shape[0], image_shape[1], image_shape[2]

    if os.path.isfile(path + "/" + bin_file):
        # load images from binary file if specified (already normalized)
        train_images = np.fromfile(path + "/" + bin_file, dtype=np.float32)
        train_images = train_images.reshape(train_images.shape[0] // (r * c * ch), r, c, ch).astype('float32')
    else:
        # load and normalize all png files from given path
        images = list(Path(path).glob('*.png'))
        np_images = []

        for img in images:
            im = Image.open(img).convert("RGBA")  # load img as RGBA
            im.load()
            image = Image.new("RGB", im.size, (255, 255, 255))
            image.paste(im, mask=im.split()[3])  # convert imt to RGB with white bg

            if image.size != (r, c):
                image = image.resize((r, c))

            np_images.append(np.asarray(image))

        np_images = np.asarray(np_images)
        train_images = np_images.reshape(np_images.shape[0], r, c, ch).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    return train_images


def _save_imgs_as_bin(path, file_name_with_path):
    imgs = load_images(path)
    imgs.tofile(file_name_with_path)


if __name__ == '__main__':
    pth = './datasets/'
    imgs = load_images(pth, image_shape=(32, 32, 3), bin_file='swords_images.bin')
    print(imgs.shape)
    i = 0
    while input() == "1" or i >= len(imgs):
        PIL_image = Image.fromarray(np.uint8((imgs[i] + 127.5) * 127.5)).convert('RGB')
        PIL_image.show()
        i += 1
