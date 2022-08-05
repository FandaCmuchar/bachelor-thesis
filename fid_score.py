import warnings
import numpy as np
from scipy.linalg import sqrtm
from keras.datasets import cifar10
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Code was retrieved from:
# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# https://github.com/bioinf-jku/TTUR/blob/master/fid.py


class FID:
    def __init__(self, train_images, resize_shape=(299, 299, 3), batch_size=128):
        self.resize_shape = resize_shape
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=self.resize_shape)
        self.batch_size = batch_size
        self.train_len = len(train_images)

        # calculate mean and cov for training data
        self.train_mean, self.train_cov = self.calculate_mean_cov(train_images, self.batch_size)

    # scale an array of images to a new size
    def scale_images(self, images):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, self.resize_shape, 0)
            # store
            images_list.append(new_image)
        return np.asarray(images_list)

    # calculate frechet inception distance
    def calculate_mean_cov(self, images, batch_size):
        num_batches = len(images) // batch_size
        if len(images) > num_batches * batch_size:
            num_batches += 1
        pred_arr = np.zeros((len(images), 2048))
        for i in range(1, num_batches):
            start = (i - 1) * batch_size
            end = start + batch_size

            batch = images[start:end]
            batch = self.scale_images(batch.astype('float32'))
            batch = preprocess_input(batch)
            batch_res = self.model.predict(batch)
            pred_arr[start:end] = batch_res.reshape(len(batch), -1)
            del batch

        print('Mean and cov calculated')
        return np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)

    def calculate_fid(self, images):
        gen_mean, gen_cov = self.calculate_mean_cov(images, self.batch_size)

        # calculate sqrt of product between cov
        dot = self.train_cov.dot(gen_cov)
        if np.isnan(dot).any():
            dot = np.nan_to_num(dot)
        if np.isinf(dot).any():
            dot[~np.isfinite(dot)] = 10.e3
        covmean = sqrtm(dot)

        if not np.isfinite(covmean).all():
            offset = np.eye(gen_cov.shape[0]) * 1e6
            covmean = sqrtm((gen_cov + offset).dot(self.train_cov + offset))

        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # calculate score
        diff = self.train_mean - gen_mean
        fid = diff.dot(diff) + np.trace(self.train_cov + gen_cov - 2.0 * covmean)

        if fid == np.inf or fid == np.nan:
            warnings.warn(f"FID was calculated as: {fid}")

        return fid


if __name__ == '__main__':
    # simple example that loads cifar10 dataset and calculate fid score for test to train images
    (images1, _), (images2, _) = cifar10.load_data()
    np.random.shuffle(images1)
    fid = FID(images1[:10000], (299, 299, 3), 128)
    del images1
    print('FID: %.3f' % fid.calculate_fid(images2[:10000]))

