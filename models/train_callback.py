import tensorflow as tf
from .helper_functions import save_fid, save_loss, save_imgs

"""
Implement callback functions that are called in learning process
"""


class TrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, noise, output_path, save_rate, img_save_rate, checkpoint, prefix, fid=None, fid_save_rate=None):
        self.output_path = output_path
        self.save_rate = save_rate
        self.img_save_rate = img_save_rate
        self.checkpoint = checkpoint
        self.checkpoint_prefix = prefix
        self.observed_noise = noise
        self.batch_loss = None
        self.fid = fid
        self.fid_score = None
        self.fid_save_rate = fid_save_rate if fid_save_rate is not None else img_save_rate

    def on_train_begin(self, logs=None):
        self.batch_loss = [[], []]
        self.fid_score = []

    def on_batch_end(self, batch, logs):
        self.batch_loss[0].append(logs.get('d_loss'))
        self.batch_loss[1].append(logs.get('g_loss'))

    def on_epoch_end(self, epoch, logs=None):
        # save checkpoint
        if epoch % self.save_rate == 0:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        # save image and loss graph
        if epoch % self.img_save_rate == 0:
            imgs = self.model.generator(self.observed_noise, training=False)
            save_imgs(f"{epoch}. epoch", 'image_at_epoch_{:04d}.png'.format(epoch), 4, 4, imgs, self.output_path)

            save_loss(epoch, self.batch_loss, self.output_path + f"/loss_at_{epoch}.png",
                      x_label='epoch', y_label='loss', title='Training loss')

        # calculate FID score and save graph
        if epoch % self.fid_save_rate == 0 and self.fid is not None:
            noise = tf.random.normal(shape=(self.fid.train_len, self.model.noise_dim))
            images = self.model.generator.predict(noise)
            self.fid_score.append(self.fid.calculate_fid(images))
            print(f"FID at epoch {epoch} is: {self.fid_score[-1]}")
            save_fid(epoch, self.fid_score, self.output_path + f"/fid_score_at_{epoch}.png",
                     x_label='epoch', y_label='fid', title='FID score')
