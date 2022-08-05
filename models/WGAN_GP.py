import tensorflow as tf

"""
Code inspired/retrieved from https://keras.io/examples/generative/wgan_gp/
"""


class WGAN_GP(tf.keras.Model):
    def __init__(self, discriminator, generator, noise_dim, d_steps=3, gp_weight=10):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, real_imgs):
        if isinstance(real_imgs, tuple):
            real_imgs = real_imgs[0]

        batch_size = tf.shape(real_imgs)[0]
        # train discriminator for k steps
        for i in range(self.d_steps):
            noise = tf.random.normal([batch_size, self.noise_dim])

            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                # count D(x) and D(G(z))
                real_output = self.discriminator(real_imgs, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                # count gradient penalty and loss (wasserstein difference with means)
                grad_pen = self.gradient_penalty(real_imgs, generated_images, batch_size)
                disc_loss = self.discriminator_loss(real_output, fake_output) + self.gp_weight * grad_pen

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # train generator for 1 step
        noise = tf.random.normal([batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)

            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return {"d_loss": disc_loss, "g_loss": gen_loss}

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        return fake_loss - real_loss

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self, real_imgs, fake_imgs, batch_size):
        # Get the interpolated images
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_imgs - real_imgs
        interpolated = fake_imgs + alpha * diff
        # x + a(x~ - x) = x - ax + ax~ = (1-a)x + ax~ => a substitute (1-e) then -> ex + (1-e)x~ (as in original paper)
        # computation is shorter, result same

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
