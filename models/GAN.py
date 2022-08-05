import tensorflow as tf


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, noise_dim, d_steps=3):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.d_steps = d_steps

    def compile(self, d_optimizer, g_optimizer, cross_entropy):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.cross_entropy = cross_entropy

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
                # get discriminator binary cross entropy loss
                disc_loss = self.discriminator_loss(real_output, fake_output)

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
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
