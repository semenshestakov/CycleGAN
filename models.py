import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa


def get_generator_on_vgg16(activate="sigmoid", _del=True) -> tf.keras.Model:
    """
    :return: Model on Trainable params: 3,926,091 and Non-trainable params: 14,714,688 on base VGG16

    Model.input (batch, None, None, 3)
    Model.output (batch, None, None, 3)
    """

    blocks = {
        "block1_conv2": 32,
        "block2_conv2": 64,
        "block3_conv3": 128,
        "block4_conv3": 256,
        "block5_conv3": 256,
    }

    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3)
    )
    vgg16.trainable = False

    x = None

    for block in sorted(blocks.keys(), reverse=True):

        l = vgg16.get_layer(block).output
        if x is not None:
            l = concatenate([
                l, x
            ])
        l = Conv2D(blocks[block], (3, 3), activation="relu", padding="same")(l)
        if block != "block1_conv2":
            x = Conv2DTranspose(blocks[block] // 2, (3, 3), strides=(2, 2), padding="same", activation="relu")(l)

    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)

    output = Conv2D(3, (2, 2), activation=activate, padding="same", name="output")(x)

    model = tf.keras.Model(vgg16.input, output, name="model_skip_on_vgg16")
    inp = Input((256, 256, 3))
    x = inp
    if _del:
        x = tf.keras.layers.Lambda(lambda x: x * 255.0)(x)
    x = model(x)
    return tf.keras.Model(inp, x)


def get_discriminator(leaky_relu_slope=0.2, depth=3, n=40, inp_shape=(256, 256, 3)) -> tf.keras.Model:
    """
    :param leaky_relu_slope:
    :param depth:
    :param dropout:
    :param n:
    :return: Model Trainable params: 2,228,569
    Model.input (batch, None, None, 3)
    Model.output (batch, )
    """

    x = inp = Input(shape=inp_shape)

    for _ in range(depth):
        x = Conv2D(n * 2, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
        x = tfa.layers.InstanceNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
        x = LeakyReLU(alpha=leaky_relu_slope)(x)
        n *= 2

    output = Conv2D(1, kernel_size=3, strides=2, padding="same")(x)
    return tf.keras.Model(inp, output, name="discriminator")


def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


class CycleGAN(tf.keras.Model):
    """
    on_real_accuracy_to_real - точность дискриминатора на реальных картинках для реальных изображений
    on_real_accuracy_to_fake - точность дискриминатора на сгенерированных для реальных изображений
    on_monet_accuracy_to_monet - точность дискриминатора на картинах для картин
    on_monet_accuracy_to_fake - точность дискриминатора на сгенерированных для картин
    total_cycle_loss
    total_gen_g_loss - Monet -> real gen
    total_gen_f_loss - real -> monet
    """

    def __init__(self, data_plot: np.array, path="", lambda_cycle=10.0, lambda_identity=0.5, bias_plot=0,
                 iterval_save=5,func_for_plot=None):

        super(CycleGAN, self).__init__()

        self.generator_g = get_generator_on_vgg16()
        self.generator_f = get_generator_on_vgg16()
        self.discriminator_x = get_discriminator()
        self.discriminator_y = get_discriminator()

        self.path = path
        self.save_iterval = iterval_save
        self.__bias_plot = bias_plot
        self._plot = np.clip(data_plot, 0, 255.0)
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self._lambda_plot = lambda x: x if func_for_plot is None else func_for_plot

    def discriminator_loss(self, real, generated):
        real_loss = self.loss(tf.ones_like(real), real)
        generated_loss = self.loss(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss(tf.ones_like(generated), generated)

    def compile(
            self,
            discriminator_optimizer_real=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            discriminator_optimizer_fake=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            generator_optimizer_real=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),  # 3
            generator_optimizer_fake=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),  # 3

            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

            on_real_accuracy_to_real=tf.keras.metrics.BinaryAccuracy(name="on_real_accuracy_to_real"),
            on_real_accuracy_to_fake=tf.keras.metrics.BinaryAccuracy(name="on_real_accuracy_to_fake"),
            on_monet_accuracy_to_monet=tf.keras.metrics.BinaryAccuracy(name="on_monet_accuracy_to_monet"),
            on_monet_accuracy_to_fake=tf.keras.metrics.BinaryAccuracy(name="on_monet_accuracy_to_fake"),

            **kwargs):
        super().compile(**kwargs)

        self.loss = loss
        self.discriminator_optimizer_real = discriminator_optimizer_real
        self.discriminator_optimizer_fake = discriminator_optimizer_fake
        self.generator_optimizer_real = generator_optimizer_real
        self.generator_optimizer_fake = generator_optimizer_fake

        self.on_real_accuracy_to_real = on_real_accuracy_to_real
        self.on_real_accuracy_to_fake = on_real_accuracy_to_fake
        self.on_monet_accuracy_to_monet = on_monet_accuracy_to_monet
        self.on_monet_accuracy_to_fake = on_monet_accuracy_to_fake

    def MAE(self, x, y):
        return tf.reduce_mean(tf.abs(x - y))

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = self.MAE(real_image, cycled_image)

        return loss1 * self.lambda_cycle

    def identity_loss(self, real_image, same_image):
        loss = self.MAE(real_image, same_image)
        return loss * self.lambda_identity * self.lambda_cycle

    @tf.function
    def train_step(self, data):
        x_monet, x_real = data
        real_x, real_y = x_monet, x_real
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)  # 1
            disc_real_y = self.discriminator_y(real_y, training=True)  # 1
            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        self.generator_optimizer_real.apply_gradients(zip(generator_g_gradients,
                                                          self.generator_g.trainable_variables))

        self.generator_optimizer_fake.apply_gradients(zip(generator_f_gradients,
                                                          self.generator_f.trainable_variables))

        self.discriminator_optimizer_real.apply_gradients(zip(discriminator_x_gradients,
                                                              self.discriminator_x.trainable_variables))

        self.discriminator_optimizer_fake.apply_gradients(zip(discriminator_y_gradients,
                                                              self.discriminator_y.trainable_variables))

        self.on_real_accuracy_to_real.update_state(1.0, step(disc_real_y))
        self.on_monet_accuracy_to_monet.update_state(1.0, step(disc_real_x))
        self.on_real_accuracy_to_fake.update_state(0.0, step(disc_fake_y))
        self.on_monet_accuracy_to_fake.update_state(0.0, step(disc_fake_x))

        stat = {
            s.name: s.result() for s in [
                self.on_real_accuracy_to_real,
                self.on_real_accuracy_to_fake,
                self.on_monet_accuracy_to_monet,
                self.on_monet_accuracy_to_fake
            ]
        }
        stat["total_cycle_loss"] = total_cycle_loss
        stat["monet->real"] = total_gen_g_loss
        stat["real->monet"] = total_gen_f_loss

        return stat

    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=5, interval=2):

        epoch = epoch + self.__bias_plot if epoch is not None else epoch
        num_cols = self._plot.shape[0]

        generated_images = self._lambda_plot(self.generator_f.predict(self._plot))
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if row == 0:
                    plt.imshow(self._lambda_plot(self._plot[col]))
                else:
                    plt.imshow(generated_images[col])

                plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'{self.path}image_at_epoch{epoch}.png')
        plt.show()

        if bool(epoch) and epoch % self.save_iterval == 0:
            self.discriminator_x.save(f"{self.path}discriminator_monet ep{epoch}")
            self.discriminator_y.save(f"{self.path}discriminator_real ep{epoch}")
            self.generator_f.save(f"{self.path}generator_real_to_monet ep{epoch}")
            self.generator_g.save(f"{self.path}generator_monet_to_real ep{epoch}")


if __name__ == '__main__':
    import data.extract_data as ed

    data = ed.Data(path="data/", batch_size=5)
    _, x_plot = data[0]  # 200mb
    x_plot = x_plot[:6]
    model = CycleGAN(x_plot)  # 1gb
    model.plot_images()
    # model.compile()
    # model.fit(data, epochs=10, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)])
    # model.fit(data,epochs=10,callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)])
    # get_discriminator().summary()
    # get_generator_on_vgg16().summary()
