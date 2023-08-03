import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np


def get_generator_on_vgg16() -> tf.keras.Model:
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
        x = Conv2DTranspose(blocks[block] // 2, (3, 3), strides=(2, 2), padding="same", activation="relu")(l)

    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)

    output = Conv2D(3, (2, 2), activation="sigmoid", padding="same", name="output")(x)

    return tf.keras.Model(vgg16.input, output)


def get_discriminator(leaky_relu_slope=0.2, depth=4, dropout=0.4, n=16) -> tf.keras.Model:
    """

    :param leaky_relu_slope:
    :param depth:
    :param dropout:
    :param n:
    :return: Model Trainable params: 1,244,641
    Model.input (batch, None, None, 3)
    Model.output (batch, 1)
    """

    x = inp = Input(shape=(28, 28, 1))
    x_fl = Flatten()(inp)
    x_fl = Dropout(dropout)(x_fl)
    x_fl = Dense(128, LeakyReLU(alpha=leaky_relu_slope))(x_fl)

    for _ in range(depth):
        x = Conv2D(n, kernel_size=3, padding="same", use_bias=False, )(x)
        x = BatchNormalization(scale=False)(x)
        x = LeakyReLU(alpha=leaky_relu_slope)(x)

        x = Conv2D(n * 2, kernel_size=4, strides=2, padding="same", use_bias=False, )(x)
        x = BatchNormalization(scale=False)(x)
        x = LeakyReLU(alpha=leaky_relu_slope)(x)
        n *= 2

    x = Flatten()(x)
    x = concatenate([x_fl, x])
    x = Dropout(dropout)(x)
    x = Dense(256, LeakyReLU(alpha=leaky_relu_slope))(x)
    x = Dropout(dropout)(x)
    x = Dense(128, LeakyReLU(alpha=leaky_relu_slope))(x)
    x = Dropout(dropout)(x)

    output = Dense(1)(x)
    return tf.keras.Model(inp, output, name="discriminator")


def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


class GAN(tf.keras.Model):
    """
    on_real_accuracy_to_real - точность дискриминатора на реальных картинках для реальных изображений
    on_real_accuracy_to_fake - точность дискриминатора на сгенерированных для реальных изображений
    on_monet_accuracy_to_monet - точность дискриминатора на картинах для картин
    on_monet_accuracy_to_fake - точность дискриминатора на сгенерированных для картин
    """

    def __init__(self, data_plot: np.array, path=""):
        super(GAN, self).__init__()

        self.generator_monet_to_real = get_generator_on_vgg16()
        self.generator_real_to_monet = get_generator_on_vgg16()
        self.discriminator_real = get_discriminator()
        self.discriminator_monet = get_discriminator()

        self._plot = data_plot

    def adversarial_loss(self, real_logits, generated_logits):
        real_labels = tf.ones(shape=(real_logits.shape[0], 1))
        generated_labels = tf.zeros(shape=(generated_logits.shape[0], 1))

        generator_loss = tf.keras.losses.mse(
            real_labels, generated_logits
        )
        discriminator_loss = tf.keras.losses.mse(
            tf.concat([real_labels, generated_labels], axis=0), tf.concat([real_logits, generated_logits], axis=0)
        )
        # tf.keras.losses.binary_crossentropy from_logits=True,

        return generator_loss, discriminator_loss

    def compile(
            self,
            discriminator_optimizer_real=tf.keras.optimizers.Adam(1e-4),
            discriminator_optimizer_fake=tf.keras.optimizers.Adam(1e-4),
            generator_optimizer_real=tf.keras.optimizers.Adam(1e-3),
            generator_optimizer_fake=tf.keras.optimizers.Adam(1e-3),

            on_real_accuracy_to_real=tf.keras.metrics.BinaryAccuracy(name="on_real_accuracy_to_real"),
            on_real_accuracy_to_fake=tf.keras.metrics.BinaryAccuracy(name="on_real_accuracy_to_fake"),
            on_monet_accuracy_to_monet=tf.keras.metrics.BinaryAccuracy(name="on_monet_accuracy_to_monet"),
            on_monet_accuracy_to_fake=tf.keras.metrics.BinaryAccuracy(name="on_monet_accuracy_to_fake"),

            **kwargs):
        super().compile(**kwargs)

        self.discriminator_optimizer_real = discriminator_optimizer_real
        self.discriminator_optimizer_fake = discriminator_optimizer_fake
        self.generator_optimizer_real = generator_optimizer_real
        self.generator_optimizer_fake = generator_optimizer_fake

        self.on_real_accuracy_to_real = on_real_accuracy_to_real
        self.on_real_accuracy_to_fake = on_real_accuracy_to_fake
        self.on_monet_accuracy_to_monet = on_monet_accuracy_to_monet
        self.on_monet_accuracy_to_fake = on_monet_accuracy_to_fake

    @tf.function
    def train_step(self, data):
        x_monet, x_real = data

        with tf.GradientTape() as tape_monet, tf.GradientTape() as tape_real:
            real_from_monet = self.generator_monet_to_real(x_monet, trainable=True)
            dis_res_real_gen = self.discriminator_real(real_from_monet, trainable=True)
            dis_res_real = self.discriminator_real(x_real, trainable=True)
            res_cycle_monet = self.generator_real_to_monet(real_from_monet, trainable=True)
            loss_cycle_monet = tf.losses.mse(x_monet, res_cycle_monet)
            loss_dis_monet_gen, loss_dis_real = self.adversarial_loss(dis_res_real, dis_res_real_gen)

            monet_from_real = self.generator_real_to_monet(x_real, trainable=True)
            dis_res_monet_gen = self.discriminator_monet(monet_from_real, trainable=True)
            dis_res_monet = self.discriminator_monet(x_monet, trainable=True)
            res_cycle_real = self.generator_monet_to_real(monet_from_real, trainable=True)
            loss_cycle_real = tf.losses.mse(x_real, res_cycle_real)
            loss_dis_real_gen, loss_dis_monet = self.adversarial_loss(dis_res_monet, dis_res_monet_gen)

        cycle_grad_monet = tape_monet.gradient(loss_cycle_monet, self.generator_monet_to_real.trainable_weights)
        dis_grad_monet_gen = tape_monet.gradient(loss_dis_monet_gen, self.generator_monet_to_real.trainable_weights)
        dis_grad_real = tape_monet.gradient(loss_dis_real, self.discriminator_real.trainable_weights)
        # real_from_gen_monet_grad = tape_monet.gradient(res_cycle_monet, self.generator_real_to_monet.trainable_weights)

        cycle_grad_real = tape_real.gradient(loss_cycle_real, self.generator_real_to_monet.trainable_weights)
        dis_grad_real_gen = tape_real.gradient(loss_dis_real_gen, self.generator_real_to_monet.trainable_weights)
        dis_grad_monet = tape_real.gradient(loss_dis_monet, self.discriminator_monet.trainable_weights)
        # monet_from_gen_real_grad = tape_real.gradient(res_cycle_real, self.generator_monet_to_real.trainable_weights)

        self.generator_optimizer_fake.apply_gradients(
            zip(cycle_grad_monet, self.generator_monet_to_real.trainable_weights))
        self.generator_optimizer_fake.apply_gradients(
            zip(dis_grad_monet_gen, self.generator_monet_to_real.trainable_weights))
        self.discriminator_optimizer_real.apply_gradients(zip(dis_grad_real, self.discriminator_real.trainable_weights))
        # self.generator_optimizer_real.apply_gradients(real_from_gen_monet_grad,self.generator_monet_to_real.trainable_weights)

        self.generator_optimizer_real.apply_gradients(
            zip(cycle_grad_real, self.generator_real_to_monet.trainable_weights))
        self.generator_optimizer_real.apply_gradients(
            zip(dis_grad_real_gen, self.generator_real_to_monet.trainable_weights))
        self.discriminator_optimizer_fake.apply_gradients(
            zip(dis_grad_monet, self.discriminator_monet.trainable_weights))
        # self.generator_optimizer_fake.apply_gradients(monet_from_gen_real_grad, self.generator_monet_to_real.trainable_weights)

        self.on_real_accuracy_to_real.update_state(1.0, step(dis_res_real))
        self.on_real_accuracy_to_fake.update_state(0.0, step(dis_res_real_gen))

        self.on_monet_accuracy_to_monet.update_state(1.0, step(dis_res_monet))
        self.on_monet_accuracy_to_fake.update_state(0.0, step(dis_res_monet_gen))

        stat = {
            s.name: s.result() for s in [
                self.on_real_accuracy_to_real,
                self.on_real_accuracy_to_fake,
                self.on_monet_accuracy_to_monet,
                self.on_monet_accuracy_to_fake
            ]
        }
        stat["mse_cycle_money"] = loss_cycle_monet
        stat["mse_cycle_real"] = loss_cycle_real
        stat["loss"] = loss_cycle_real + loss_cycle_monet

        return stat

    # def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=5, interval=2):
    #
    #     num_images = num_rows * num_cols
    #     generated_images = model_noise([self.__noise, self.__class_img])
    #     plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
    #     for row in range(num_rows):
    #         for col in range(num_cols):
    #             index = row * num_cols + col
    #             plt.subplot(num_rows, num_cols, index + 1)
    #             plt.imshow(generated_images[index], cmap="Greys")
    #             plt.axis("off")
    #     plt.tight_layout()
    #     plt.savefig(f'{self.path}image_at_epoch{epoch}.png')
    #     plt.show()

    # if epoch % 25 == 0 and epoch:
    #     self.discremenator.save(f"{self.path}Model disc ep{epoch}")
    #     self.model_noiser.save(f"{self.path}Model gen ep{epoch}")


# generator_optimizer = tf.keras.optimizers.Adam(1e-3)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
# noise_acc = tf.keras.metrics.BinaryAccuracy(name="gen_acc")
# disc_acc = tf.keras.metrics.BinaryAccuracy(name="disc_acc")


# model = GAN(model_noise, model_d,path=path)
# model.compile(
#     generator_optimizer, discriminator_optimizer,
# noise_acc, disc_acc
# )


if __name__ == '__main__':
# print(
#     get_generator_on_vgg16().summary()
# )
# print(
#     get_discriminator().summary()
# )
