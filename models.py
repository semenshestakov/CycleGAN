import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import matplotlib.pyplot as plt


def get_generator_on_vgg16() -> tf.keras.Model:
    """
    :return: Model on Trainable params: 3,926,091 and Non-trainable params: 14,714,688 on base VGG16

    Model.input (batch, None, None, 3)
    Model.output (batch, None, None, 3)
    """

    def sigmoid(x):
        return tf.nn.sigmoid(x)

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
        if block != "block1_conv2":
            x = Conv2DTranspose(blocks[block] // 2, (3, 3), strides=(2, 2), padding="same", activation="relu")(l)

    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)

    output = Conv2D(3, (2, 2), activation=sigmoid, padding="same", name="output")(x)

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

    x = inp = Input(shape=(256, 256, 3))
    x_fl = Flatten()(inp)
    x_fl = Dropout(dropout)(x_fl)
    x_fl = Dense(256, LeakyReLU(alpha=leaky_relu_slope))(x_fl)

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
    x = Dense(512, LeakyReLU(alpha=leaky_relu_slope))(x)
    x = Dropout(dropout)(x)
    x = Dense(256, LeakyReLU(alpha=leaky_relu_slope))(x)
    x = Dropout(dropout)(x)

    output = Dense(1)(x)
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
    cycle_money -
    cycle_real -
    """

    def __init__(self, data_plot: np.array, path="", lambda_cycle=10.0, lambda_identity=0.5, batch_size=20, bias_plot=0,
                 iterval_save=5):

        super(CycleGAN, self).__init__()

        self.generator_monet_to_real = get_generator_on_vgg16()
        self.generator_real_to_monet = get_generator_on_vgg16()
        self.discriminator_real = get_discriminator()
        self.discriminator_monet = get_discriminator()

        self.path = path
        self.save_iterval = iterval_save
        self.__bias_plot = bias_plot
        self._plot = np.clip(data_plot, 0, 255.0)
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.BATCH = batch_size

    def adversarial_loss(self, real_logits, generated_logits):
        # try:/
        real_labels = tf.ones(shape=(real_logits.shape[0], 1))
        generated_labels = tf.zeros(shape=(generated_logits.shape[0], 1))
        # except TypeError:
        #     real_labels = tf.ones(shape=(1, 1))
        #     generated_labels = tf.zeros(shape=(1, 1))

        generator_loss = self.loss(
            real_labels, generated_logits,
        )
        discriminator_loss = self.loss(
            tf.concat([real_labels, generated_labels], axis=0), tf.concat([real_logits, generated_logits], axis=0),
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)

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

    @tf.function
    def train_step(self, data):
        x_monet, x_real = data
        fl = True
        if x_monet.shape[0] is None:
            print("Start...")
            x_monet, x_real = tf.random.normal(shape=(self.BATCH, 256, 256, 3)), tf.random.normal(
                shape=(self.BATCH, 256, 256, 3))
        with tf.GradientTape(persistent=True) as tape_monet, tf.GradientTape(persistent=True) as tape_real:
            real_from_monet = self.generator_monet_to_real(x_monet, training=fl)
            dis_res_real_gen = self.discriminator_real(real_from_monet, training=fl)
            dis_res_real = self.discriminator_real(x_real, training=fl)
            res_cycle_monet = self.generator_real_to_monet(real_from_monet, training=fl)
            loss_cycle_monet = self.MAE(x_monet, res_cycle_monet) * self.lambda_cycle
            loss_dis_monet_gen, loss_dis_real = self.adversarial_loss(dis_res_real, dis_res_real_gen)

            monet_from_real = self.generator_real_to_monet(x_real, training=fl)
            dis_res_monet_gen = self.discriminator_monet(monet_from_real, training=fl)
            dis_res_monet = self.discriminator_monet(x_monet, training=fl)
            res_cycle_real = self.generator_monet_to_real(monet_from_real, training=fl)
            loss_cycle_real = self.MAE(x_real, res_cycle_real) * self.lambda_cycle
            loss_dis_real_gen, loss_dis_monet = self.adversarial_loss(dis_res_monet, dis_res_monet_gen)

            identity_gen_monet = self.generator_monet_to_real(x_real, training=fl)
            identity_gen_real = self.generator_real_to_monet(x_monet, training=fl)
            identity_monet_loss = self.MAE(x_real, identity_gen_monet * self.lambda_cycle * self.lambda_identity)
            identity_real_loss = self.MAE(x_monet, identity_gen_real * self.lambda_cycle * self.lambda_identity, )

            total_loss_gen_monet = loss_dis_monet_gen + loss_cycle_monet + identity_monet_loss
            total_loss_gen_real = loss_dis_real_gen + loss_cycle_real + identity_real_loss

        grad_total_monet_gen = tape_monet.gradient(total_loss_gen_monet, self.generator_monet_to_real.trainable_weights)
        dis_grad_real = tape_monet.gradient(loss_dis_real, self.discriminator_real.trainable_weights)
        # real_from_gen_monet_grad = tape_monet.gradient(res_cycle_monet, self.generator_real_to_monet.trainable_weights)

        grad_total_real_gen = tape_real.gradient(total_loss_gen_real, self.generator_real_to_monet.trainable_weights)
        dis_grad_monet = tape_real.gradient(loss_dis_monet, self.discriminator_monet.trainable_weights)
        # monet_from_gen_real_grad = tape_real.gradient(res_cycle_real, self.generator_monet_to_real.trainable_weights)

        self.generator_optimizer_fake.apply_gradients(
            zip(grad_total_monet_gen, self.generator_monet_to_real.trainable_weights))
        self.discriminator_optimizer_real.apply_gradients(zip(dis_grad_real, self.discriminator_real.trainable_weights))
        # self.generator_optimizer_real.apply_gradients(real_from_gen_monet_grad,self.generator_monet_to_real.trainable_weights)

        self.generator_optimizer_real.apply_gradients(
            zip(grad_total_real_gen, self.generator_real_to_monet.trainable_weights))
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
        stat["cycle_money"] = loss_cycle_monet
        stat["cycle_real"] = loss_cycle_real
        stat["loss"] = (total_loss_gen_real + total_loss_gen_monet) * 0.5
        stat["total_loss_gen_real"] = total_loss_gen_real
        stat["total_loss_gen_monet"] = total_loss_gen_monet

        return stat

    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=5, interval=2):

        epoch = epoch + self.__bias_plot if epoch is not None else epoch
        num_cols = self._plot.shape[0]

        generated_images = self.generator_real_to_monet.predict(self._plot) / 255.0
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if row == 0:
                    plt.imshow(self._plot[col] / 255.0)
                else:
                    plt.imshow(generated_images[col])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'{self.path}image_at_epoch{epoch}.png')
        plt.show()

        if bool(epoch) and epoch % self.save_iterval == 0:
            self.discriminator_monet.save(f"{self.path}discriminator_monet ep{epoch}")
            self.discriminator_real.save(f"{self.path}discriminator_real ep{epoch}")
            self.generator_real_to_monet.save(f"{self.path}generator_real_to_monet ep{epoch}")
            self.generator_monet_to_real.save(f"{self.path}generator_monet_to_real ep{epoch}")


if __name__ == '__main__':
    import data.extract_data as ed

    data = ed.Data(path="data/", batch_size=1)
    _, x_plot = data[0]  # 200mb
    x_plot = x_plot[:5]
    model = CycleGAN(x_plot)  # 1gb
    model.plot_images()
    model.compile()
    model.fit(data, epochs=10, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)])
    # model.fit(data,epochs=10,callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)])
