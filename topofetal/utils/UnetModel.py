
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from utils import network_utils

from utils import topoloss as topoloss


loss_tracker = keras.metrics.Mean(name="loss")
val_loss_tracker = keras.metrics.Mean(name="val_loss")

ce_tracker = keras.metrics.Mean(name="ce")
val_ce_tracker = keras.metrics.Mean(name="val_ce")

topo_tracker = keras.metrics.Mean(name="topo")
val_topo_tracker = keras.metrics.Mean(name="val_topo")

topo_0d_tracker = keras.metrics.Mean(name="topo_0d")
val_topo_0d_tracker = keras.metrics.Mean(name="val_topo_0d")

topo_1d_tracker = keras.metrics.Mean(name="topo_1d")
val_topo_1d_tracker = keras.metrics.Mean(name="val_topo_1d")

# tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float32')


class UnetModel(keras.Model):

    def __init__(
        self,
        p_patch_size=64,
        p_num_channels=1,
        p_num_classes=2,
        p_lambda_topoloss=0,
        p_lambda_hybrid=0,
        p_min_pers_th=0,
        name="Unetmodel",
        **kwargs
    ):
        super(UnetModel, self).__init__(name=name, **kwargs)
        self.m_patch_size = p_patch_size
        self.m_num_channels = p_num_channels
        self.m_num_classes = p_num_classes
        self.m_lambda_hybrid = p_lambda_hybrid
        self.m_lambda_topoloss = p_lambda_topoloss
        self.m_min_pers_th = p_min_pers_th

        initializer = tf.keras.initializers.HeNormal()

        self.conv1_1 = layers.Conv2D(name="conv1_1", filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn1_1= layers.BatchNormalization(name="bn1_1")
        self.act1_1= layers.Activation(keras.activations.relu)
        self.conv1_2 = layers.Conv2D(name="conv1_2", filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn1_2= layers.BatchNormalization(name="bn1_2")
        self.act1_2= layers.Activation(keras.activations.relu)
        self.pool1 = layers.MaxPool2D(name="pool1", pool_size=2)

        self.conv2_1 = layers.Conv2D(name="conv2_1", filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn2_1= layers.BatchNormalization(name="bn2_1")
        self.act2_1= layers.Activation(keras.activations.relu)
        self.conv2_2 = layers.Conv2D(name="conv2_2", filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn2_2= layers.BatchNormalization(name="bn2_2")
        self.act2_2= layers.Activation(keras.activations.relu)
        self.pool2 = layers.MaxPool2D(name="pool2", pool_size=2)

        self.conv3_1 = layers.Conv2D(name="conv3_1", filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn3_1= layers.BatchNormalization(name="bn3_1")
        self.act3_1= layers.Activation(keras.activations.relu)
        self.conv3_2 = layers.Conv2D(name="conv3_2", filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn3_2= layers.BatchNormalization(name="bn3_2")
        self.act3_2= layers.Activation(keras.activations.relu)
        self.pool3 = layers.MaxPool2D(name="pool3", pool_size=2)

        self.conv4_1 = layers.Conv2D(name="conv4_1", filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn4_1= layers.BatchNormalization(name="bn4_1")
        self.act4_1= layers.Activation(keras.activations.relu)
        self.conv4_2 = layers.Conv2D(name="conv4_2", filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn4_2= layers.BatchNormalization(name="bn4_2")
        self.act4_2= layers.Activation(keras.activations.relu)
        self.pool4 = layers.MaxPool2D(name="pool4", pool_size=2)

        self.conv5_1 = layers.Conv2D(name="conv5_1", filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn5_1= layers.BatchNormalization(name="bn5_1")
        self.act5_1= layers.Activation(keras.activations.relu)
        self.conv5_2 = layers.Conv2D(name="conv5_2", filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn5_2= layers.BatchNormalization(name="bn5_2")
        self.act5_2= layers.Activation(keras.activations.relu)

        self.up6 = layers.UpSampling2D(name="up6", size=(2, 2))
        self.merge6 = layers.Concatenate(name="merge6", axis=-1)
        self.conv6_1 = layers.Conv2D(name="conv6_1", filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn6_1= layers.BatchNormalization(name="bn6_1")
        self.act6_1= layers.Activation(keras.activations.relu)
        self.conv6_2 = layers.Conv2D(name="conv6_2", filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn6_2= layers.BatchNormalization(name="bn6_2")
        self.act6_2= layers.Activation(keras.activations.relu)

        self.up7 = layers.UpSampling2D(name="up7", size=(2, 2))
        self.merge7 = layers.Concatenate(name="merge7", axis=-1)
        self.conv7_1 = layers.Conv2D(name="conv7_1", filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn7_1= layers.BatchNormalization(name="bn7_1")
        self.act7_1= layers.Activation(keras.activations.relu)
        self.conv7_2 = layers.Conv2D(name="conv7_2", filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn7_2= layers.BatchNormalization(name="bn7_2")
        self.act7_2= layers.Activation(keras.activations.relu)

        self.up8 = layers.UpSampling2D(name="up8", size=(2, 2))
        self.merge8 = layers.Concatenate(name="merge8", axis=-1)
        self.conv8_1 = layers.Conv2D(name="conv8_1", filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn8_1= layers.BatchNormalization(name="bn8_1")
        self.act8_1= layers.Activation(keras.activations.relu)
        self.conv8_2 = layers.Conv2D(name="conv8_2", filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn8_2= layers.BatchNormalization(name="bn8_2")
        self.act8_2= layers.Activation(keras.activations.relu)

        self.up9 = layers.UpSampling2D(name="up9", size=(2, 2))
        self.merge9 = layers.Concatenate(name="merge9", axis=-1)
        self.conv9_1 = layers.Conv2D(name="conv9_1", filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn9_1= layers.BatchNormalization(name="bn9_1")
        self.act9_1= layers.Activation(keras.activations.relu)
        self.conv9_2 = layers.Conv2D(name="conv9_2", filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer)
        self.bn9_2= layers.BatchNormalization(name="bn9_2")
        self.act9_2= layers.Activation(keras.activations.relu)

        self.pred = layers.Conv2D(name="pred", filters=self.m_num_classes, kernel_size=1, activation='softmax', padding='same')

    @tf.function
    def call(self, inputs):
        res_conv1 = self.act1_2(self.bn1_2(self.conv1_2(self.act1_1(self.bn1_1(self.conv1_1(tf.cast(inputs, dtype=tf.float32)))))))
        res_pool1 = self.pool1(res_conv1)

        res_conv2 = self.act2_2(self.bn2_2(self.conv2_2(self.act2_1(self.bn2_1(self.conv2_1(res_pool1))))))
        res_pool2 = self.pool2(res_conv2)

        res_conv3 = self.act3_2(self.bn3_2(self.conv3_2(self.act3_1(self.bn3_1(self.conv3_1(res_pool2))))))
        res_pool3 = self.pool3(res_conv3)

        res_conv4 = self.act4_2(self.bn4_2(self.conv4_2(self.act4_1(self.bn4_1(self.conv4_1(res_pool3))))))
        res_pool4 = self.pool4(res_conv4)

        res_conv5 = self.act5_2(self.bn5_2(self.conv5_2(self.act5_1(self.bn5_1(self.conv5_1(res_pool4))))))

        res_up6 = self.up6(res_conv5)
        res_merg6 = self.merge6([res_up6, res_conv4])
        res_conv6 = self.act6_2(self.bn6_2(self.conv6_2(self.act6_1(self.bn6_1(self.conv6_1(res_merg6))))))

        res_up7 = self.up7(res_conv6)
        res_merg7 = self.merge7([res_up7, res_conv3])
        res_conv7 = self.act7_2(self.bn7_2(self.conv7_2(self.act7_1(self.bn7_1(self.conv7_1(res_merg7))))))

        res_up8 = self.up8(res_conv7)
        res_merg8 = self.merge8([res_up8, res_conv2])
        res_conv8 = self.act8_2(self.bn8_2(self.conv8_2(self.act8_1(self.bn8_1(self.conv8_1(res_merg8))))))

        res_up9 = self.up9(res_conv8)
        res_merg9 = self.merge9([res_up9, res_conv1])
        res_conv9 = self.act9_2(self.bn9_2(self.conv9_2(self.act9_1(self.bn9_1(self.conv9_1(res_merg9))))))

        predictions = self.pred(res_conv9)

        return predictions


    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            hybrid = network_utils.bce_dice_loss(y, y_pred, p_lambda_hybrid=self.m_lambda_hybrid, p_num_classes = self.m_num_classes)


            if self.m_lambda_topoloss > 0:
                topo_0d, topo_1d = tf.numpy_function(self.getTopoLoss_map,
                                                     inp=[y, y_pred, self.m_min_pers_th],
                                                     Tout=[tf.float32, tf.float32])
                topo = tf.cast(tf.add(topo_0d, topo_1d), dtype=tf.float32)

                loss = tf.add(tf.cast(tf.scalar_mul((1-self.m_lambda_topoloss), hybrid), dtype=tf.float32),
                                  tf.cast(tf.scalar_mul(self.m_lambda_topoloss, topo), dtype=tf.float32))
            else:
                loss = hybrid

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        ce_tracker.update_state(hybrid)
        if self.m_lambda_topoloss > 0:
            topo_tracker.update_state(topo)
            topo_0d_tracker.update_state(topo_0d)
            topo_1d_tracker.update_state(topo_1d)

        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        d1 = {"loss": loss_tracker.result()}
        d1["bce"]= ce_tracker.result()
        if self.m_lambda_topoloss > 0:
            d1["topo"]= topo_tracker.result()
            d1["topo_0d"]= topo_0d_tracker.result()
            d1["topo_1d"]= topo_1d_tracker.result()
        d2 = {m.name: m.result() for m in self.metrics}
        d1.update(d2)

        return d1

    @tf.function
    def test_step(self, data):
        x, y = data

        y_pred = self(x, training=False)  # Forward pass

        val_hybrid = network_utils.bce_dice_loss(y, y_pred, p_lambda_hybrid=self.m_lambda_hybrid, p_num_classes = self.m_num_classes)

        if self.m_lambda_topoloss > 0:
            val_topo_0d, val_topo_1d = tf.numpy_function(self.getTopoLoss_map,
                                                         inp=[y, y_pred, self.m_min_pers_th],
                                                         Tout=[tf.float32, tf.float32])
            val_topo = tf.cast(tf.add(val_topo_0d, val_topo_1d), dtype=tf.float32)

            val_loss = tf.add(tf.cast(tf.scalar_mul((1-self.m_lambda_topoloss), val_hybrid), dtype=tf.float32),
                              tf.cast(tf.scalar_mul(self.m_lambda_topoloss, val_topo), dtype=tf.float32))
        else:
            val_topo_0d, val_topo_1d = tf.numpy_function(self.getTopoLoss_map,
                                                         inp=[y, y_pred, self.m_min_pers_th],
                                                         Tout=[tf.float32, tf.float32])
            val_topo = tf.cast(tf.add(val_topo_0d, val_topo_1d), dtype=tf.float32)

            val_loss = val_hybrid

        val_loss_tracker.update_state(val_loss)
        val_ce_tracker.update_state(val_hybrid)

        val_topo_tracker.update_state(val_topo)
        val_topo_0d_tracker.update_state(val_topo_0d)
        val_topo_1d_tracker.update_state(val_topo_1d)

        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        d1 = {"loss": val_loss_tracker.result()}
        d1["bce"] = val_ce_tracker.result()
        d1["topo"] = val_topo_tracker.result()
        d1["topo_0d"] = val_topo_0d_tracker.result()
        d1["topo_1d"] = val_topo_1d_tracker.result()

        d2 = {m.name: m.result() for m in self.metrics}
        d1.update(d2)

        return d1


    ###########################################################################
    ###########################################################################
    ###########################################################################

    # @tf.function
    def getTopoLoss_map(self, gt, likelihood, min_pers_th):
        losses_per_patch_0d, losses_per_patch_1d = [], []
        for ii in range(gt.shape[0]):
            loss_0d, loss_1d = topoloss.getTopoLoss(likelihood[ii], gt[ii], min_pers_th)
            losses_per_patch_0d.append(loss_0d)
            losses_per_patch_1d.append(loss_1d)

        return [tf.cast(tf.reduce_mean(losses_per_patch_0d), tf.float32), tf.cast(tf.reduce_mean(losses_per_patch_1d), tf.float32)]
