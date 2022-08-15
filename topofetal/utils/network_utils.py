# Author: Priscille de Dumast
# Date: 15.08.2022

from utils import UnetModel

import tensorflow as tf
from tensorflow.keras import optimizers

def get_model(p_dm, lambda_topoloss=0, lambda_hybrid = 0, min_pers_th=0):
    m = UnetModel.UnetModel(p_num_classes=p_dm.m_n_classes,
                            p_patch_size=p_dm.m_patch_size,
                            p_num_channels=p_dm.m_n_channels,
                            p_lambda_topoloss = lambda_topoloss,
                            p_lambda_hybrid = lambda_hybrid,
                            p_min_pers_th=min_pers_th)
    return m

def compile_model(p_model, p_lr_init):
    p_model.compile(optimizer=optimizers.Adam(learning_rate=p_lr_init),
                    metrics=['accuracy', dice_coeff_multilabel])
    return p_model


@tf.function
def dice_coeff(y_true, y_pred, smooth=1.):
    """This function computes the soft dice coefficient between the predicted mask and the ground truth mask
    Args:
        y_true (tf.Tensor): ground truth mask
        y_pred (tf.Tensor): predicted mask
        smooth (float): value added for numerical stability (avoid division by 0)
    Returns:
        dice_coefficient (tf.Tensor): dice coefficient
    """
    # flatten vectors and cast to float32
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))
    dice_coefficient = (2. * intersection + smooth) / (union + smooth)
    return dice_coefficient

@tf.function
def dice_coeff_multilabel(y_true, y_pred, p_num_classes = 2, p_weights=None):
    dice = 0
    for index in range(1, p_num_classes):
        if p_weights is None:
            dice += dice_coeff(y_true[:, :, index], y_pred[:, :, index])
        else:
            dice += p_weights[index] * dice_coeff(y_true[:, :, index], y_pred[:, :, index])

    return dice / (p_num_classes - 1)  # taking average

@tf.function
def dice_loss(y_true, y_pred, p_num_classes = 2, p_weights=None):
    """This function computes the dice loss as 1-dsc_coeff
    Args:
        y_true (tf.Tensor): ground truth mask
        y_pred (tf.Tensor): predicted mask
    Returns:
        dsc_loss (tf.Tensor): dice loss
    """
    dsc_loss = 1 - dice_coeff_multilabel(y_true, y_pred, p_num_classes = p_num_classes, p_weights=p_weights)
    return dsc_loss



@tf.function
def bce_dice_loss(y_true, y_pred, p_lambda_hybrid=0, p_num_classes=2, p_weights=None):
    """This function combines the binary cross-entropy loss with the Dice loss into one unique hybrid loss
    Args:
        y_true (tf.Tensor): label volume
        y_pred (tf.Tensor): prediction volume
        loss_lambda (float): value to balance/weight the two terms of the loss
    Returns:
            hybrid_loss (tf.Tensor): sum of the two losses
    """

    ce = tf.nn.weighted_cross_entropy_with_logits( tf.cast(y_true, tf.float32), y_pred, pos_weight=9, name='weighted_cross_entropy_with_logits')
    ce_loss = tf.reduce_mean(ce)  # reduce the result to get the final loss

    dsc_loss = dice_loss(y_true, y_pred, p_num_classes = p_num_classes, p_weights=p_weights)

    hybrid_loss = (1-p_lambda_hybrid) * ce_loss + p_lambda_hybrid *  dsc_loss

    return hybrid_loss
