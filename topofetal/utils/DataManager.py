# Author: Priscille de Dumast
# Date: 15.08.2022

import numpy as np
import nibabel as nib

import itertools
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches

from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import torch
import torchio as tio

np.random.seed(1312)
tf.random.set_seed(1312)


class DataManager():
    def __init__(self,
                 p_patch_size = 64,
                 p_extraction_step = 64,
                 p_extraction_axis = [2],
                 p_n_classes = 2,
                 p_n_channels = 1):

        self.m_patch_size = p_patch_size
        self.m_extraction_step = p_extraction_step
        self.m_extraction_axis = p_extraction_axis

        self.m_n_classes = p_n_classes
        self.m_n_channels = p_n_channels

        self.m_patch_shape = (1, self.m_patch_size, self.m_patch_size)
        self.m_extraction_shape = (1, self.m_extraction_step, self.m_extraction_step)

    #
    ## Patches to images
    def reconstruct_volume(self, patches, expected_shape):
        rec_volume, vote_volume = self.perform_voting(patches, expected_shape)
        return rec_volume, vote_volume

    def perform_voting(self, patches, expected_shape):
        output_shape = self.m_patch_shape
        vote_img = np.zeros(expected_shape + (self.m_n_classes,))
        coordinates = self.generate_indexes( output_shape, expected_shape)

        for count, coord in enumerate(coordinates):
            selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
            selection += [slice(None)]
            vote_img[selection] += patches[count]

        return np.argmax(vote_img, axis=3), vote_img

    def generate_indexes(self, p_output_shape, p_expected_shape):
        ndims = len(p_output_shape)
        poss_shape = [
            p_output_shape[i] + self.m_extraction_shape[i] * ((p_expected_shape[i] - p_output_shape[i]) // self.m_extraction_shape[i]) for i
            in range(ndims)]
        idxs = [range(p_output_shape[i], poss_shape[i] + 1, self.m_extraction_shape[i]) for i in range(ndims)]
        return itertools.product(*idxs)


    def FeTA_extract_patches_3d_to_2d(self, volume):

        patches = sk_extract_patches(volume, patch_shape=self.m_patch_shape, extraction_step=self.m_extraction_shape)

        ndim = len(volume.shape)
        npatches = np.prod(patches.shape[:ndim])

        return patches.reshape((npatches,) + self.m_patch_shape)


    def FeTA_load_subject_patches(self, path_t2w, path_dseg):

        if not isinstance(path_t2w, str):
            path_t2w = bytes.decode(path_t2w.numpy())
            path_dseg = bytes.decode(path_dseg.numpy())

        volume_nii = nib.load(path_t2w)
        image_np = np.asanyarray(volume_nii.dataobj)

        volume_nii = nib.load(path_dseg)
        dseg_np = np.asanyarray(volume_nii.dataobj)

        image_np = image_np * (1 * (dseg_np > 0))

        image_np = np.pad(image_np, ((self.m_patch_size, self.m_patch_size), (self.m_patch_size, self.m_patch_size),
                                     (self.m_patch_size, self.m_patch_size)), 'constant', constant_values=0)
        dseg_np = np.pad(dseg_np, ((self.m_patch_size, self.m_patch_size), (self.m_patch_size, self.m_patch_size),
                                   (self.m_patch_size, self.m_patch_size)), 'constant', constant_values=0)

        images = np.zeros((0, self.m_patch_size, self.m_patch_size), dtype=np.float32)
        labels = np.zeros((0, self.m_patch_size, self.m_patch_size), dtype=np.int16)

        for i, ax in enumerate(self.m_extraction_axis):

            image_np_sp = np.swapaxes(image_np, 0, ax)
            image_patches = self.FeTA_extract_patches_3d_to_2d(volume=image_np_sp).squeeze(axis=1)
            image_patches = np.float32(image_patches)

            dseg_np_sp = np.swapaxes(dseg_np, 0, ax)
            dseg_patches = self.FeTA_extract_patches_3d_to_2d(volume=dseg_np_sp).squeeze(axis=1)
            dseg_patches = np.int16(dseg_patches)

            ## Remove empty patches, with almost no MR contrast
            ind_non_empty_patches = np.where(np.max(image_patches, axis=(1, 2)) > 1)[0]
            image_patches = image_patches[ind_non_empty_patches]
            dseg_patches = dseg_patches[ind_non_empty_patches]

            images = np.concatenate((images, image_patches), axis=0)
            labels = np.concatenate((labels, dseg_patches), axis=0)

        # Label correspondance between FeTA1Corrected and original FeTA1
        labels_b = np.zeros(labels.shape)
        labels = np.where(labels > 8, 0, labels)

        if self.m_n_classes == 2: labels_b = 1 * (labels == 2)

        labels_b = to_categorical(labels_b, num_classes=self.m_n_classes)
        images = np.expand_dims(images, -1)

        images = np.float32(images)
        labels_b = np.int16(labels_b)

        # print(os.path.basename(path_t2w),"Extracted images : images.shape  ", images.shape, "  ||  labels.shape ", labels_b.shape)
        return images, labels_b


    def tf_load_data(self, image, label):
        [image, label] = tf.py_function(self.FeTA_load_subject_patches, inp=[image, label], Tout=[tf.float32, tf.int16])
        image.set_shape((None, self.m_patch_size, self.m_patch_size, self.m_n_channels))
        label.set_shape((None, self.m_patch_size, self.m_patch_size, self.m_n_classes))
        return [image, label]

    def tf_gaussian_noise(self, image, label=None):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1, dtype=tf.float32)
        image = tf.math.add(image, noise)
        if label is None:
            return image
        return image, label

    def filter_cortex_patches_ds(self, x, y):
        return tf.math.greater(tf.reduce_sum(y[:, :, 1]), tf.cast(100, tf.int16))

    def cast_ds_int16(self, x, y=None):
        if y is None:
            return tf.cast(x, tf.int16)
        return tf.cast(x, tf.int16), tf.cast(y, tf.int16)

    def cast_ds_float32_int16(self, x, y=None):
        if y is None:
            return tf.cast(x, tf.float32)
        return tf.cast(x, tf.float32), tf.cast(y, tf.int16)


    #
    # - Augmentation
    #

    def random_augment_image(self, image, label):

        im = torch.from_numpy(np.expand_dims(image.numpy(), axis=0))
        lab = torch.from_numpy(np.expand_dims(np.expand_dims(label[:,:,1].numpy(), axis=-1), axis=0))
        patch_torch = tio.ScalarImage(tensor=im)
        labels_torch = tio.LabelMap(tensor=lab)
        subject_torchio = tio.Subject(t2w=patch_torch, dseg=labels_torch)

        bias_probability = 0.2
        blur_probability = 0.2
        gamma_probability = 0.2
        noise_probability = 0.2
        elastic_deformation_probability = 0.2
        flip_probability = 0.5

        random_bias = tio.RandomBiasField( p = bias_probability,
                                           coefficients = 0.5,
                                           order = 2)
        subject_torchio = random_bias(subject_torchio)


        random_blur = tio.RandomBlur( p = blur_probability,
                                      std = (0.5,0.5))
        subject_torchio = random_blur(subject_torchio)


        random_noise = tio.RandomNoise( p = noise_probability,
                                        mean = 0,
                                        std = 0.05)
        subject_torchio = random_noise(subject_torchio)


        random_gamma = tio.RandomGamma( p = gamma_probability,
                                        log_gamma = (-0.3, 0.3))
        subject_torchio = random_gamma(subject_torchio)


        random_elastic_deformation = tio.RandomElasticDeformation(p = elastic_deformation_probability,
                                                                  num_control_points = 5,
                                                                  max_displacement = 7,
                                                                  locked_borders = 2,
                                                                  image_interpolation = 'linear',
                                                                  label_interpolation = 'nearest')
        subject_torchio = random_elastic_deformation(subject_torchio)


        random_flip = tio.RandomFlip(flip_probability = flip_probability,
                                     axes = (0,1))
        subject_torchio = random_flip(subject_torchio)


        im = subject_torchio.t2w.numpy()[0]
        lab_tmp = subject_torchio.dseg.numpy()[0]
        lab = np.stack((1-lab_tmp[:,:,0], lab_tmp[:,:,0]), axis=-1)
        return im, lab


    def tf_random_augment_image(self, image, label):
        im_shape = image.shape
        label_shape = label.shape
        [image, label] = tf.py_function(self.random_augment_image, inp=[image, label], Tout=[tf.float32, tf.int16])
        image.set_shape(im_shape)
        label.set_shape(label_shape)
        return image, label
