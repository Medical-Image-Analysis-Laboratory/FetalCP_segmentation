# Author: Priscille de Dumast
# Date: 15.08.2022

"""Modules for fetal CP segmentation inference.
"""

import os
import numpy as np
import SimpleITK as sitk

import logging
import tensorflow as tf

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import traits, TraitedSpec, File, BaseInterface, BaseInterfaceInputSpec

from topofetal.utils import DataManager
from topofetal.utils import network_utils

logging.getLogger('tensorflow').disabled = True


class SegmenterInputSpec(BaseInterfaceInputSpec):
    """Class used for the input of CP segmentation inference.
    """
    input_image = File(desc='Input image to segment.', mandatory=True)

    input_extraction_axis = traits.Int(-1, desc='Extraction axis', usedefault=True, mandatory=False)


    input_network_dir = traits.Directory(desc='Input network directory', mandatory=True)
    input_network_name = traits.Str(desc='Input network name', mandatory=True)
    input_fold = traits.Int(0, desc='Fold of network to infer', mandatory=True)


class SegmenterOutputSpec(TraitedSpec):
    """Class used for the output of CP segmentation inference.
    """
    output_labelmap = File(desc='Output binary label map.')
    output_likelihood = File(desc='Output likelihood map.')

class Segmenter(BaseInterface):
    """Class used for the inference of fetal CP segmentation.
    """

    input_spec = SegmenterInputSpec
    output_spec = SegmenterOutputSpec

    m_model = None
    m_dm = None

    def generate_network_path(self):
        return os.path.join(self.inputs.input_network_dir,
                            self.inputs.input_network_name,
                            self.inputs.input_network_name + '_' + str(self.inputs.input_fold),
                            self.inputs.input_network_name + '_' + str(self.inputs.input_fold))

    def _gen_filename(self, name):
        suf = '_' + self.inputs.input_network_name + '_' + str(self.inputs.input_fold)
        _, bname, ext = split_filename(self.inputs.input_image)
        if name == 'output_labelmap':
            output = bname + suf + ext
            return os.path.abspath(output)
        if name == 'output_likelihood':
            output = bname + suf + '_votes' + ext
            return os.path.abspath(output)
        return None

    def _eval(self):
        ax_to_swap = self.inputs.input_extraction_axis if self.inputs.input_extraction_axis != -1 else self.m_dm.m_extraction_axis

        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(self.inputs.input_image)
        image_sitk = reader.Execute()

        image = sitk.GetArrayFromImage(image_sitk)
        image = np.swapaxes(image, 0, ax_to_swap)

        image = np.pad(image, ((self.m_dm.m_patch_size, self.m_dm.m_patch_size),
                               (self.m_dm.m_patch_size, self.m_dm.m_patch_size),
                               (self.m_dm.m_patch_size, self.m_dm.m_patch_size)),
                       'constant', constant_values=0)

        image_patches = self.m_dm.FeTA_extract_patches_3d_to_2d(volume=image).squeeze(axis=1)
        image_patches = image_patches.astype(np.float32)
        image_patches = np.expand_dims(image_patches, axis=-1)

        intensities_thd = 100
        ii_non_empty_patches = np.where(np.sum(image_patches > intensities_thd, axis=(1, 2)) > 10)[0]

        image_patches = image_patches.astype(np.int16)

        pred = np.zeros((image_patches.shape[0], self.m_dm.m_patch_size, self.m_dm.m_patch_size, self.m_dm.m_n_classes))

        data_to_predict = tf.data.Dataset \
            .from_tensor_slices(image_patches[ii_non_empty_patches]) \
            .map(self.m_dm.cast_ds_float32_int16)\
            .map(lambda x: tf.image.per_image_standardization(x))

        tf.get_logger().setLevel('INFO')
        pred[ii_non_empty_patches] = self.m_model.predict(data_to_predict.batch(32))

        pred_sub_recon, vote_sub_recon = self.m_dm.reconstruct_volume(patches=pred, expected_shape=image.shape)

        pred_sub_recon = np.where( image > 0, pred_sub_recon, 0)
        pred_recon = pred_sub_recon[
                     self.m_dm.m_patch_size:-self.m_dm.m_patch_size,
                     self.m_dm.m_patch_size:-self.m_dm.m_patch_size,
                     self.m_dm.m_patch_size:-self.m_dm.m_patch_size
                     ]
        pred_recon = np.swapaxes(pred_recon, 0, ax_to_swap)

        out = sitk.GetImageFromArray(pred_recon)
        out.CopyInformation(image_sitk)
        out = sitk.Cast(out, sitk.sitkUInt8)
        writer.SetFileName(self._gen_filename('output_labelmap'))
        writer.Execute(out)

        for d in range(vote_sub_recon.shape[-1]):
            vote_sub_recon[...,d] = np.where( image > 0, vote_sub_recon[...,d], 0)
        vote_recon = vote_sub_recon[
                     self.m_dm.m_patch_size:-self.m_dm.m_patch_size,
                     self.m_dm.m_patch_size:-self.m_dm.m_patch_size,
                     self.m_dm.m_patch_size:-self.m_dm.m_patch_size
                     ]
        vote_recon = np.swapaxes(vote_recon, 0, ax_to_swap)

        out = sitk.GetImageFromArray(vote_recon)
        out.CopyInformation(image_sitk)
        writer.SetFileName(self._gen_filename('output_likelihood'))
        writer.Execute(out)

        return


    def _run_interface(self, runtime):

        tf.get_logger().setLevel('INFO')
        self.m_dm = DataManager.DataManager( p_patch_size=64,
                                         p_extraction_step=16,
                                         p_extraction_axis=2,
                                         p_n_classes=2)

        self.m_model = network_utils.get_model(self.m_dm, lambda_topoloss=0)
        self.m_model.load_weights( self.generate_network_path() )

        try:
            self._eval()
        except Exception as e:
            print('Failed running Segmenter._eval() from Segmenter._run_interface()')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_labelmap'] = os.path.abspath( self._gen_filename('output_labelmap') )
        outputs['output_likelihood'] = os.path.abspath( self._gen_filename('output_likelihood') )
        return outputs
