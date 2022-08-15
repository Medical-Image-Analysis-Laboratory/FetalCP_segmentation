# Author: Priscille de Dumast
# Date: 15.08.2022

"""Modules for fetal CP segmentation preprocessing
"""

import os

import SimpleITK as sitk
import numpy as np

from nipype.interfaces.base import traits, TraitedSpec, File, BaseInterface, BaseInterfaceInputSpec


class ConvertLabelmapInputSpec(BaseInterfaceInputSpec):
    """Class used for the input of ConvertLabelmap.
    """

    in_map = File(desc='Input image filename', mandatory=True)
    correspondance = traits.Dict(mandatory=True, desc = 'Correspondance between in_labels (key) and out_labels (value)')


class ConvertLabelmapOutputSpec(TraitedSpec):
    """Class used for the output of ConvertLabelmap.
    """
    out_map = File(desc='Output label map.')


class ConvertLabelmap(BaseInterface):
    """Class used for to convert labels in tissue segmentation
    """

    input_spec = ConvertLabelmapInputSpec
    output_spec = ConvertLabelmapOutputSpec

    def _gen_filename(self):
        return os.path.abspath(os.path.basename(self.inputs.in_map))

    def _convert_labels(self, in_array):
        out_array = np.zeros(in_array.shape, dtype=in_array.dtype)

        for in_label, out_label in self.inputs.correspondance.items():
            out_array = np.where(in_array == in_label, out_label, out_array)

        return out_array

    def _run_interface(self, runtime):
        try:
            reader = sitk.ImageFileReader()
            writer = sitk.ImageFileWriter()

            reader.SetFileName(self.inputs.in_map)
            in_map_sitk = reader.Execute()
            in_array = sitk.GetArrayFromImage(in_map_sitk)

            out_array = self._convert_labels(in_array)

            out_map_sitk = sitk.GetImageFromArray(out_array)
            out_map_sitk.CopyInformation(in_map_sitk)

            writer.SetFileName(self._gen_filename())
            out_map_sitk = sitk.Cast(out_map_sitk, sitk.sitkUInt8)
            writer.Execute(out_map_sitk)

        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_map'] = os.path.abspath( self._gen_filename() )
        return outputs


class ExtractLabelInputSpec(BaseInterfaceInputSpec):
    """Class used for the input of ExtractLabel.
    """
    in_map = File(desc='Input image filename', mandatory=True)
    label = traits.Int(1, usedefault=True)


class ExtractLabelOutputSpec(TraitedSpec):
    """Class used for the output of ExtractLabel.
    """
    out_map = File(desc='Output label map.')


class ExtractLabel(BaseInterface):
    """Class used to extract a specific label out of a multi-label map.
    """
    input_spec = ExtractLabelInputSpec
    output_spec = ExtractLabelOutputSpec

    def _gen_filename(self):
        return os.path.abspath(os.path.basename(self.inputs.in_map))

    def _label_filtration(self):
        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(self.inputs.in_map)

        map_sitk = reader.Execute()

        binarizer = sitk.BinaryThresholdImageFilter()
        binarizer.SetLowerThreshold(self.inputs.label)
        binarizer.SetUpperThreshold(self.inputs.label)

        map_sitk = binarizer.Execute(map_sitk)

        writer.SetFileName(self._gen_filename())
        map_sitk = sitk.Cast(map_sitk, sitk.sitkUInt8)
        writer.Execute(map_sitk)

        return

    def _run_interface(self, runtime):

        try:
            self._label_filtration()
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_map'] = os.path.abspath(self._gen_filename())
        return outputs


class BinaryThresholdImageInputSpec(BaseInterfaceInputSpec):
    """Class used for the input of BinaryThresholdImage.
    """
    in_map = File(desc='Input image filename', mandatory=True)
    threshold = traits.Float(0.1, usedefault=True)


class BinaryThresholdImageOutputSpec(TraitedSpec):
    """Class used for the output of BinaryThresholdImage.
    """
    out_map = File(desc='Output label map.')


class BinaryThresholdImage(BaseInterface):
    """Class used to perform a binary threshold of an image.
    """
    input_spec = BinaryThresholdImageInputSpec
    output_spec = BinaryThresholdImageOutputSpec

    def _gen_filename(self):
        return os.path.abspath(os.path.basename(self.inputs.in_map))

    def _label_filtration(self):
        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(self.inputs.in_map)

        map_sitk = reader.Execute()
        map_sitk = sitk.Cast(map_sitk, sitk.sitkUInt8)

        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(map_sitk)

        binarizer = sitk.BinaryThresholdImageFilter()
        binarizer.SetLowerThreshold(self.inputs.threshold)
        binarizer.SetUpperThreshold(minmax.GetMaximum()+1)

        map_sitk = binarizer.Execute(map_sitk)

        writer.SetFileName(self._gen_filename())
        map_sitk = sitk.Cast(map_sitk, sitk.sitkUInt8)
        writer.Execute(map_sitk)

        return

    def _run_interface(self, runtime):

        try:
            self._label_filtration()
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_map'] = os.path.abspath(self._gen_filename())
        return outputs


class MaskImageInputSpec(BaseInterfaceInputSpec):
    """Class used for the input of MaskImage.
    """
    in_image = File(desc='Input image path', mandatory=True)
    in_mask = File(desc='Input mask path', mandatory=True)

class MaskImageOutputSpec(TraitedSpec):
    """Class used for the output of MaskImage.
    """
    out_image = File(desc='Output masked image.')

class MaskImage(BaseInterface):
    """Class used to mask an image by another.
    """
    input_spec = MaskImageInputSpec
    output_spec = MaskImageOutputSpec

    def _gen_filename(self):
        return os.path.abspath(os.path.basename(self.inputs.in_image))

    def _run_interface(self, runtime):
        try:
            reader = sitk.ImageFileReader()
            writer = sitk.ImageFileWriter()

            reader.SetFileName(self.inputs.in_image)
            image_sitk = reader.Execute()

            reader.SetFileName(self.inputs.in_mask)
            mask_sitk = reader.Execute()
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

            mask_image = sitk.MaskImageFilter()
            image_sitk = mask_image.Execute(image_sitk, mask_sitk)

            writer.SetFileName(self._gen_filename())
            writer.Execute(image_sitk)

        except Exception as e:
            print('Failed')
            print(e)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_image'] = os.path.abspath(self._gen_filename())
        return outputs
