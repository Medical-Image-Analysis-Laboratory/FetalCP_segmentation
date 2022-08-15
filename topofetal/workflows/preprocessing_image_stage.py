# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the preprocessing of SR reconstructed images for
 fetal CP segmentation."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util

from topofetal.interfaces import preprocessing

def create_preproc_image_stage(p_name="preproc_image_stage"):
    """Create a Preprocessing workflow for segmentation inference
    Parameters
    ----------
    ::
        p_name : name of workflow (default: preproc_image_stage)
    Inputs::
        inputnode.input_image : Input image
        inputnode.input_dseg : Input tissue segmentation
    Outputs::
        outputnode.output_image : Output masked image
    """

    inference_stage = pe.Workflow(name=p_name)

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=['input_image', 'input_dseg']),
        name='inputnode'
    )

    binarize_tiv = pe.Node(
        interface=preprocessing.BinaryThresholdImage(),
        name='binarize_tiv'
    )
    binarize_tiv.inputs.threshold = 1

    mask_sr = pe.Node(
        interface=preprocessing.MaskImage(),
        name='mask_sr'
    )

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=['output_image']),
        name='outputnode')

    inference_stage.connect(inputnode, 'input_dseg',
                            binarize_tiv, 'in_map')

    inference_stage.connect(binarize_tiv, 'out_map',
                            mask_sr, 'in_mask')
    inference_stage.connect(inputnode, 'input_image',
                            mask_sr, 'in_image')

    inference_stage.connect(mask_sr, 'out_image',
                            outputnode, 'output_image')

    return inference_stage
