# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the fetal CP segmentation inference."""


from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util

from topofetal.interfaces import segmentation
from topofetal.interfaces import postprocessing


def create_inference_stage(p_in_axis = [0,1,2], p_testing_mode=False, name="inference_stage"):
    """Create a segmentation inference workflow
    Parameters
    ----------
    ::
        name : name of workflow (default: inference_stage)
        p_in_axis : axes of extraction of the 2D patches (list)
        p_testing_mode : if true, majority voting between axis is not performed
    Inputs
    ------
    ::
        inputnode.input_image : Input SR image to segment
    Outputs
    -------
    ::
        outputnode.output_prediction : if in testing mode, final prediction as a majority vote over all extraction axis
        outputnode.output_metrics : if in testing mode, metrics of global prediction
        outputnode.output_likelihoods : if not in testing mode,
                                        likelihood maps of each extraction axis (list)
    """

    inference_stage = pe.Workflow(name=name)

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['input_image',
                    'input_network_dir', 'input_network_name', 'input_fold']),
        name='inputnode'
    )

    segmenter = pe.Node(
        interface=segmentation.Segmenter(),
        name='segmenter'
    )
    segmenter.iterables = [('input_extraction_axis', p_in_axis)]

    if not p_testing_mode:
        majorityvoting = pe.JoinNode(
            interface=postprocessing.PredictionMerger(),
            name='majorityvoting',
            joinsource=segmenter.name,
            joinfield='input_images'
        )

        output_fields = ['output_prediction', 'output_metrics']
        outputnode = pe.Node(
            interface=util.IdentityInterface(fields= output_fields),
            name='outputnode'
        )
    else:
        outputnode = pe.JoinNode(
            interface=util.IdentityInterface(fields=['output_likelihoods']),
            joinsource=segmenter.name, joinfield='output_likelihoods',
        name='outputnode')

    inference_stage.connect(inputnode, 'input_image',
                            segmenter, 'input_image')
    inference_stage.connect(inputnode, 'input_network_dir',
                            segmenter, 'input_network_dir')
    inference_stage.connect(inputnode, 'input_network_name',
                            segmenter, 'input_network_name')
    inference_stage.connect(inputnode, 'input_fold',
                            segmenter, 'input_fold')

    if not p_testing_mode:
        inference_stage.connect(segmenter, 'output_labelmap',
                                majorityvoting, 'input_images')

        inference_stage.connect(majorityvoting, 'output_image_cc',
                                outputnode, 'output_prediction')
    else:
        inference_stage.connect(segmenter, 'output_likelihood',
                                outputnode, 'output_likelihoods')

    return inference_stage
