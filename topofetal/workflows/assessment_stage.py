# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the assessment of CP automatic segmentation
compared to a ground truth reference label map."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util

from topofetal.interfaces import postprocessing


def create_assessment_stage(p_name="assessment_stage"):
    """Create a Preprocessing workflow for segmentation inference
    Parameters
    ----------
    ::
        p_name : name of workflow (default: assessment_stage)
        p_dataset :
    Inputs::
        inputnode.input_ground_truth : Input ground truth CP label map
        inputnode.input_prediction : Input predicted segmentation label map
        inputnode.input_fold : ID of the fold being assessed
    Outputs::
        outputnode.output_metrics : Output metrics (.csv file)
        outputnode.output_cortex_with_holes : Output label map of TP and FN_holes
    Example
    -------
    """

    assessment_stage = pe.Workflow(name=p_name)

    input_fields = ['input_ground_truth', 'input_prediction', 'input_fold']
    output_fields = ['output_metrics', 'output_cortex_with_holes']

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields = input_fields),
        name='inputnode'
    )

    metrics = pe.Node(
        interface=postprocessing.MetricsComputerVolume(),
        name='metrics'
    )

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields = output_fields),
        name='outputnode'
    )

    assessment_stage.connect(inputnode, 'input_prediction',
                             metrics, 'input_prediction')

    assessment_stage.connect(inputnode, 'input_ground_truth',
                             metrics, 'input_ground_truth')
    assessment_stage.connect(inputnode, 'input_fold',
                             metrics, 'fold')


    assessment_stage.connect(metrics, 'output_metrics',
                             outputnode, 'output_metrics')
    assessment_stage.connect(metrics, 'output_cortex_with_holes',
                             outputnode, 'output_cortex_with_holes')

    return assessment_stage
