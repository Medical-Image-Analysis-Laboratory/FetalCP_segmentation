# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the ensemble learning stage of
fetal CP segmentation inference."""


from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util

from topofetal.interfaces import postprocessing

def create_ensemble_stage(p_folds, p_name="ensemble_stage"):
    """Create an ensemble learning workflow for segmentation inference.
    Compute the majority voting of an ensemble of predictions.
    Parameters
    ----------
    ::
        p_name : name of workflow (default: ensemble_stage)
        p_folds : Folds to consider (list)
    Inputs
    ------
    ::
        inputnode.input_predictions : Ensemble of predictions (list)
    Outputs
    ------
    ::
        outputnode.output_prediction : Output prediction
    """


    ensemble_stage = pe.Workflow(name=p_name)

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['input_predictions_'+str(i+1) for i in range(len(p_folds))]
        ),
        name='inputnode'
    )

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=['output_prediction']),
        name='outputnode'
    )

    merge_votes = pe.Node(
        interface=util.Merge(len(p_folds)),
        name='merge_votes'
    )

    ensemble_segmentation = pe.Node(
        interface=postprocessing.PredictionMerger(),
        name='ensemble_segmentation'
    )

    for i in range(len(p_folds)):
        ensemble_stage.connect(inputnode, 'input_predictions_'+str(i+1),
                               merge_votes, 'in' + str(i + 1))


    ensemble_stage.connect(merge_votes, 'out',
                           ensemble_segmentation, 'input_images')

    ensemble_stage.connect(ensemble_segmentation, 'output_image_cc',
                           outputnode, 'output_prediction')

    return ensemble_stage
