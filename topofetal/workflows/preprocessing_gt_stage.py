# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the preprocessing of the ground truth segmentation in fetal CP segmentation inference."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util

from topofetal.interfaces import preprocessing

def create_preproc_gt_stage(p_dataset, p_name="preproc_gt_stage"):
    """Create a Preprocessing workflow for segmentation inference
    Parameters
    ----------
    ::
        p_name : name of workflow (default: preproc_gt_stage)
        p_dataset :
    Inputs::
        inputnode.input_dseg : Input multi-tissue label map
    Outputs::
        outputnode.output_image : Output cortical plate label map
    Example
    -------
    """


    preproc_gt_stage = pe.Workflow(name=p_name)

    inputnode = pe.Node(
        interface=util.IdentityInterface( fields=['input_dseg']),
        name='inputnode'
    )

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=['output_map']),
        name='outputnode'
    )


    correspondance_to_feta_labels = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

    convert_labelmap = pe.Node(
        interface=preprocessing.ConvertLabelmap(),
        name='convert_labelmaps'
    )
    convert_labelmap.inputs.correspondance = correspondance_to_feta_labels

    extract_cortexlabel = pe.Node(
        interface=preprocessing.ExtractLabel(),
        name='extract_cortexlabel'
    )

    extract_cortexlabel.inputs.label = 2 # 2: FeTA cortical GM label

    preproc_gt_stage.connect(inputnode, 'input_dseg',
                             convert_labelmap, 'in_map')

    preproc_gt_stage.connect(convert_labelmap, 'out_map',
                             extract_cortexlabel, 'in_map')

    preproc_gt_stage.connect(extract_cortexlabel, 'out_map',
                             outputnode, 'output_map')

    return preproc_gt_stage
