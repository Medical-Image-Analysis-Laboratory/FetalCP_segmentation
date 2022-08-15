# Author: Priscille de Dumast
# Date: 15.08.2022

from nipype import config, logging
from nipype.interfaces.io import DataGrabber, DataSink
import nipype.pipeline.engine as pe

from nipype.interfaces.ants.utils import *

from topofetal.workflows import inference_stage
from topofetal.workflows import preprocessing_image_stage
from topofetal.workflows import preprocessing_gt_stage
from topofetal.workflows import ensemble_stage


class InferencePipeline:
    """Class used to represent the workflow of inference
    of fetal automatic CP segmentation.

    Attributes
    -----------
    m_dataset : string
        Dataset of the subject of interest (required)
    m_subject : string
        Subject ID to process (required)
    m_session : string
        Session ID to process (optional)
    m_output_dir : string
        Output directory
    m_wf : pe.Workflow()
        Nipype workflow of the inference and assessment
    m_net_name : string
        Network name
    m_net_dir : string
        Path to network directory
    m_folds : list of int
        Folds to consider
    """
    m_dataset = None
    m_subject = None
    m_session = None

    m_output_dir = None
    m_wf = None

    m_net_name = None
    m_net_dir = None
    m_folds = None


    def __init__(self,
                 p_data_dir,
                 p_output_dir,
                 p_subject,
                 p_net_dir,
                 p_net_name):
        """Constructor of InferencePipeline class instance."""

        self.m_data_dir = p_data_dir
        self.m_subject = p_subject

        self.m_output_dir = p_output_dir
        self.m_net_name = p_net_name
        self.m_net_dir = p_net_dir

        self.m_folds = [0,1,2,3]


    def create_workflow(self):
        """Create the Nipype workflow
        for inference segmentation of the fetal CP.
        """
        wf_base_dir = os.path.join(self.m_output_dir, "nipype_segment_sr", self.m_subject, "net-{}".format(self.m_net_name))
        final_res_dir = os.path.join(self.m_output_dir, "fetalCP_segment_sr", self.m_subject)

        if not os.path.exists(wf_base_dir):
            os.makedirs(wf_base_dir)
        print("Process directory: {}".format(wf_base_dir))

        self.m_wf = pe.Workflow(name="inference",base_dir=wf_base_dir)

        # Initialization
        if os.path.isfile(os.path.join(wf_base_dir, "pypeline_" + self.m_subject + ".log")):
            os.unlink(os.path.join(wf_base_dir, "pypeline_" + self.m_subject + ".log"))

        config.update_config({'logging': {'log_directory': os.path.join(wf_base_dir),
                                          'log_to_file': True},
                              'execution': {
                                  'remove_unnecessary_outputs': False,
                                  'keep_unnecessary_outputs': True,
                                  'stop_on_first_crash': True,
                                  'stop_on_first_rerun': False,
                                  'crashfile_format': "txt",
                                  'write_provenance': False
                              },
                              'monitoring': {'enabled': True}
                              })
        logging.update_logging(config)
        iflogger = logging.getLogger('nipype.interface')

        iflogger.info("**** Processing ****")


        dg = pe.Node(interface=DataGrabber(outfields=['SR', 'dseg']), name='data_grabber')

        t2w = os.path.join(self.m_data_dir, self.m_subject, 'anat', self.m_subject+'_rec-irtk_T2w.nii.gz')
        dseg = os.path.join(self.m_data_dir, self.m_subject, 'anat', self.m_subject+'_rec-irtk_dseg.nii.gz')

        dg.inputs.template = '*'
        dg.inputs.sort_filelist = True
        dg.inputs.field_template = dict(SR=t2w, dseg=dseg)

        segmentation_preproc_image = preprocessing_image_stage.create_preproc_image_stage(
            p_name = 'segmentation_preproc_image'
        )

        segmentation_preproc_gt = preprocessing_gt_stage.create_preproc_gt_stage(
            p_dataset = self.m_dataset,
            p_name = 'segmentation_preproc_gt'
        )

        segmentation_inference_workflows = []

        for i_fold in self.m_folds:
            segmentation_inference = inference_stage.create_inference_stage(
                p_testing_mode=True,
                name = 'inference_stage_' + str(i_fold)
            )
            segmentation_inference.inputs.inputnode.input_network_dir = self.m_net_dir
            segmentation_inference.inputs.inputnode.input_network_name = self.m_net_name
            segmentation_inference.inputs.inputnode.input_fold = i_fold

            segmentation_inference_workflows.append(segmentation_inference)


        segmentation_ensemble = ensemble_stage.create_ensemble_stage(
            p_folds = self.m_folds,
            p_name='segmentation_ensemble'
        )

        datasinker = pe.Node(interface=DataSink(), name='datasinker')
        datasinker.inputs.base_directory = final_res_dir
        datasinker.inputs.substitutions = [('_votes_cc_metrics_holes', '_cc_metrics_holes'),
                                           ('_votes_cc',  '_cc')]

        self.m_wf.connect(dg, 'SR',
                          segmentation_preproc_image, 'inputnode.input_image')

        self.m_wf.connect(dg, 'dseg',
                          segmentation_preproc_image, 'inputnode.input_dseg')
        self.m_wf.connect(dg, 'dseg',
                          segmentation_preproc_gt, 'inputnode.input_dseg')

        for i, segmentation_inference in enumerate(segmentation_inference_workflows):

            self.m_wf.connect(segmentation_preproc_image, 'outputnode.output_image',
                              segmentation_inference, 'inputnode.input_image')

            self.m_wf.connect(segmentation_inference, 'outputnode.output_likelihoods',
                              segmentation_ensemble, 'inputnode.input_predictions_' + str(i + 1))

        self.m_wf.connect(segmentation_ensemble, 'outputnode.output_prediction',
                          datasinker, 'anat.@output_prediction')


    def run(self):
        """Execute the workflow.
        """
        self.m_wf.write_graph(dotfilename='graph.dot', graph2use='colored', format='png', simple_form=True)
        # self.m_wf.config['remove_unnecessary_outputs'] = False
        res = self.m_wf.run()

        return res
