# Author: Priscille de Dumast
# Date: 15.08.2022

"""Post-inference modules in the segmentation of fetal CP.
"""

import os
import numpy as np
import pandas as pd

import tensorflow as tf
import SimpleITK as sitk
import skimage

import medpy
from medpy import metric
import skimage.measure
from scipy import ndimage
import gudhi as gd

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import traits, TraitedSpec, File, BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import InputMultiPath

from topofetal.interfaces import miscallenous as misc


class PredictionMergerInputSpec(BaseInterfaceInputSpec):
    """Class used for the input of PredictionMerger.
    """
    input_images = InputMultiPath(File(mandatory=True), desc='Input image filenames to be normalized')


class PredictionMergerOutputSpec(TraitedSpec):
    """Class used for the output of PredictionMerger.
    """
    output_image = File(desc='Output label map.')
    output_image_cc = File(desc='Output label map.')



class PredictionMerger(BaseInterface):
    """Class used to perform a majority voting between over multiple predictions.
    """
    input_spec = PredictionMergerInputSpec
    output_spec = PredictionMergerOutputSpec

    def _gen_filename(self, name):
        if name == 'output_image':
            _, name, ext = split_filename(self.inputs.input_images[0])
            output = name + ext
            return os.path.abspath(output)
        if name == 'output_image_cc':
            _, name, ext = split_filename(self.inputs.input_images[0])
            output = name + '_cc' + ext
            return os.path.abspath(output)
        return None

    def _merge_prediction(self):

        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(self.inputs.input_images[0])
        image_sitk = reader.Execute()
        image = sitk.GetArrayFromImage(image_sitk)

        if len(self.inputs.input_images) > 1:
            for p in self.inputs.input_images[1:]:
                reader.SetFileName(p)
                image += sitk.GetArrayFromImage(reader.Execute())

        res = np.argmax(image, axis=3)

        out = sitk.GetImageFromArray(res)
        out.CopyInformation(image_sitk)
        out = sitk.Cast(out, sitk.sitkUInt8)

        writer.SetFileName(self._gen_filename('output_image'))
        writer.Execute(out)

        # Extract the biggest connected component
        all_labels = skimage.measure.label(res, background=0)
        largestCC = 1 * (all_labels == np.argmax(np.bincount(all_labels.flat)[1:])+1)

        out = sitk.GetImageFromArray(largestCC)
        out.CopyInformation(image_sitk)
        out = sitk.Cast(out, sitk.sitkUInt8)

        writer.SetFileName(self._gen_filename('output_image_cc'))
        writer.Execute(out)

        return

    def _run_interface(self, runtime):
        try:
            self._merge_prediction()
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = os.path.abspath( self._gen_filename('output_image') )
        outputs['output_image_cc'] = os.path.abspath( self._gen_filename('output_image_cc') )
        return outputs


############################################

class MetricsComputerVolumeInputSpec(BaseInterfaceInputSpec):
    """Class used for the input of MetricsComputerVolume.
    """
    input_prediction = File(mandatory=True, desc='Input prediction')
    input_ground_truth = File(mandatory=False, desc='Input ground truth reference segmentation')

    fold = traits.Int(default_value=-1, usedefault=True, mandatory=False)

class MetricsComputerVolumeOutputSpec(TraitedSpec):
    """Class used for the output of MetricsComputerVolume.
    """
    output_metrics = File(desc='Output csv file')
    output_metrics_holes = File(desc='Output csv file holes')

    output_cortex_with_holes = File(desc='Label map with holes')


class MetricsComputerVolume(BaseInterface):
    """Class used for the computation of whole cortical volume metrics.
    """

    input_spec = MetricsComputerVolumeInputSpec
    output_spec = MetricsComputerVolumeOutputSpec

    m_num_classes = 2

    def _gen_filename(self, name):
        _, bname, _ = split_filename(self.inputs.input_prediction)
        if name == 'output_metrics':
            output = bname + '_metrics.csv'
            return os.path.abspath(output)
        if name == 'output_metrics_holes':
            output = bname + '_metrics_holes.csv'
            return os.path.abspath(output)
        if name == 'output_cortex_with_holes':
            output = bname + '_output_cortex_with_holes.nii.gz'
            return os.path.abspath(output)
        return None

    def get_simplex_of_mask(self, mask, p_max_alpha_square=0.7):
        points = [[index[0], index[1], index[2]] for index, x in np.ndenumerate(mask) if x]
        simplex_tree = gd.AlphaComplex(points=points).create_simplex_tree(
            max_alpha_square=p_max_alpha_square)
        return simplex_tree

    def compute_betti_numbers_of_mask(self, mask, p_max_alpha_square=0.7):
        simplex_tree = self.get_simplex_of_mask(mask, p_max_alpha_square=p_max_alpha_square)
        simplex_tree.persistence()
        bns = simplex_tree.betti_numbers()
        return bns

    def _compute_holes_persistence(self, m, p):

        def compute_persistence_with_cubical_complex(nda):
            nda_cubic = gd.CubicalComplex(dimensions=nda.shape, top_dimensional_cells=nda.flatten(order='F'))
            pers = nda_cubic.persistence(homology_coeff_field=2, min_persistence=0.99)
            pairs_nda = nda_cubic.cofaces_of_persistence_pairs()
            pairs_nda_struct = pairs_nda[0]
            return [pairs_nda_struct[2].shape[0], pairs_nda_struct[1].shape[0], pairs_nda_struct[0].shape[0]], pers

        # Ground truth
        gt_cubic_bn, _ = compute_persistence_with_cubical_complex(m)

        # Iteration 0: Before dilatation
        i_iter = 0
        pr_m = p * m
        pr_m_cubic_bn = compute_persistence_with_cubical_complex(pr_m)

        # Iteration i > 0
        pr_dil_m = pr_m
        sum_pr_dil_m = pr_m

        element_structurant = ndimage.generate_binary_structure(3, 3)

        while pr_m_cubic_bn != gt_cubic_bn:
            i_iter += 1
            pr_dil_m = ndimage.binary_dilation(pr_dil_m, structure=element_structurant, mask=m)
            pr_m_cubic_bn, _ = compute_persistence_with_cubical_complex(pr_dil_m)
            print(i_iter, 'iteration temporary (?) Betti numbers', pr_m_cubic_bn)
            sum_pr_dil_m += pr_dil_m

        final_number_of_iterations = i_iter

        # Perform one additional iteration, to split 1d holes of the ground truth from the others in the persistence
        print()
        i_iter += 1
        pr_dil_m = ndimage.binary_dilation(pr_dil_m, structure=element_structurant, mask=m)
        sum_pr_dil_m += pr_dil_m
        _, diag = compute_persistence_with_cubical_complex(sum_pr_dil_m)

        holes = [(dim, bd) for (dim, bd) in diag if dim == 1 and bd[1] <= final_number_of_iterations + 1]
        holes_pers = [hole[1][1] - hole[1][0] for hole in holes]

        return holes_pers

    def _compute_volume_of_false_negative_connected_to_a_trou(self, m, p):
        p_m = p * m

        mask_des_potential_trous = np.logical_or(m, p_m).astype(int)
        mask_des_potential_trous = mask_des_potential_trous - \
                                   np.logical_or(np.logical_and(m, p), p).astype(int)

        gaussianFilter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussianFilter.SetSigma(3)

        p_b_m = sitk.GetArrayFromImage(gaussianFilter.Execute(sitk.GetImageFromArray(p_m)))
        p_b_m = p_b_m * m
        p_b_m = np.where(p_m, p, p_b_m)

        padwith = 1

        nda = np.pad(p_b_m, padwith, 'constant', constant_values=0)
        nda_cubic = gd.CubicalComplex(dimensions=nda.shape,
                                      top_dimensional_cells=nda.flatten(order='F'))
        nda_cubic.persistence(homology_coeff_field=2, min_persistence=0.3)
        pairs_nda = nda_cubic.cofaces_of_persistence_pairs()

        # Filter pairs_lh to remove background component
        pairs_nda_struct = pairs_nda[0]
        bcp_lh_1d = [np.unravel_index(pair_of_holes[0], nda.shape, order='F') for pair_of_holes in
                     pairs_nda_struct[1]]

        b_d = np.zeros(nda.shape)
        for t in bcp_lh_1d:
            b_d[t[0], t[1], t[2]] = 1
        b_d = b_d[padwith:-padwith, padwith:-padwith, padwith:-padwith]

        tous_les_trous = ndimage.binary_propagation(input=b_d,
                                                    structure=ndimage.generate_binary_structure(3, 1),
                                                    mask=mask_des_potential_trous).astype(int)

        cortex_avec_ses_trous = (m + 2 * tous_les_trous).astype(int)
        cortex_avec_ses_trous = np.where(cortex_avec_ses_trous == 3, 2, cortex_avec_ses_trous).astype(int)

        volumes_holes_fn = len(np.where(cortex_avec_ses_trous == 2)[0]) / len(np.where(m)[0])

        return volumes_holes_fn, cortex_avec_ses_trous

    def _compute(self):

        model = self.inputs.input_prediction.split('net-')[1].split('/')[0] if 'net-' in self.inputs.input_prediction else 'Manual'
        sub = os.path.basename(self.inputs.input_prediction).split('_')[0]


        reader = sitk.ImageFileReader()

        reader.SetFileName(self.inputs.input_ground_truth)
        gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
        gt = sitk.GetArrayFromImage(gt_sitk)

        reader.SetFileName(self.inputs.input_prediction)
        pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
        pr = sitk.GetArrayFromImage(pr_sitk)

        orig_shape = gt.shape
        vxl_spacing = gt_sitk.GetSpacing()
        gt, pr, coord = misc.crop_volumes(gt, pr)

        gt = tf.keras.utils.to_categorical(gt, num_classes=self.m_num_classes)
        pr = tf.keras.utils.to_categorical(pr, num_classes=self.m_num_classes)

        names = ['Subject', 'Model', 'Label',
                 'DSC',                                  # Similarity
                 'ASSD',                                 # Boundary/Surface-based
                 'BN0', 'BN1', 'BN2', 'VolumeHolesFN']   # Topology

        metrics = []

        for label in range(1,self.m_num_classes):
            m = gt[:, :, :, label]
            p = pr[:, :, :, label]

            sm = sitk.Cast( sitk.GetImageFromArray(m), sitk.sitkUInt8 )
            sp = sitk.Cast( sitk.GetImageFromArray(p), sitk.sitkUInt8 )
            sm.SetSpacing(vxl_spacing)
            sp.SetSpacing(vxl_spacing)

            # ------------------------- #
            # - Overlap-based metrics - #
            # ------------------------- #
            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_measures_filter.Execute(sm, sp)

            dsc = overlap_measures_filter.GetDiceCoefficient()

            # -------------------------- #
            # - Boundary-based metrics - #
            # -------------------------- #

            average_sym_surf_dist =  medpy.metric.binary.assd(p, m, voxelspacing=vxl_spacing, connectivity=2)

            # -------------------------- #
            # - Topology-based metrics - #
            # -------------------------- #

            # Biggest connected component of the ground truth
            all_labels = skimage.measure.label(m, background=0)
            m = 1 * (all_labels == np.argmax(np.bincount(all_labels.flat)[1:]) + 1)

            # Betti numbers
            max_alpha_square = 0.75
            [p_bn_0, p_bn_1, p_bn_2] = self.compute_betti_numbers_of_mask(p, p_max_alpha_square=max_alpha_square)

            hole_ratio = 0
            if self.inputs.fold == -1:
                hole_ratio, cortex_avec_ses_trous = self._compute_volume_of_false_negative_connected_to_a_trou(m, p)

                [min_0, max_0, min_1, max_1, min_2, max_2] = coord
                cortex_avec_ses_trous = np.pad(cortex_avec_ses_trous,
                                               pad_width=((min_0, orig_shape[0]-max_0), (min_1, orig_shape[1]-max_1), (min_2, orig_shape[2]-max_2)),
                                               mode='constant',
                                               constant_values=0)

                tmp = sitk.GetImageFromArray(cortex_avec_ses_trous)
                tmp.CopyInformation(gt_sitk)

                writer = sitk.ImageFileWriter()
                
                writer.SetFileName(self._gen_filename('output_cortex_with_holes'))
                tmp = sitk.Cast(tmp, sitk.sitkUInt8)
                writer.Execute(tmp)


            row = [sub, model, label,
                   dsc, # Overlap similarity
                   average_sym_surf_dist, # Distance difference
                   p_bn_0, p_bn_1, p_bn_2, hole_ratio]  # Topology-based measures

            metrics.append(row)

        df_metrics = pd.DataFrame(metrics, columns=names)
        df_metrics.insert(loc=0, column='Fold', value=[self.inputs.fold for i in range(len(metrics))])
        df_metrics.to_csv(self._gen_filename('output_metrics'), index=False, header=True,sep=',')

        return

    def _run_interface(self, runtime):

        try:
            self._compute()
        except Exception as e:
            print('Failed call to _compute() from _run_interface()')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_metrics'] = os.path.abspath(self._gen_filename('output_metrics'))
        outputs['output_metrics_holes'] = os.path.abspath(self._gen_filename('output_metrics_holes'))
        outputs['output_cortex_with_holes'] = os.path.abspath(self._gen_filename('output_cortex_with_holes'))
        return outputs


def merge_metrics(input_image, network_name, in_files):
    import pandas as pd, os
    from nipype.utils.filemanip import split_filename
    res = pd.concat([pd.read_csv(s, index_col=False) for s in in_files])
    out_file = os.path.abspath(split_filename(input_image)[1] + '_' + network_name + '.csv')
    res.to_csv(out_file, index=False, header=True, sep=',')
    return out_file
