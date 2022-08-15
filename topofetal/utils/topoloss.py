#
# Author: Priscille de Dumast
# Date: 15.08.2022
# Code adapted from https://github.com/HuXiaoling/TopoLoss

import numpy
import gudhi as gd

import numpy as np

def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.00001, pers_thresh_perfect=0.99):

    if (len(lh_dgm.shape) != 2):
        # To avoid this error:
        ## lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
        ## IndexError: too many indices for array: array is 1 - dimensional, but 2 were indexed
        idx_struct_to_fix = list();
        idx_struct_to_remove = list()
        return idx_struct_to_fix, idx_struct_to_remove

    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_struct = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_struct = gt_pers.size  # number of struct in gt

    if (gt_pers is None or gt_n_struct == 0):
        idx_struct_to_fix = list();
        idx_struct_to_remove = list(set(range(lh_pers.size)))
        idx_struct_perfect = list();
    else:
        # more lh dots than gt dots
        if (lh_pers.size < gt_n_struct):
            print('(lh_pers.size < gt_n_struct)', (lh_pers.size < gt_n_struct))
            gt_n_struct = lh_pers.size
        # assert lh_pers.size >= gt_n_struct

        ## check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect
        # assert tmp.sum() == gt_pers.size

        # get "perfect struct" - struct which do not need to be fixed, i.e., find top
        # lh_n_struct_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the struct
        # formed by the padded boundary
        # if no struct is ~1 (ie >.999) then just take all struct with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]

        # The k=lh_n_struct_perfect first struct are the perfect ones.
        lh_n_struct_perfect = tmp.sum()
        idx_struct_perfect = lh_pers_sorted_indices[:lh_n_struct_perfect] if lh_n_struct_perfect >= 1 else list()

        # find top gt_n_struct indices
        idx_struct_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_struct];

        # the difference is struct to be fixed to perfect
        idx_struct_to_fix = list(set(idx_struct_to_fix_or_perfect) - set(idx_struct_perfect))
        # remaining struct are all to be removed
        idx_struct_to_remove = lh_pers_sorted_indices[gt_n_struct:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    # TODO values below this are small dents so dont fix them; tune this value?
    idx_valid = np.where(lh_pers > pers_thresh)[0]

    idx_struct_to_remove = list(set(idx_struct_to_remove).intersection(set(idx_valid)))

    return idx_struct_to_fix, idx_struct_to_remove


def getCriticalPoints(lh):

    lh_vector = np.asarray(lh).flatten()
    lh_cubic = gd.CubicalComplex(dimensions=[lh.shape[0], lh.shape[1]], top_dimensional_cells=lh_vector)

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # Filter pairs_lh to remove background component
    pairs_lh_struct = pairs_lh[0]
    pairs_lh_cc_0d = pairs_lh_struct[1]
    pairs_lh_holes_1d = pairs_lh_struct[0]

    # return persistence diagram, birth/death critical points of 0-dim structures = connected
    pd_lh_0d = numpy.array([[lh_vector[pair_of_cc[0]], lh_vector[pair_of_cc[1]]] for pair_of_cc in pairs_lh_cc_0d])
    bcp_lh_0d = numpy.array([np.unravel_index(pair_of_cc[0], lh.shape) for pair_of_cc in pairs_lh_cc_0d])
    dcp_lh_0d = numpy.array([np.unravel_index(pair_of_cc[1], lh.shape) for pair_of_cc in pairs_lh_cc_0d])

    # return persistence diagram, birth/death critical points of 1-dim structures = holes
    pd_lh_1d = numpy.array([[lh_vector[pair_of_holes[0]], lh_vector[pair_of_holes[1]]] for pair_of_holes in pairs_lh_holes_1d])
    bcp_lh_1d = numpy.array([np.unravel_index(pair_of_holes[0], lh.shape) for pair_of_holes in pairs_lh_holes_1d])
    dcp_lh_1d = numpy.array([np.unravel_index(pair_of_holes[1], lh.shape) for pair_of_holes in pairs_lh_holes_1d])

    return pd_lh_0d, bcp_lh_0d, dcp_lh_0d, pd_lh_1d, bcp_lh_1d, dcp_lh_1d


def compute_loss_xd(lh_patch, idx_struct_to_fix_xd, idx_struct_to_remove_xd, bcp_lh_xd, dcp_lh_xd):
    patch_shape = lh_patch.shape
    topo_cp_weight_map_xd = np.zeros(patch_shape)
    topo_cp_ref_map_xd = np.zeros(patch_shape)

    for hole_indx in idx_struct_to_fix_xd:

        if (int(bcp_lh_xd[hole_indx][0]) >= 0 and int(bcp_lh_xd[hole_indx][0]) < patch_shape[0] and int(bcp_lh_xd[hole_indx][1]) >= 0 and int(bcp_lh_xd[hole_indx][1]) < patch_shape[1]):
            topo_cp_weight_map_xd[int(bcp_lh_xd[hole_indx][0]), int(bcp_lh_xd[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
            topo_cp_ref_map_xd[int(bcp_lh_xd[hole_indx][0]), int(bcp_lh_xd[hole_indx][1])] = 0

        if (int(dcp_lh_xd[hole_indx][0]) >= 0 and int(dcp_lh_xd[hole_indx][0]) < patch_shape[0] and int(dcp_lh_xd[hole_indx][1]) >= 0 and int(dcp_lh_xd[hole_indx][1]) < patch_shape[1]):
            topo_cp_weight_map_xd[int(dcp_lh_xd[hole_indx][0]), int(dcp_lh_xd[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
            topo_cp_ref_map_xd[int(dcp_lh_xd[hole_indx][0]), int(dcp_lh_xd[hole_indx][1])] = 1

    for hole_indx in idx_struct_to_remove_xd:

        if (int(bcp_lh_xd[hole_indx][0]) >= 0 and int(bcp_lh_xd[hole_indx][0]) < patch_shape[0] and int(bcp_lh_xd[hole_indx][1]) >= 0 and int(bcp_lh_xd[hole_indx][1]) < patch_shape[1]):
            topo_cp_weight_map_xd[int(bcp_lh_xd[hole_indx][0]), int(bcp_lh_xd[hole_indx][1])] = 1  # push birth to death  # push to diagonal

            if (int(dcp_lh_xd[hole_indx][0]) >= 0 and int(dcp_lh_xd[hole_indx][0]) < patch_shape[0] and int(dcp_lh_xd[hole_indx][1]) >= 0 and int(dcp_lh_xd[hole_indx][1]) < patch_shape[1]):
                topo_cp_ref_map_xd[int(bcp_lh_xd[hole_indx][0]), int(bcp_lh_xd[hole_indx][1])] = lh_patch[int(dcp_lh_xd[hole_indx][0]), int(dcp_lh_xd[hole_indx][1])]
            else:
                topo_cp_ref_map_xd[int(bcp_lh_xd[hole_indx][0]), int(bcp_lh_xd[hole_indx][1])] = 1

        if (int(dcp_lh_xd[hole_indx][0]) >= 0 and int(dcp_lh_xd[hole_indx][0]) < patch_shape[0] and int(dcp_lh_xd[hole_indx][1]) >= 0 and int(dcp_lh_xd[hole_indx][1]) < patch_shape[1]):
            topo_cp_weight_map_xd[int(dcp_lh_xd[hole_indx][0]), int(dcp_lh_xd[hole_indx][1])] = 1  # push death to birth # push to diagonal

            if (int(bcp_lh_xd[hole_indx][0]) >= 0 and int(bcp_lh_xd[hole_indx][0]) < patch_shape[0] and int(bcp_lh_xd[hole_indx][1]) >= 0 and int(bcp_lh_xd[hole_indx][1]) < patch_shape[1]):
                topo_cp_ref_map_xd[int(dcp_lh_xd[hole_indx][0]), int(dcp_lh_xd[hole_indx][1])] = lh_patch[int(bcp_lh_xd[hole_indx][0]), int(bcp_lh_xd[hole_indx][1])]
            else:
                topo_cp_ref_map_xd[int(dcp_lh_xd[hole_indx][0]), int(dcp_lh_xd[hole_indx][1])] = 0

    loss_topo_xd = (((lh_patch * topo_cp_weight_map_xd) - topo_cp_ref_map_xd) ** 2).sum()
    # print('loss_topo_xd: {:02f}'.format(loss_topo_xd))
    return loss_topo_xd

def filter_cp(lh_bce_bin, pd_lh_0d, bcp_lh_0d, dcp_lh_0d, pd_lh_1d, bcp_lh_1d, dcp_lh_1d):
    # - 0.d
    bcp_filtered_lh_0d = []
    dcp_filtered_lh_0d = []
    pd_filtered_lh_0d = []

    i_0d = 0
    for bcp, dcp in zip(bcp_lh_0d.tolist(), dcp_lh_0d.tolist()):
        if lh_bce_bin[bcp[0], bcp[1]] or lh_bce_bin[dcp[0], dcp[1]]:
            bcp_filtered_lh_0d.append([bcp[0], bcp[1]])
            dcp_filtered_lh_0d.append([dcp[0], dcp[1]])
            pd_filtered_lh_0d.append([pd_lh_0d[i_0d][0], pd_lh_0d[i_0d][1]])
        i_0d += 1

    bcp_filtered_lh_0d = np.asarray(bcp_filtered_lh_0d)
    dcp_filtered_lh_0d = np.asarray(dcp_filtered_lh_0d)
    pd_filtered_lh_0d = np.asarray(pd_filtered_lh_0d)

    # - 1.d
    bcp_filtered_lh_1d = []
    dcp_filtered_lh_1d = []
    pd_filtered_lh_1d = []

    i_1d = 0
    for bcp, dcp in zip(bcp_lh_1d.tolist(), dcp_lh_1d.tolist()):
        if lh_bce_bin[bcp[0], bcp[1]] or lh_bce_bin[dcp[0], dcp[1]]:
            bcp_filtered_lh_1d.append([bcp[0], bcp[1]])
            dcp_filtered_lh_1d.append([dcp[0], dcp[1]])
            pd_filtered_lh_1d.append([pd_lh_1d[i_1d][0], pd_lh_1d[i_1d][1]])
        i_1d += 1

    bcp_filtered_lh_1d = np.asarray(bcp_filtered_lh_1d)
    dcp_filtered_lh_1d = np.asarray(dcp_filtered_lh_1d)
    pd_filtered_lh_1d = np.asarray(pd_filtered_lh_1d)

    return pd_filtered_lh_0d, bcp_filtered_lh_0d, dcp_filtered_lh_0d, pd_filtered_lh_1d, bcp_filtered_lh_1d, dcp_filtered_lh_1d


def preproc_patch(patch):
    patch = np.pad(patch, (1, 1), 'constant', constant_values = np.max(patch))
    patch = np.pad(patch, (1, 1), 'constant', constant_values = 0)
    return patch


def getTopoLoss(likelihood, gt, min_pers_th):

    if likelihood.ndim == 3:
        likelihood = likelihood[:,:,1]
        gt = gt[:,:,1]

    likelihood_proc = preproc_patch(likelihood)
    gt_proc = preproc_patch(gt)

    pd_lh_0d, bcp_lh_0d, dcp_lh_0d, pd_lh_1d, bcp_lh_1d, dcp_lh_1d = getCriticalPoints(likelihood_proc)
    pd_gt_0d, bcp_gt_0d, dcp_gt_0d, pd_gt_1d, bcp_gt_1d, dcp_gt_1d = getCriticalPoints(gt_proc)

    idx_struct_to_fix_0d, idx_struct_to_remove_0d = compute_dgm_force(pd_lh_0d, pd_gt_0d, pers_thresh=min_pers_th)
    idx_struct_to_fix_1d, idx_struct_to_remove_1d = compute_dgm_force(pd_lh_1d, pd_gt_1d, pers_thresh=min_pers_th)

    loss_topo_0d = compute_loss_xd(lh_patch=likelihood_proc,
                                   idx_struct_to_fix_xd=idx_struct_to_fix_0d,
                                   idx_struct_to_remove_xd=idx_struct_to_remove_0d,
                                   bcp_lh_xd=bcp_lh_0d,
                                   dcp_lh_xd=dcp_lh_0d)

    loss_topo_1d = compute_loss_xd(lh_patch=gt_proc,
                                   idx_struct_to_fix_xd=idx_struct_to_fix_1d,
                                   idx_struct_to_remove_xd=idx_struct_to_remove_1d,
                                   bcp_lh_xd=bcp_lh_1d,
                                   dcp_lh_xd=dcp_lh_1d)

    return loss_topo_0d, loss_topo_1d

