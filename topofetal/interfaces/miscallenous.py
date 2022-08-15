# Author: Priscille de Dumast
# Date: 15.08.2022

"""TopoFetal utils functions."""

import numpy as np
import skimage


def crop_volumes(p_gt, p_pr):

    gt_ri = skimage.measure.regionprops((p_gt > 0).astype(np.uint8), p_gt)
    gt_min_0, gt_min_1, gt_min_2, gt_max_0, gt_max_1, gt_max_2 = gt_ri[0].bbox

    pr_ri = skimage.measure.regionprops((p_pr > 0).astype(np.uint8), p_pr)
    pr_min_0, pr_min_1, pr_min_2, pr_max_0, pr_max_1, pr_max_2 = pr_ri[0].bbox

    min_0 = max(0, min(gt_min_0 - 2, pr_min_0 - 2))
    max_0 = min(p_gt.shape[0], max(gt_max_0 + 2, pr_max_0 + 2))

    min_1 = max(0, min(gt_min_1 - 2, pr_min_1 - 2))
    max_1 = min(p_gt.shape[1], max(gt_max_1 + 2, pr_max_1 + 2))

    min_2 = max(0, min(gt_min_2 - 2, pr_min_2 - 2))
    max_2 = min(p_gt.shape[2], max(gt_max_2 + 2, pr_max_2 + 2))

    n_gt = p_gt[min_0:max_0, min_1:max_1, min_2:max_2]
    n_pr = p_pr[min_0:max_0, min_1:max_1, min_2:max_2]

    return n_gt, n_pr, [min_0,max_0, min_1,max_1, min_2,max_2]

