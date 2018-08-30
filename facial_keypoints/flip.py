from skimage import transform
from skimage.transform import SimilarityTransform, AffineTransform
import random
import numpy as np


def transform_img(data,
                  loc_w_batch=2,
                  max_rotation=0.01,
                  max_shift=2,
                  max_shear=0,
                  max_scale=0.01, mode="edge"):
    '''
    data : list of numpy arrays containing a single image
    e.g., data = [X, y, w] or data = [X, y]
    X.shape = (height, width, NfeatX)
    y.shape = (height, width, Nfeaty)
    w.shape = (height, width, Nfeatw)
    NfeatX, Nfeaty and Nfeatw can be different

    affine transformation for a single image

    loc_w_batch : the location of the weights in the fourth dimention
    [,,,loc_w_batch]
    '''
    scale = (np.random.uniform(1 - max_scale, 1 + max_scale),
             np.random.uniform(1 - max_scale, 1 + max_scale))
    rotation_tmp = np.random.uniform(-1 * max_rotation, max_rotation)
    translation = (np.random.uniform(-1 * max_shift, max_shift),
                   np.random.uniform(-1 * max_shift, max_shift))
    shear = np.random.uniform(-1 * max_shear, max_shear)
    tform = AffineTransform(
        scale=scale,  # ,
        ## Convert angles from degrees to radians.
        rotation=np.deg2rad(rotation_tmp),
        translation=translation,
        shear=np.deg2rad(shear)
    )

    for idata, d in enumerate(data):
        if idata != loc_w_batch:
            ## We do NOT need to do affine transformation for weights
            ## as weights are fixed for each (image,landmark) combination
            data[idata] = transform.warp(d, tform, mode=mode)
    return data


def transform_imgs(data, lm,
                   loc_y_batch=1,
                   loc_w_batch=2):
    '''
    data : list of numpy arrays containing a single image
    e.g., data = [X, y, w] or data = [X, y]
    X.shape = (height, width, NfeatX)
    y.shape = (height, width, Nfeaty)
    w.shape = (height, width, Nfeatw)
    NfeatX, Nfeaty and Nfeatw can be different

    affine transformation for a single image
    '''
    Nrow = data[0].shape[0]
    Ndata = len(data)
    data_transform = [[] for i in range(Ndata)]
    for irow in range(Nrow):
        data_row = []
        for idata in range(Ndata):
            data_row.append(data[idata][irow])
        ## affine transformation
        data_row_transform = transform_img(data_row,
                                           loc_w_batch)
        ## horizontal flip
        data_row_transform = horizontal_flip(data_row_transform,
                                             lm,
                                             loc_y_batch,
                                             loc_w_batch)

        for idata in range(Ndata):
            data_transform[idata].append(data_row_transform[idata])

    for idata in range(Ndata):
        data_transform[idata] = np.array(data_transform[idata])

    return (data_transform)


def horizontal_flip(data, lm, loc_y_batch=1, loc_w_batch=2):
    '''
    flip the image with 50% chance

    lm is a dictionary containing "orig" and "new" key
    This must indicate the potitions of heatmaps that need to be flipped
    landmark_order = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
                      "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}

    data = [X, y, w]
    w is optional and if it is in the code, the position needs to be specified
    with loc_w_batch

    X.shape (height,width,n_channel)
    y.shape (height,width,n_landmarks)
    w.shape (height,width,n_landmarks)
    '''
    lo, ln = np.array(lm["orig"]), np.array(lm["new"])

    assert len(lo) == len(ln)
    if np.random.choice([0, 1]) == 1:
        return (data)

    for i, d in enumerate(data):
        d = d[:, ::-1, :]
        data[i] = d

    data[loc_y_batch] = swap_index_for_horizontal_flip(
        data[loc_y_batch], lo, ln)

    # when horizontal flip happens to image, we need to heatmap (y) and weights y and w
    # do this if loc_w_batch is within data length
    if loc_w_batch < len(data):
        data[loc_w_batch] = swap_index_for_horizontal_flip(
            data[loc_w_batch], lo, ln)
    return (data)


def swap_index_for_horizontal_flip(y_batch, lo, ln):
    '''
    lm = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
          "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}
    lo, ln = np.array(lm["orig"]), np.array(lm["new"])
    '''
    y_orig = y_batch[:, :, lo]
    y_batch[:, :, lo] = y_batch[:, :, ln]
    y_batch[:, :, ln] = y_orig
    return (y_batch)