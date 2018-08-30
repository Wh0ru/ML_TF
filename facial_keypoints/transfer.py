import numpy as np

input_height,input_width=96,96

def transfer_target(y_pred,thresh=0,n_points=64):
    y_pred_xy=[]
    for i in range(y_pred.shape[0]):
        hm=y_pred[i]
        y_pred_xy.append(transfer_xy_coord(hm,n_points,thresh))
    return np.array(y_pred_xy)

def transfer_xy_coord(hm,n_points=64,thresh=0.2):
    assert len(hm.shape)==3
    Nlandmark=hm.shape[-1]
    est_xy=[]
    for i in range(Nlandmark):
        hmi=hm[:,:,i]
        est_xy.extend(get_ave_xy(hmi,n_points,thresh))
    return (est_xy)

def get_ave_xy(hmi,n_points=4,thresh=0):
    if n_points<1:
        hsum,n_points=np.sum(hmi),len(hmi.flatten())
        ind_hmi=np.array([range(input_width)]*input_height)
        i1=np.sum(ind_hmi*hmi)/hsum
        ind_hmi=np.array([range(input_height)]*input_width).T
        i0=np.sum(ind_hmi*hmi)/hsum
    else:
        ind=hmi.argsort(axis=None)[-n_points:]
        topind=np.unravel_index(ind,hmi.shape)
        index=np.unravel_index(hmi.argmax(),hmi.shape)
        i0,i1,hsum=0,0,0
        for ind in zip(topind[0],topind[1]):
            h=hmi[ind[0],ind[1]]
            hsum+=h
            i0+=ind[0]*h
            i1+=ind[1]*h

        i0/=hsum
        i1/=hsum
    if hsum/n_points<=thresh:
        i0,i1=-1,-1
    return [i0,i1]