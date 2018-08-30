import numpy as np

def find_weight(y_tra):
    '''

    :param y_tra: (None,96,96,15)
    :return:(None,96,96,15)
    '''
    weight=np.zeros_like(y_tra)
    count0,count1=0,0
    for irow in range(y_tra.shape[0]):
        for ifeat in range(y_tra.shape[-1]):
            if np.all(y_tra[irow,:,:,ifeat]==0):
                value=0
                count0+=1
            else:
                value=1
                count1+=1
            weight[irow,:,:,ifeat]=value
    return weight

def flatten_except_1dim(weight,ndim=2):
    n=weight.shape[0]
    if ndim==2:
        shape=(n,-1)
    elif ndim==3:
        shape=(n,-1,1)
    else:
        print("Not implemented!")
    weight=weight.reshape(*shape)
    return weight