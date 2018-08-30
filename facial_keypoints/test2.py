from keras.models import load_model
from guassian import load2d
import numpy as np
import pandas as pd
from transfer import transfer_target

sigma=5
X_test, y_test,_,_ = load2d(test=True,sigma=sigma)
output_height,output_width=96,96
nClasses=15
BATCH_SIZE=64

def get_test_batch(X_test):
    for start in range(0,len(X_test),BATCH_SIZE):
        X_batch=[]
        end=min(start+BATCH_SIZE,len(X_test))
        X_batch.append(X_test[start:end])
        yield X_batch

model=load_model('weights.best_00.hdf5')
y_pred=model.predict(X_test)
y_pred=y_pred.reshape(-1,output_height,output_width,nClasses)
y_pred=transfer_target(y_pred,thresh=0,n_points=25)

df=pd.DataFrame(y_pred,columns=['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
       'right_eye_center_y', 'left_eye_inner_corner_x',
       'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
       'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
       'right_eye_inner_corner_y', 'right_eye_outer_corner_x',
       'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
       'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x',
       'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
       'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
       'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
       'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x',
       'mouth_right_corner_y', 'mouth_center_top_lip_x',
       'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
       'mouth_center_bottom_lip_y'])
df['ImageId']=np.arange(1783)+1
_id=df['ImageId']
df=df.drop('ImageId',axis=1)
df.insert(0,'ImageId',_id)
# df.to_csv('result.csv',index=False)

idlookup=pd.read_csv('IdLookupTable.csv')
for i in range(27124):
    idlookup.iloc[i,3]=float(df[df['ImageId']==idlookup.loc[i]['ImageId']][idlookup.loc[i]['FeatureName']])

idlookup.drop(['ImageId','FeatureName'],axis=1,inplace=True)
idlookup.to_csv('result3.csv',index=False)