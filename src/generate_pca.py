import os
import time
import numpy as np
from sklearn.decomposition import PCA
from mydataset import subject_names_full, gesture_names_full, test_subjects_full

save_dir = '../pca_mean'
processed_dir = '../processed'

jnt_xyz = None

for subject in subject_names_full:
    if subject in test_subjects_full: continue # do not construct pca from test subjects
    time1 = time.time()
    for gesture in gesture_names_full:
        gesture_folder = os.path.join(processed_dir, subject, gesture)
        if not os.path.exists(gesture_folder): continue
        for seq_idx in os.listdir(os.path.join(processed_dir, subject, gesture)):
            gt_xyz = np.load(os.path.join(processed_dir, subject, gesture, seq_idx, 'gt_xyz.npy'))
            if jnt_xyz is None:
                jnt_xyz = gt_xyz
            else:
                jnt_xyz = np.vstack((jnt_xyz, gt_xyz))      
                
pca = PCA()
jnt_xyz = jnt_xyz.reshape(-1, 21 * 3)
pca_mean = np.mean(jnt_xyz)
pca.fit(jnt_xyz)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(os.path.join(save_dir, 'PCA_coeff.npy'), pca.components_)
np.save(os.path.join(save_dir, 'PCA_mean.npy'), pca_mean)

print('Done generate pca in {}s'.format(time.time() - time1))