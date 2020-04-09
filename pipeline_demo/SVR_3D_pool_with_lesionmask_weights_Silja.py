""" 

Resting state fMRI connectivity analysis with Support Vector Regression (SVR)

This script implements the method described in [1].

Overview of the method:
1. Parcellate the brain into ROIs according to Harvard-Oxford atlas
2. On each iteration, remove one ROI from data 1 and generate a linear
model based on timeseries of the remaining voxels
3. Predict the timeseries of the removed ROI with the linear model
4. Calculate correlation between the predicted timeseries and true timeseries in data 2
5. Repeat 1-4 using data 2 for training and data 1 for testing
6. Calculate correlation between model 1 and model 2 estimated from independent datasets
7. Map correlation and reprodicibility coefficients to ROIs in standard brain anatomy


Hanna Halme // 2015
Questions and comments: hanna.halme@hus.fi

[1] Craddock, R.C., Milham, M.P., & LaConte, S.M. (2013). Predicting intrinsic
brain activity. Neuroimage, 82, 127-136.

Later additions: 

8. Excludes NaN_ROIs (and symmetrical ROIs).
9. Applies lesion mask.
10. Saves voxel weighs that are used in the model to predict the acticity of a ROI under inquiry as .nii.gz images. Results are reported as images in folder 'coeffmaps1' and '2' in the same folder as SVR results.  
11.  Saves the number and percentage of voxels per ROI after mask inclusion as a .text file.

Riikka Ruuth / Simo Vanni / Silja Raty // 2019

"""
import os
import numpy as np
import nibabel
from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.io import savemat
import multiprocessing
import glob
import sys # SV addition for print 20.6.2018


# TODO: set parameters 

#subj_dirs = ['Revis_0004_rs', 'Revis_0005_rs', 'Revis_0007_rs', 'Revis_0009_rs', 'Revis_0011_rs', 'Revis_0012_rs', 'Revis_0014_rs', 'Revis_0015_rs', 'Revis_0016_rs', 'Revis_0017_rs', 'Revis_0020_rs']  #no old lesion
#subjects = [ 'Revis_0003_rs', 'Revis_0018_rs', 'Revis_0020_rs', 'Revis_0022_rs' #old lesion
subj_dirs = ['Revis_0003_rs']

# Set ROIs to be excluded
exclude_ROI = [15,16,17,18,21,22,27,28,29,30,49,50,53,54,65,66,67,68,73,74,75,76]

# Name of folders for data model images (Silja)
modeldir = 'coeffmaps'


# Process all patients in a directory
for k in range(0, len(subj_dirs)):
    for s in (['session1']):

	# SV addition for printing
	print "Starting analysis for directory " + subj_dirs[k] 
	orig_stdout = sys.stdout	
	logfilename = subj_dirs[k] + '_'+ s + '_log.txt'
	logfilehandle = open(logfilename, "w+")
	sys.stdout = logfilehandle

        subj = subj_dirs[k]
        data_path = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data/'+subj+'/'+ s 
	results_path = '/opt1/MR_data/Silja_Raty/New_rs/results/Unmasked_results_lesion/'+subj+'/'
        print "\n\nCurrently processing subject: ", subj, s

	# Create output folders for model data (Silja)
	if not (os.path.isdir(results_path + s +'/'+ modeldir +'1')): 
    	    os.mkdir(results_path + s +'/'+ modeldir +'1')
	if not (os.path.isdir(results_path + s +'/'+ modeldir +'2')): 
    	    os.mkdir(results_path + s +'/'+ modeldir +'2')
            
    	print "\n\nCurrently processing subject: ", subj, s
	

	# Preprocessed fMRI data 
        func1 = data_path +'/func1/warpall_global/corr_epi_trim_reoriented_calc_tproject_warp.nii.gz'
        func2 = data_path + '/func2/warpall_global/corr_epi_trim_reoriented_calc_tproject_warp.nii.gz'
	lesion_mask = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data/'+subj+'/session1/anat/nonlinear_reg/lesion_seg_warp.nii.gz' 
	#old_lesion_mask = glob.glob('/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data/'+subj+'/session1/anat/nonlinear_reg/old_lesion_seg_warp.nii.gz') #choose this one for ones with no old lesion
	old_lesion_mask = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data/'+subj+'/session1/anat/nonlinear_reg/old_lesion_seg_warp.nii.gz' #choose this one for ones with an old lesion

        print "Collecting timeseries data..."
        func1_data = nibabel.load(func1).get_data()
        func2_data = nibabel.load(func2).get_data()
        func1_data = np.nan_to_num(func1_data)
        func2_data = np.nan_to_num(func2_data)
        volnum = np.shape(func1_data)[-1]
	lesion_mask_data = nibabel.load(lesion_mask).get_data()
	#if old_lesion_mask: # remove when running patients with old lesion
	old_lesion_mask_data = nibabel.load(old_lesion_mask).get_data()
	lesion_mask_data = np.nan_to_num(lesion_mask_data)
	#if old_lesion_mask: # remove when running patients with old lesion
	old_lesion_mask_data = np.nan_to_num(old_lesion_mask_data)
	
        print "Running support vector regression..."
        # Number of ROIs in Harvard-Oxford atlas
        roin=96

        #linear kernel & other parameters according to Craddock's paper
        svr = SVR(kernel="linear", C=100.0, epsilon=0.1, max_iter=30000)
        scaler = StandardScaler()
        pipe = Pipeline([('scaler', scaler),('clf',svr)])

        corrcoef1 = np.zeros([roin])
        corrcoef2 = np.zeros([roin])

        reprod_coef = np.zeros([roin])

    	for i in range(0,roin):

	    if i+1 in exclude_ROI:
		continue

	    print "\nROI "+str(i+1)

            # Reload ROI data on every iteration
            ROI_atlas = nibabel.load('/usr/share/fsl/5.0/data/atlases/HarvardOxford/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz') #.get_data()
	    ROI_data = ROI_atlas.get_data()
	    #ROI_data_orig = ROI_atlas.get_data()
	    #mask_data = nibabel.load(results_path+'/binary_std_cortex_mask.nii.gz').get_data()     		    
	    
	    prev_n_vox = np.size(np.where(ROI_data==i+1),1)# previous number of voxels
	    #ROI_data = ROI_data_orig*mask_data
	    affine = ROI_atlas.affine # RR changed .get_affine() to .affine

	    # Define voxels belonging to lesion mask and set to zero
            ids3 = np.where(lesion_mask_data==1)
            ROI_data[ids3[0],ids3[1],ids3[2]] = 0

	    # Define voxels belonging to old lesion mask and set to zero
            #if old_lesion_mask: # remove when running patients with old lesion
	    ids3 = np.where(old_lesion_mask_data==2)
            ROI_data[ids3[0],ids3[1],ids3[2]] = 0

            # Define voxels belonging to a current ROI and set to zero
            ids = np.where(ROI_data==i+1)
            ROI_data[ids[0],ids[1], ids[2]] = 0

            # Define voxels belonging to excluded ROIs and set to zero
 	    for roi_to_remove in exclude_ROI:
    	        ROI_data[ROI_data==roi_to_remove] = 0

            ids2 = np.where(ROI_data>0)
    	
	    cur_n_vox = np.size(ids,1)# current (masked) number of voxels

	    if np.isnan(cur_n_vox) or cur_n_vox==0:
		
	    	print "Unmasked voxel count: "+str(prev_n_vox)
	    	print "Masked voxel count: 0"
	    	print "Voxel inclusion rate: 0"
	    	print "Mean 1: 0"	  
	    	print "Mean 2: 0"	  
	    	print "Std 1: 0"  
	    	print "Std 2: 0"	  
    
            	corrcoef1[i] = 0
            	corrcoef2[i] = 0
            	reprod_coef[i] = 0	

            	model1, model2, ROI_data = [], [], []

		continue
    	

            roi_timeseries1 = np.mean(func1_data[ids[0],ids[1], ids[2], :], axis=0)
            roi_timeseries2 = np.mean(func2_data[ids[0],ids[1], ids[2], :], axis=0)
    	
	    print "Unmasked voxel count: "+str(prev_n_vox)
	    print "Masked voxel count: "+str(cur_n_vox)
	    print "Voxel inclusion rate: "+str(100*cur_n_vox/prev_n_vox)+" %"
	    print "Mean 1: "+str(np.mean(roi_timeseries1))	  
	    print "Mean 2: "+str(np.mean(roi_timeseries2))	  
	    print "Std 1: "+str(np.std(roi_timeseries1))	  
	    print "Std 2: "+str(np.std(roi_timeseries2))	  
  
	    # Remove background and ROI voxels from training data
            brain_timeseries1 = func1_data[ids2[0], ids2[1], ids2[2], :]
            brain_timeseries2 = func2_data[ids2[0], ids2[1], ids2[2], :]
    	
            # Remaining ROI timeseries in fMRI2 = true timeseries	
            roi_timeseries1 =  (roi_timeseries1-np.mean(roi_timeseries1))/np.std(roi_timeseries1)
            roi_timeseries2 =  (roi_timeseries2-np.mean(roi_timeseries2))/np.std(roi_timeseries2)
    
            # Predict timeseries with SVR	
            temp = pipe.fit(brain_timeseries1.T, roi_timeseries1)
  	    pred_labels1 = temp.predict(brain_timeseries2.T)
            model1 = svr.coef_
            pred_labels1 =  (pred_labels1-np.mean(pred_labels1))/np.std(pred_labels1)
    
            # Same thing again, now use the other data for training...
            pred_labels2 = pipe.fit(brain_timeseries2.T, roi_timeseries2).predict(brain_timeseries1.T)
            model2 = svr.coef_
            pred_labels2 =  (pred_labels2-np.mean(pred_labels2))/np.std(pred_labels2)

            # Add to predicted timeseries 1
            corrcoef1[i] = np.corrcoef(pred_labels1, roi_timeseries2)[0,1]
            #print i+1, ' accuracy 1: ', corrcoef1[i]
            corrcoef2[i] = np.corrcoef(pred_labels2, roi_timeseries1)[0,1]
            #print i+1, ' accuracy 2: ', corrcoef2[i]
            reprod_coef[i] = np.corrcoef(model1, model2)[0,1]	
            #print i+1, ' reproducibility: ', reprod_coef[i]
	      	
	    # Print model weight maps (Silja)
	    predmap1 = np.zeros(np.shape(ROI_data))
	    predmap1[ids2] = model1
            predimg1 = nibabel.Nifti1Image(predmap1, affine)
	    nibabel.save(predimg1,results_path + s +'/'+ modeldir +'1/model1_ROI%02d.nii.gz'%(i+1))
	    predmap2 = np.zeros(np.shape(ROI_data))
	    predmap2[ids2] = model2
            predimg2 = nibabel.Nifti1Image(predmap2, affine)
            nibabel.save(predimg2,results_path + s +'/'+ modeldir +'2/model2_ROI%02d.nii.gz'%(i+1))
	    
            print i, '1: ', corrcoef1[i]
            print i, '2: ', corrcoef2[i]
            print reprod_coef[i]

            #model1, model2, ROI_data = [], [], []
        
            # Save results

    	    # Map correlation coefficients to a brain atlas
            print "Done "+subj+"! Mapping correlation coefficient to a brain volume..."
            corr_coef_map1 = np.zeros(np.shape(ROI_data))
            corr_coef_map2 = np.zeros(np.shape(ROI_data))
            reprod_coef_map = np.zeros(np.shape(ROI_data))

            for j in range(0,roin):
	        ids = np.where(ROI_data==j+1)
                corr_coef_map1[ids] = corrcoef1[j]
                corr_coef_map2[ids] = corrcoef2[j]
                reprod_coef_map[ids] = reprod_coef[j]
	     
		
    	
	# Save correlations as mat files
	savemat(results_path +'/'+ s + '/corrcoef_fmri1', {'corrcoef_fmri1':corrcoef1})
    	savemat(results_path +'/'+ s + '/corrcoef_fmri2', {'corrcoef_fmri2':corrcoef2})
    	savemat(results_path +'/'+ s + '/reprod_coef', {'reprod_coef':reprod_coef})
	
	print "SVR done with patient: ", subj

	sys.stdout = orig_stdout # SV addition for printing
	logfilehandle.close() # SV addition for printing
	
	print "SVR done with patient: " + subj + s # SV addition for printing

'''
if __name__ == '__main__':
    pool_size=10
    pool = multiprocessing.Pool(processes=pool_size)
    try:
	pool.map(SVR_analysis, subj_dirs)
    finally:
	pool.close()
	pool.join()
'''



