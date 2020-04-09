'''
Reads voxel-wise activation prediction model weights from .nii.gz images and stores average values for each MNI152-atlas ROI to excel

Columns in excel are ROIs that are predicted, rows are ROIs which make the prediction
'''

import os
import numpy as np
import nibabel
import xlwt


# TODO: set parameters 

# Data folders
datapath = '/opt1/MR_data/Silja_Raty/New_rs/results/Unmasked_results_lesion/'
subjects = ['Revis_0002_rs', 'Revis_0003_rs', 'Revis_0004_rs', 'Revis_0005_rs', 'Revis_0007_rs', 'Revis_0009_rs', 'Revis_0011_rs', 'Revis_0012_rs', 'Revis_0014_rs', 'Revis_0015_rs', 'Revis_0016_rs', 'Revis_0017_rs', 'Revis_0018_rs', 'Revis_0020_rs', 'Revis_0021_rs', 'Revis_0022_rs']
#subjects = ['Revis_0005_rs']
#subjects = ['Revis_2001_rs', 'Revis_2002_rs', 'Revis_2003_rs', 'Revis_2004_rs', 'Revis_2006_rs', 'Revis_2008_rs', 'Revis_2009_rs', 'Revis_2010_rs', 'Revis_2011_rs', 'Revis_2012_rs', 'Revis_2014_rs', 'Revis_2015_rs']
session = 'session2'

# ROI atlas
atlaspath = '/usr/share/fsl/5.0/data/atlases/HarvardOxford/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz'
labelspath = 'HO_labels.txt'
exclude_ROI = [14,15,16,17,20,21,26,27,28,29,48,49,52,53,64,65,66,67,72,73,74,75] # Indexing starting from 0, include both hemispheres

# Static choices
voxelmetrics = 'median'
patientmetrics = 'median'

# END TODO



# Read in MNI 152 ROI atlas and cortical ROI labels
ROI_atlas = nibabel.load(atlaspath).get_data()
cort_labels = np.loadtxt(labelspath, delimiter=',', usecols=(4,),dtype=str)

# Exclude excludable ROIs from ROI list
roin = 96
rois = [i for i in range(roin) if i not in exclude_ROI]
roin = len(rois)
cort_labels = np.delete(cort_labels, exclude_ROI)

# Create array for all weights
allweights = np.zeros([len(subjects), roin, roin])


# Read in weights of all subjects
for k in range(len(subjects)):
       	
    subj = subjects [k]
    print "\n\nCurrently processing subject: ", subj

    # Create excel for patient's prediction weigths
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Weights")

    # Read in all ROIs in ROI-list (rois)
    for i in range(roin):
        n = rois[i]	
        myweights = np.zeros([2,roin])

        # Write ROI names to excel
        sheet1.write(0,i+1, cort_labels[i])
        sheet1.write(i+1,0, cort_labels[i])

        for j in range(2):
	
            # Read in one prediction weight image
            weightfile = datapath+subj+'/'+session+'/coeffmaps'+str(j+1)+'/model'+str(j+1)+'_ROI'+str(n+1).zfill(2)+'.nii.gz'
	    weightdata = nibabel.load(weightfile).get_data()
            weightdata = np.nan_to_num(weightdata)

            # Average over all voxels in each ROI
            for m in range(roin):

		if i==m:
		    continue
		
                weights = weightdata[ROI_atlas==rois[m]+1]
		weights = weights[weights != 0] # Removes 0 voxels from mean calculation 

		# Leave 0 to ROIs with no voxels
		if np.shape(weights)[0] == 0:
		    continue

		if voxelmetrics == 'mean':	
		    myweights[j,m] = np.mean(weights)
		else:
		    myweights[j,m] = np.median(weights)

	   # Mean between the two scans in the same session
        allweights[k,i] = np.mean(myweights, axis=0)

        # Write voxel averages to excel
        for m in range(roin):
            sheet1.write(m+1,i+1, allweights[k,i,m])

    # Save this patient's weights to excel
    book.save(datapath+'/model_weights_patients/'+subj+'_'+session+'_prediction_model_weights.xls')

'''        
# Take average over all patients
if patientmetrics == 'mean':
	avgweights = np.mean(allweights, axis=0)	
else:
	avgweights = np.median(allweights, axis=0)	

# Create excel for average patient values
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Weights")

# Write ROI names and prediction model weights to excel
for i in range(roin):
    sheet1.write(0,i+1, cort_labels[i])
    sheet1.write(i+1,0, cort_labels[i])

    for j in range(roin):
        sheet1.write(j+1,i+1,avgweights[i,j])

# Save weights in excel
book.save(datapath+subj+'/'+session+'/average_prediction_model_weights_all.xls')
'''
