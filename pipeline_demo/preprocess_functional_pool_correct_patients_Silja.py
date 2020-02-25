"""
Nipype-pipeline: fMRI preprocessing

Original version: Hanna Halme // 2015


Improvements for 2019 version version:
- preprocessed anatomical files can be read from another folder, set anat_path
- number of dummies to remove can be easily set by user to n_dummies parameter
- dummies deleted at the first step
- DRIFTER removed
- several new regressors integrated
- lesion masks can be integrated

If you add DRIFTER, check that you use the correct run_drifter_noSPM_Riikka.m file, which 
- accepts TR and n_dummies as input
- trims physiological signals according to n_dummies
- uses the correct TR in noise removal 


Updated version: Riikka Ruuth // 2019

"""

import nipype.interfaces.utility as util     # Utilities
import nipype.interfaces.fsl as fsl	     # fsl
import nipype.interfaces.afni as afni	     # afni
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.interfaces.nipy import SpaceTimeRealigner
from nipype.interfaces.matlab import MatlabCommand    # matlab
from nipype.interfaces.nipy.preprocess import Trim
from nipype.interfaces.utility import Function
import os
import numpy as np
import multiprocessing
import nibabel
import traceback
import glob


# TODO: set parameters correctly
data_path = '/opt1/MR_data/Silja_Raty/New_rs/image_data/'
results_path = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data/'
anat_path = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data/'


n_dummies = 6 # number of dummies to be deleted from the beginning of the time series
mytr = 2.0 # TR in seconds
myminf = 0.01 # minimum frequency of bandpass filter in Hz
mymaxf = 0.10 # maximum frequency of bandpass filter in Hz
censor_thr = 0.60 # threshold for censoring time points, change if needed

global_reg = True # Boolean, is global signal regression included or not

MNI_brain = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
ventricle_mask = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_strucseg_periph.nii.gz'

overwrite = True # overwrite pre-calculated regressor signals 

acquisitions = ['func1','func2']
#subjects = os.listdir(data_path)
subjects = ['Revis_0007_rs']

# END TODO


# get original working directory
owd = os.getcwd()


# help function, file name to list
def str2list(in_file): 
    return [in_file]

# normalize motion parameters from NIPY to millimeters and degrees
# output order [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
def normalize_motion_data(in_file):
    import os
    import numpy as np
    from nipype.utils.misc import normalize_mc_params
    
    mot_par_orig = np.loadtxt(in_file)
    mot_par_rad = np.zeros(mot_par_orig.shape)
   
    for i in range(len(mot_par_orig)): 
        mot_par_rad[i] = normalize_mc_params(mot_par_orig[i], 'NIPY') # normalized to radians

    scale = 180/np.pi
    mot_par_deg = mot_par_rad*[1,1,1,scale,scale,scale] # normalized to degrees

    out_file = 'normalized_motion_parameters.txt'
    np.savetxt(out_file, mot_par_deg)
    
    return os.path.abspath(out_file)

# scale given 1D data between -1 and 1, and calculate and scale quadratures
# in_file (string): name of input file
# muticol (boolean): are there multiple columns in in_file or not
def scale_and_quadrature(in_file, multicol):
        import os
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler

        col_sep = '\n' # separator for single column
        if multicol: col_sep = ' ' # column separator for multiple columns

        mydata = np.loadtxt(in_file, delimiter=col_sep)

        if not multicol:
            mydata = mydata.reshape(-1,1) # change to 2D for fit_transform function

        quadr_data = mydata*mydata

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled_data = scaler.fit_transform(mydata)
        scaled_quadr = scaler.fit_transform(quadr_data)

        out_file = 'scaled_parameters.txt'
        out_quadr_file = 'quadratures.txt'

        np.savetxt(out_file, scaled_data, delimiter=col_sep)
        np.savetxt(out_quadr_file, scaled_quadr, delimiter=col_sep)

        return os.path.abspath(out_file), os.path.abspath(out_quadr_file)


# create regressors for band pass filtering
# minf (float): minimum frequency [Hz]
# maxf (float): maximum frequency [Hz]
# example_file (string): example file of regressor parameters, used only to check the lenght
# tr (float): repetition time in MRI [s]
def bandpass(minf, maxf, example_file, tr):
        import os
        import numpy as np
        example_par = np.loadtxt(example_file)
        datalength = np.shape(example_par)[0]
        out_file = 'bandpass_rall.1D'
        os.system('1dBport -nodata '+str(datalength)+' '+str(tr)+' -band '+str(minf)+' '+str(maxf)+' -invert -nozero > '+out_file)
        return (os.path.abspath(out_file), 'bandpass')


# check that image has voxels on both hemispheres
# in_file (string): name of the image to be checked 
def check_bilateralism(in_file):
        import nibabel
        import numpy as np
        import os
        mymask = nibabel.load(in_file)
        mymask = mymask.get_data()
        errors = 0
        out_file = 'bilateralism_info.txt'
        fp =  open(out_file, 'w')
        fp.write("Checked file "+in_file+"\n")
        if not np.max(mymask[:mymask.shape[0]/2-1,:,:]):
            print "WARNING: " + in_file + " has no voxels on left hemisphere!"
            fp.write("WARNING: no voxels on left hemisphere!\n")
            errors = errors+1
        if not np.max(mymask[mymask.shape[0]/2:,:,:]):
            print "WARNING: " + in_file + " has no voxels on right hemisphere!"
            fp.write("WARNING: no voxels on right hemisphere!\n")
            errors = errors+1
        if not errors:
            fp.write("File is bilateral.\n")
        fp.close()
        return os.path.abspath(out_file)


# concatenate regressor time series and create correct input for deconvolve
# mot (string): file name of motion paramateres
# CSF (string): file name of CSF noise paramateres
# WM (string): file name of WM noise paramateres
# include_global (boolean): is global signal regression included or not
# glob (string): file name of global noise paramateres
# ending d means derivative, q quadrature and dq quadrature of the derivative
def concatenate_regressors(mot, motd, motq, motdq, CSF, CSFd, CSFq, CSFdq, WM, WMd, WMq, WMdq, include_global, glob, globd, globq, globdq):
        num_regressor = 32 # number of regressors
        reg_file_args = (
                " -stim_file 1  " + mot    + "'[0]' -stim_base 1  -stim_label 1  trans_x"
                " -stim_file 2  " + mot    + "'[1]' -stim_base 2  -stim_label 2  trans_y"
                " -stim_file 3  " + mot    + "'[2]' -stim_base 3  -stim_label 3  trans_z"
                " -stim_file 4  " + mot    + "'[3]' -stim_base 4  -stim_label 4  rot_x"
                " -stim_file 5  " + mot    + "'[4]' -stim_base 5  -stim_label 5  rot_y"
                " -stim_file 6  " + mot    + "'[5]' -stim_base 6  -stim_label 6  rot_z"
                " -stim_file 7  " + motd   + "'[0]' -stim_base 7  -stim_label 7  trans_x_d"
                " -stim_file 8  " + motd   + "'[1]' -stim_base 8  -stim_label 8  trans_y_d"
                " -stim_file 9  " + motd   + "'[2]' -stim_base 9  -stim_label 9  trans_z_d"
                " -stim_file 10 " + motd   + "'[3]' -stim_base 10 -stim_label 10 rot_x_d"
                " -stim_file 11 " + motd   + "'[4]' -stim_base 11 -stim_label 11 rot_y_d"
                " -stim_file 12 " + motd   + "'[5]' -stim_base 12 -stim_label 12 rot_z_d"
                " -stim_file 13 " + motq   + "'[0]' -stim_base 13 -stim_label 13 trans_x_q"
                " -stim_file 14 " + motq   + "'[1]' -stim_base 14 -stim_label 14 trans_y_q"
                " -stim_file 15 " + motq   + "'[2]' -stim_base 15 -stim_label 15 trans_z_q"
                " -stim_file 16 " + motq   + "'[3]' -stim_base 16 -stim_label 16 rot_x_q"
                " -stim_file 17 " + motq   + "'[4]' -stim_base 17 -stim_label 17 rot_y_q"
                " -stim_file 18 " + motq   + "'[5]' -stim_base 18 -stim_label 18 rot_z_q"
                " -stim_file 19 " + motdq  + "'[0]' -stim_base 19 -stim_label 19 trans_x_dq"
                " -stim_file 20 " + motdq  + "'[1]' -stim_base 20 -stim_label 20 trans_y_dq"
                " -stim_file 21 " + motdq  + "'[2]' -stim_base 21 -stim_label 21 trans_z_dq"
                " -stim_file 22 " + motdq  + "'[3]' -stim_base 22 -stim_label 22 rot_x_dq"
                " -stim_file 23 " + motdq  + "'[4]' -stim_base 23 -stim_label 23 rot_y_dq"
                " -stim_file 24 " + motdq  + "'[5]' -stim_base 24 -stim_label 24 rot_z_dq"
                " -stim_file 25 " + CSF    + "'[0]' -stim_base 25 -stim_label 25 CSF"
                " -stim_file 26 " + CSFd   + "'[0]' -stim_base 26 -stim_label 26 CSF_d"
                " -stim_file 27 " + CSFq   + "'[0]' -stim_base 27 -stim_label 27 CSF_q"
                " -stim_file 28 " + CSFdq  + "'[0]' -stim_base 28 -stim_label 28 CSF_dq"
                " -stim_file 29 " + WM     + "'[0]' -stim_base 29 -stim_label 29 WM"
                " -stim_file 30 " + WMd    + "'[0]' -stim_base 30 -stim_label 30 WM_d"
                " -stim_file 31 " + WMq    + "'[0]' -stim_base 31 -stim_label 31 WM_q"
                " -stim_file 32 " + WMdq   + "'[0]' -stim_base 32 -stim_label 32 WM_dq"
        )
        if include_global:
            num_regressor = 36
            reg_file_args_2 = (
                " -stim_file 33 " + glob   + "'[0]' -stim_base 33 -stim_label 33 glob"
                " -stim_file 34 " + globd  + "'[0]' -stim_base 34 -stim_label 34 glob_d"
                " -stim_file 35 " + globq  + "'[0]' -stim_base 35 -stim_label 35 glob_q"
                " -stim_file 36 " + globdq + "'[0]' -stim_base 36 -stim_label 36 glob_dq"
            )
            reg_file_args = reg_file_args + reg_file_args_2
        reg_file_args = (" -num_stimts "+str(num_regressor)) + reg_file_args

        fp = open('regressors.txt', 'w')
        fp.write(reg_file_args)
        fp.close()
        return reg_file_args


# preprocessing pipeline
# i (string): name of currect subject
def prepro_func(i):
	try:
		subj = i
		for s in (['session2']):

		    # Define input files: 2xfMRI + 1xMPRAGE
		    func1 = data_path + subj +'/Functional_scans/'+s[:-2]+s[-1]+'_a/epi.nii.gz' #choose this for patients
		    func2 = data_path + subj +'/Functional_scans/'+s[:-2]+s[-1]+'_b/epi.nii.gz' #choose this for patients
		    #anat = glob.glob(anat_path + subj +'/'+ s + '/anat/reorient/anat_*.nii.gz') #choose this for session 1   
		    lesion_mask_file = anat_path + subj +'/session1/anat/reorient/lesion_seg.nii.gz'    
		    old_lesion_mask_file = glob.glob(anat_path + subj +'/session1/anat/reorient/old_lesion_seg.nii.gz') #choose this for ones with no old lesion  
		    #old_lesion_mask_file = anat_path + subj +'/session1/anat/reorient/old_lesion_seg.nii.gz' #choose this for ones with old lesion 
		    anat = glob.glob(anat_path + subj +'/'+ s + '/anat/anat2hr/anat_*.nii.gz') #choose this for sessions 2 and 3
		    anat_CSF = glob.glob(anat_path + subj +'/session1/seg_anat/segmentation/anat_*_pve_0.nii.gz') # don't change, same for all sessions
		    anat_WM = glob.glob(anat_path + subj +'/session1/seg_anat/segmentation/anat_*_pve_2.nii.gz') # don't change, same for all sessions
		    anat_GM = glob.glob(anat_path + subj +'/session1/seg_anat/segmentation/anat_*_pve_1.nii.gz') # don't change, same for all sessions
		    anat2MNI_fieldwarp = glob.glob(anat_path+subj+'/session1/anat/nonlinear_reg/anat_*_fieldwarp.nii.gz') # don't change, same for all sessions

		    if not os.path.isdir(data_path+subj+'/'+s): # No data exists
			continue

		    if not os.path.isfile(func1):
			print '1. functional file '+func1+' not found. Skipping!'
			continue

		    if not os.path.isfile(func2):
			print '2. functional file '+func2+' not found. Skipping!'
			continue

		    if not anat:
			print 'Preprocessed anatomical file not found. Skipping!'
			continue
		    if len(anat)>1:
			print 'WARNING: found multiple files of preprocessed anatomical image!'
			continue
		    anat = anat[0]

		    if not anat2MNI_fieldwarp:
			print 'Anatomical registration to MNI152-space field file not found. Skipping!'
			continue
		    if len(anat2MNI_fieldwarp)>1:
			print 'WARNING: found multiple files of anat2MNI fieldwarp!'
			continue
		    anat2MNI_fieldwarp = anat2MNI_fieldwarp[0]

		    if not anat_CSF:
		        anat_CSF = glob.glob(anat_path + subj +'/'+ s + '/seg_anat/segmentation/anat_*_pve_0.nii.gz')
		    	if not anat_CSF:
			    print 'Anatomical segmentation CSF file not found. Skipping!'
			    continue
		    if len(anat_CSF)>1:
			print 'WARNING: found multiple files of anatomical CSF file!'
			continue
		    anat_CSF = anat_CSF[0]

		    if not anat_WM:
		        anat_WM = glob.glob(anat_path + subj +'/'+ s + '/seg_anat/segmentation/anat_*_pve_2.nii.gz')
		    	if not anat_WM:
			    print 'Anatomical segmentation WM file not found. Skipping!'
			    continue
		    if len(anat_WM)>1:
			print 'WARNING: found multiple files of anatomical WM file!'
			continue
		    anat_WM = anat_WM[0]

		    if not anat_GM:
		        anat_GM = glob.glob(anat_path + subj +'/'+ s + '/seg_anat/segmentation/anat_*_pve_1.nii.gz')
		    	if not anat_GM:
			    print 'Anatomical segmentation GM file not found. Skipping!'
			    continue
		    if len(anat_GM)>1:
			print 'WARNING: found multiple files of anatomical GM file!'
			continue
		    anat_GM = anat_GM[0]

		    if not os.path.isdir(results_path + subj):
			os.mkdir(results_path+subj)

		    if not os.path.isdir(results_path + subj+'/'+s):
			os.mkdir(results_path+subj+'/'+s)

		    for data in acquisitions:

			os.chdir(results_path+subj+'/'+s)
		        print "Currently processing subject: "+ subj+'/'+s +' '+data

		        #Initialize workflows
		        workflow = pe.Workflow(name = data)

		        workflow.base_dir = '.'
			inputnode = pe.Node(interface=util.IdentityInterface(fields=['source_file']), name='inputspec')
		        outputnode = pe.Node(interface=util.IdentityInterface(fields=['result_func']), name='outputspec')

		        if data == 'func1':
		            inputnode.inputs.source_file = func1
		        else:
		            inputnode.inputs.source_file = func2		

			# Remove n_dummies first volumes
		        trim = pe.Node(interface=Trim(begin_index=n_dummies), name='trim')
		        workflow.connect(inputnode, 'source_file', trim, 'in_file')

		        # Motion correction + slice timing correction
		        realign4d = pe.Node(interface=SpaceTimeRealigner(), name='realign4d')
		        #realign4d.inputs.ignore_exception=True
		        realign4d.inputs.slice_times='asc_alt_siemens'

	                realign4d.inputs.slice_info = 2 # horizontal slices
	                realign4d.inputs.tr = mytr # TR in seconds
	                workflow.connect(trim, 'out_file', realign4d, 'in_file')

		        # Reorient
		        #deoblique = pe.Node(interface=afni.Warp(deoblique=True, outputtype='NIFTI_GZ'), name='deoblique') #leave out if you don't need this
		        #workflow.connect(realign4d, 'out_file', deoblique, 'in_file')
		        reorient = pe.Node(interface=fsl.Reorient2Std(output_type='NIFTI_GZ'), name='reorient')
		        workflow.connect(realign4d, 'out_file', reorient, 'in_file')
		        
			# AFNI skullstrip and mean image skullstrip
		        tstat1 = pe.Node(interface=afni.TStat(args='-mean',outputtype="NIFTI_GZ"), name='tstat1')
		        automask = pe.Node(interface=afni.Automask(dilate=1,outputtype="NIFTI_GZ"), name='automask')
		        skullstrip = pe.Node(interface=afni.Calc(expr = 'a*b',outputtype="NIFTI_GZ"), name='skullstrip')
		        tstat2 = pe.Node(interface=afni.TStat(args='-mean',outputtype="NIFTI_GZ"), name='tstat2')

		        workflow.connect(reorient, 'out_file', tstat1, 'in_file')
		        workflow.connect(tstat1, 'out_file', automask, 'in_file')
		        workflow.connect(automask, 'out_file', skullstrip, 'in_file_b')
		        workflow.connect(reorient, 'out_file', skullstrip, 'in_file_a')
		        workflow.connect(skullstrip, 'out_file', tstat2, 'in_file')

			# Register to anatomical space #can be changed 
		        #mean2anat = pe.Node(fsl.FLIRT(bins=40, cost='normmi', dof=7, interp='nearestneighbour', searchr_x=[-180,180], searchr_y=[-180,180], searchr_z=[-180,180]), name='mean2anat')
			mean2anat = pe.Node(fsl.FLIRT(bins=40, cost='normmi', dof=7, interp='nearestneighbour'), name='mean2anat')
			#mean2anat = pe.Node(fsl.FLIRT(no_search=True), name='mean2anat')
		        mean2anat.inputs.reference = anat
		        workflow.connect(tstat2, 'out_file', mean2anat, 'in_file')
		
			# Transform mean functional image
		        warpmean = pe.Node(interface=fsl.ApplyWarp(), name='warpmean')
		        warpmean.inputs.ref_file = MNI_brain
			warpmean.inputs.field_file =  anat2MNI_fieldwarp
		        workflow.connect(mean2anat, 'out_matrix_file', warpmean, 'premat')
		        workflow.connect(tstat2, 'out_file', warpmean, 'in_file')


			# ----- inversion matrix and eroded brain mask for regression -----

			# create inverse matrix from mean2anat registration
			invmat = pe.Node(fsl.ConvertXFM(),name='invmat')
			invmat.inputs.invert_xfm = True
			workflow.connect(mean2anat, 'out_matrix_file', invmat, 'in_file')
			
			# erode functional brain mask
			erode_brain = pe.Node(fsl.ImageMaths(), name='erode_brain')
			erode_brain.inputs.args = '-kernel boxv 3 -ero' 
			workflow.connect(automask, 'out_file', erode_brain, 'in_file')
		
			# register GM mask to functional image space, this is done for quality control
			reg_GM = pe.Node(fsl.preprocess.ApplyXFM(), name='register_GM')
			reg_GM.inputs.apply_xfm = True
			reg_GM.inputs.in_file = anat_GM
			workflow.connect(tstat2, 'out_file', reg_GM, 'reference')
			workflow.connect(invmat, 'out_file', reg_GM, 'in_matrix_file')


			# --------- motion regression and censor signals ------------------

			# normalize motion parameters
			norm_motion = pe.Node(interface=Function(input_names=['in_file'], 
								 output_names=['out_file'], 
								 function=normalize_motion_data), 
								 name='normalize_motion')
			workflow.connect(realign4d, 'par_file', norm_motion, 'in_file')	

			# create censor file, for censoring motion 
			get_censor = pe.Node(afni.OneDToolPy(), name='motion_censors')
			get_censor.inputs.set_nruns = 1
			get_censor.inputs.censor_motion = (censor_thr, 'motion')
			get_censor.inputs.show_censor_count = True
			if overwrite: get_censor.inputs.args = '-overwrite'
			workflow.connect(norm_motion, 'out_file', get_censor, 'in_file')

			# compute motion parameter derivatives (for use in regression)
			deriv_motion = pe.Node(afni.OneDToolPy(), name='deriv_motion')
			deriv_motion.inputs.set_nruns = 1
			deriv_motion.inputs.derivative = True
			if overwrite: deriv_motion.inputs.args = '-overwrite'
			deriv_motion.inputs.out_file = 'motion_derivatives.txt'
			workflow.connect(norm_motion, 'out_file', deriv_motion, 'in_file')

			# scale motion parameters and get quadratures
			quadr_motion = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
							   output_names=['out_file', 'out_quadr_file'], 
							   function=scale_and_quadrature), 
							   name='quadr_motion')
			quadr_motion.inputs.multicol = True
			workflow.connect(norm_motion, 'out_file', quadr_motion, 'in_file')	

			# scale motion derivatives and get quadratures
			quadr_motion_deriv = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
								 output_names=['out_file', 'out_quadr_file'], 
								 function=scale_and_quadrature), 
								 name='quadr_motion_deriv')
			quadr_motion_deriv.inputs.multicol = True
			workflow.connect(deriv_motion, 'out_file', quadr_motion_deriv, 'in_file')	


			# -------- CSF regression signals --------------- 

			# threshold and erode CSF mask
			erode_CSF_mask = pe.Node(fsl.ImageMaths(), name='erode_CSF_mask')
			erode_CSF_mask.inputs.args = '-thr 0.5 -kernel boxv 3 -ero' 
			erode_CSF_mask.inputs.in_file = anat_CSF

			# register CSF mask to functional image space
			reg_CSF_mask = pe.Node(fsl.preprocess.ApplyXFM(), name='register_CSF_mask')
			reg_CSF_mask.inputs.apply_xfm = True
			workflow.connect(tstat2, 'out_file', reg_CSF_mask, 'reference')
			workflow.connect(invmat, 'out_file', reg_CSF_mask, 'in_matrix_file')
			
			# inverse lesion mask and remove it from CSF mask #remove this if you don't have a lesion mask
			inverse_lesion_mask = pe.Node(fsl.ImageMaths(), name='inverse_lesion_mask')
			inverse_lesion_mask.inputs.args = '-add 1 -rem 2'
			inverse_lesion_mask.inputs.in_file = lesion_mask_file
			rem_lesion = pe.Node(fsl.ImageMaths(), name='remove_lesion')
			workflow.connect(erode_CSF_mask, 'out_file', rem_lesion, 'in_file')
			workflow.connect(inverse_lesion_mask, 'out_file', rem_lesion, 'mask_file')
			'''
			# Transform lesion mask to MNI152 space #remove if lesion masks are already in MNI152 space
		        warp_lesion = pe.Node(interface=fsl.ApplyWarp(), name='warp_lesion')
		        warp_lesion.inputs.ref_file = MNI_brain
		        warp_lesion.inputs.field_file = anat2MNI_fieldwarp
			warp_lesion.inputs.in_file = lesion_mask_file
			warp_lesion.inputs.out_file = anat_path + subj +'/'+ s + '/anat/nonlinear_reg/lesion_seg_warp.nii.gz'
			warp_lesion.run()
			'''
			
			# inverse old lesion mask and remove it from CSF mask #remove this if you don't have a lesion mask
			if old_lesion_mask_file:
			    inverse_old_lesion_mask = pe.Node(fsl.ImageMaths(), name='inverse_old_lesion_mask')
			    inverse_old_lesion_mask.inputs.args = '-add 1 -rem 3'
			    #inverse_old_lesion_mask.inputs.in_file = old_lesion_mask_file[0]
			    inverse_old_lesion_mask.inputs.in_file = old_lesion_mask_file
			    rem_old_lesion = pe.Node(fsl.ImageMaths(), name='remove_old_lesion')
			    workflow.connect(rem_lesion, 'out_file', rem_old_lesion, 'in_file')
			    workflow.connect(inverse_old_lesion_mask, 'out_file', rem_old_lesion, 'mask_file')
			    workflow.connect(rem_old_lesion, 'out_file', reg_CSF_mask, 'in_file')
			    '''
			    # Transform old lesion mask to MNI152 space #remove if lesion masks are already in MNI152 space
		            warp_old_lesion = pe.Node(interface=fsl.ApplyWarp(), name='warp_old_lesion')
		            warp_old_lesion.inputs.ref_file = MNI_brain
		            warp_old_lesion.inputs.field_file = anat2MNI_fieldwarp
			    warp_old_lesion.inputs.in_file = old_lesion_mask_file
			    warp_old_lesion.inputs.out_file = anat_path + subj +'/'+ s + '/anat/nonlinear_reg/old_lesion_seg_warp.nii.gz'
			    warp_old_lesion.run()
			    '''
			
			else:
			    workflow.connect(rem_lesion, 'out_file', reg_CSF_mask, 'in_file')
			
			# threshold CSF mask and intersect with functional brain mask
			thr_CSF_mask = pe.Node(fsl.ImageMaths(), name='threshold_CSF_mask')
			thr_CSF_mask.inputs.args = '-thr 0.25' 
			workflow.connect(reg_CSF_mask, 'out_file', thr_CSF_mask, 'in_file')
			workflow.connect(erode_brain, 'out_file', thr_CSF_mask, 'mask_file')

			# extract CSF values
			get_CSF_noise = pe.Node(fsl.ImageMeants(), name='get_CSF_noise')
			workflow.connect(skullstrip, 'out_file', get_CSF_noise, 'in_file')
			workflow.connect(thr_CSF_mask, 'out_file', get_CSF_noise, 'mask')

			# compute CSF noise derivatives
			deriv_CSF = pe.Node(afni.OneDToolPy(), name='deriv_CSF')
			deriv_CSF.inputs.set_nruns = 1
			deriv_CSF.inputs.derivative = True
			if overwrite: deriv_CSF.inputs.args = '-overwrite'
			deriv_CSF.inputs.out_file = 'CSF_derivatives.txt'
			workflow.connect(get_CSF_noise, 'out_file', deriv_CSF, 'in_file')

			# scale SCF noise and get quadratures
			quadr_CSF = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
							output_names=['out_file', 'out_quadr_file'], 
							function=scale_and_quadrature), 
							name='quadr_CSF')
			quadr_CSF.inputs.multicol = False
			workflow.connect(get_CSF_noise, 'out_file', quadr_CSF, 'in_file')	

			# scale CSF noise derivatives and get quadratures
			quadr_CSF_deriv = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
							      output_names=['out_file', 'out_quadr_file'], 
							      function=scale_and_quadrature), 
							      name='quadr_CSF_deriv')
			quadr_CSF_deriv.inputs.multicol = False
			workflow.connect(deriv_CSF, 'out_file', quadr_CSF_deriv, 'in_file')	


			# -------- WM regression signals -----------------

			# threshold and erode WM mask
			erode_WM_mask = pe.Node(fsl.ImageMaths(), name='erode_WM_mask')
			erode_WM_mask.inputs.args = '-thr 0.5 -kernel boxv 7 -ero' 
			erode_WM_mask.inputs.in_file = anat_WM

			# registrer WM mask to functional image space
			reg_WM_mask = pe.Node(fsl.preprocess.ApplyXFM(), name='register_WM_mask')
			reg_WM_mask.inputs.apply_xfm = True
			workflow.connect(tstat2, 'out_file', reg_WM_mask, 'reference')
			workflow.connect(invmat, 'out_file', reg_WM_mask, 'in_matrix_file')
			workflow.connect(erode_WM_mask, 'out_file', reg_WM_mask, 'in_file')

                        # create inverse nonlinear registration MNI2anat
                        invwarp = pe.Node(fsl.InvWarp(output_type='NIFTI_GZ'), name='invwarp')
                        invwarp.inputs.warp = anat2MNI_fieldwarp
			invwarp.inputs.reference = anat

                        # transform ventricle mask to functional space
                        reg_ventricles = pe.Node(fsl.ApplyWarp(), name='register_ventricle_mask')
                        reg_ventricles.inputs.in_file = ventricle_mask
			workflow.connect(tstat2, 'out_file', reg_ventricles, 'ref_file')
                        workflow.connect(invwarp, 'inverse_warp', reg_ventricles, 'field_file')
			workflow.connect(invmat, 'out_file', reg_ventricles, 'postmat')

			# threshold WM mask and intersect with functional brain mask
			thr_WM_mask = pe.Node(fsl.ImageMaths(), name='threshold_WM_mask')
			thr_WM_mask.inputs.args = '-thr 0.25' 
			workflow.connect(reg_WM_mask, 'out_file', thr_WM_mask, 'in_file')
			workflow.connect(erode_brain, 'out_file', thr_WM_mask, 'mask_file')

                        # remove ventricles from WM mask
                        exclude_ventricles = pe.Node(fsl.ImageMaths(), name='exclude_ventricles')
			workflow.connect(thr_WM_mask, 'out_file', exclude_ventricles, 'in_file')
			workflow.connect(reg_ventricles, 'out_file', exclude_ventricles, 'mask_file')

                        # check that WM is collected from both hemispheres
			check_WM_bilat = pe.Node(interface=Function(input_names=['in_file'],
                                                                output_names=['errors'],
                                                                function=check_bilateralism),
                                                                name='check_WM_bilateralism')
                        workflow.connect(exclude_ventricles, 'out_file', check_WM_bilat, 'in_file')

			# extract WM values
			get_WM_noise = pe.Node(fsl.ImageMeants(), name='get_WM_noise')
			workflow.connect(skullstrip, 'out_file', get_WM_noise, 'in_file')
			workflow.connect(exclude_ventricles, 'out_file', get_WM_noise, 'mask')

			# compute WM noise derivatives
			deriv_WM = pe.Node(afni.OneDToolPy(), name='deriv_WM')
			deriv_WM.inputs.set_nruns = 1
			deriv_WM.inputs.derivative = True
			if overwrite: deriv_WM.inputs.args = '-overwrite'
			deriv_WM.inputs.out_file = 'WM_derivatives.txt'
			workflow.connect(get_WM_noise, 'out_file', deriv_WM, 'in_file')

			# scale WM noise and get quadratures
			quadr_WM = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
						       output_names=['out_file', 'out_quadr_file'], 
						       function=scale_and_quadrature), 
						       name='quadr_WM')
			quadr_WM.inputs.multicol = False
			workflow.connect(get_WM_noise, 'out_file', quadr_WM, 'in_file')	

			# scale WM noise derivatives and get quadratures
			quadr_WM_deriv = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
							     output_names=['out_file', 'out_quadr_file'], 
							     function=scale_and_quadrature), 
							     name='quadr_WM_deriv')
			quadr_WM_deriv.inputs.multicol = False
			workflow.connect(deriv_WM, 'out_file', quadr_WM_deriv, 'in_file')	


			# ---------- global regression signals ----------------
			
			if global_reg:
			    # register anatomical whole brain mask to functional image space
			    reg_glob_mask = pe.Node(fsl.preprocess.ApplyXFM(), name='register_global_mask')
			    reg_glob_mask.inputs.apply_xfm = True
			    reg_glob_mask.inputs.in_file = anat
			    workflow.connect(tstat2, 'out_file', reg_glob_mask, 'reference')
			    workflow.connect(invmat, 'out_file', reg_glob_mask, 'in_matrix_file')
	
			    # threshold anatomical brain mask and intersect with functional brain mask
			    thr_glob_mask = pe.Node(fsl.ImageMaths(), name='threshold_global_mask')
			    thr_glob_mask.inputs.args = '-thr -0.1'
			    workflow.connect(reg_glob_mask, 'out_file', thr_glob_mask, 'in_file')
			    workflow.connect(erode_brain, 'out_file', thr_glob_mask, 'mask_file')

			    # extract global signal values
			    get_glob_noise = pe.Node(fsl.ImageMeants(), name='get_global_noise')
			    workflow.connect(skullstrip, 'out_file', get_glob_noise, 'in_file')
			    workflow.connect(thr_glob_mask, 'out_file', get_glob_noise, 'mask')

			    # compute global noise derivative
			    deriv_glob = pe.Node(afni.OneDToolPy(), name='deriv_global')
			    deriv_glob.inputs.set_nruns = 1
			    deriv_glob.inputs.derivative = True
			    if overwrite: deriv_glob.inputs.args = '-overwrite'
			    deriv_glob.inputs.out_file = 'global_derivatives.txt'
			    workflow.connect(get_glob_noise, 'out_file', deriv_glob, 'in_file')

			    # scale global noise and get quadratures
			    quadr_glob = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
							     output_names=['out_file', 'out_quadr_file'], 
							     function=scale_and_quadrature), 
							     name='quadr_glob')
			    quadr_glob.inputs.multicol = False
			    workflow.connect(get_glob_noise, 'out_file', quadr_glob, 'in_file')	

			    # scale global noise derivatives and get quadratures
			    quadr_glob_deriv = pe.Node(interface=Function(input_names=['in_file', 'multicol'], 
								   output_names=['out_file', 'out_quadr_file'], 
								   function=scale_and_quadrature), 
								   name='quadr_glob_deriv')
			    quadr_glob_deriv.inputs.multicol = False
			    workflow.connect(deriv_glob, 'out_file', quadr_glob_deriv, 'in_file')	


			# ---------- regression matrix ----------

			# create bandpass regressors, can not be easily implemented to workflow
			get_bandpass = pe.Node(interface=Function(input_names=['minf', 'maxf', 'example_file', 'tr'], 
							   output_names=['out_tuple'], 
							   function=bandpass),
							   name='bandpass_regressors')
			get_bandpass.inputs.minf = myminf
			get_bandpass.inputs.maxf = mymaxf
			get_bandpass.inputs.tr = mytr
			workflow.connect(norm_motion, 'out_file', get_bandpass, 'example_file')

			# concatenate regressor time series
			cat_reg_name = 'cat_regressors'
			if global_reg: cat_reg_name = cat_reg_name+'_global'
                        cat_reg = pe.Node(interface=Function(input_names=['mot', 'motd', 'motq', 'motdq',
                                                                        'CSF', 'CSFd', 'CSFq', 'CSFdq',
                                                                        'WM', 'WMd', 'WMq', 'WMdq',
                                                                        'include_global', 'glob', 'globd', 'globq', 'globdq'],
                                                           output_names=['reg_file_args'],
                                                           function=concatenate_regressors),
                                                           name=cat_reg_name)
                        cat_reg.inputs.include_global = global_reg
                        workflow.connect(quadr_motion, 'out_file', cat_reg, 'mot')
                        workflow.connect(quadr_motion_deriv, 'out_file', cat_reg, 'motd')
                        workflow.connect(quadr_motion, 'out_quadr_file', cat_reg, 'motq')
                        workflow.connect(quadr_motion_deriv, 'out_quadr_file', cat_reg, 'motdq')
                        workflow.connect(quadr_CSF, 'out_file', cat_reg, 'CSF')
                        workflow.connect(quadr_CSF_deriv, 'out_file', cat_reg, 'CSFd')
                        workflow.connect(quadr_CSF, 'out_quadr_file', cat_reg, 'CSFq')
                        workflow.connect(quadr_CSF_deriv, 'out_quadr_file', cat_reg, 'CSFdq')
                        workflow.connect(quadr_WM, 'out_file', cat_reg, 'WM')
                        workflow.connect(quadr_WM_deriv, 'out_file', cat_reg, 'WMd')
                        workflow.connect(quadr_WM, 'out_quadr_file', cat_reg, 'WMq')
                        workflow.connect(quadr_WM_deriv, 'out_quadr_file', cat_reg, 'WMdq')
                        if global_reg:
                            workflow.connect(quadr_glob, 'out_file', cat_reg, 'glob')
                            workflow.connect(quadr_glob_deriv, 'out_file', cat_reg, 'globd')
                            workflow.connect(quadr_glob, 'out_quadr_file', cat_reg, 'globq')
                            workflow.connect(quadr_glob_deriv, 'out_quadr_file', cat_reg, 'globdq')
                        else:
			    cat_reg.inputs.glob = None
			    cat_reg.inputs.globd = None
			    cat_reg.inputs.globq = None
			    cat_reg.inputs.globdq = None

                        # create regression matrix 
			deconvolve_name = 'deconvolve'
			if global_reg: deconvolve_name = deconvolve_name+'_global'
                        deconvolve = pe.Node(afni.Deconvolve(), name=deconvolve_name)
                        deconvolve.inputs.polort = 2 # contstant, linear and quadratic background signals removed
                        deconvolve.inputs.fout = True
                        deconvolve.inputs.tout = True
                        deconvolve.inputs.x1D_stop = True
                        deconvolve.inputs.force_TR = mytr
                        workflow.connect(cat_reg, 'reg_file_args', deconvolve, 'args')
                        workflow.connect(get_bandpass, 'out_tuple', deconvolve, 'ortvec')
                        workflow.connect([(skullstrip, deconvolve, [(('out_file', str2list), 'in_files')] )])

			# regress out motion and other unwanted signals
			tproject_name = 'tproject'
			if global_reg: tproject_name = tproject_name+'_global'
			tproject = pe.Node(afni.TProject(outputtype="NIFTI_GZ"), name=tproject_name)
			tproject.inputs.TR = mytr
			tproject.inputs.polort = 0 # use matrix created with 3dDeconvolve, higher order polynomials not needed
			tproject.inputs.cenmode = 'NTRP' # interpolate removed time points
			workflow.connect(get_censor, 'out_file', tproject, 'censor')
			workflow.connect(skullstrip, 'out_file', tproject, 'in_file')
			workflow.connect(automask, 'out_file', tproject, 'mask')
			workflow.connect(deconvolve, 'x1D', tproject, 'ort')

		        # Transform all images
			warpall_name = 'warpall'
			if global_reg: warpall_name = warpall_name+'_global'
		        warpall = pe.Node(interface=fsl.ApplyWarp(), name=warpall_name)
		        warpall.inputs.ref_file = MNI_brain
		        warpall.inputs.field_file = anat2MNI_fieldwarp
		        workflow.connect(mean2anat, 'out_matrix_file', warpall, 'premat')
		        workflow.connect(tproject, 'out_file', warpall, 'in_file')
		        workflow.connect(warpall, 'out_file', outputnode, 'result_func')

			# Run workflow
		        workflow.write_graph()
		        workflow.run()
			
	        print "FUNCTIONAL PREPROCESSING DONE! Results in ", results_path+subj+'/'+s
	except:
	    print "Error with patient: ", subj
	    traceback.print_exc()

os.chdir(owd) # return to original working directory

if __name__ == '__main__':
	pool_size=10
	pool = multiprocessing.Pool(processes=pool_size)
	try:
		pool.map(prepro_func, subjects)
	finally:
		pool.close()
		pool.join()




