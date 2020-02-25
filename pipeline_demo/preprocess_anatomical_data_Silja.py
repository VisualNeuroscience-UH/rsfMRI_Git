""" 
Nipype-pipeline: anatomical preprocessing

Anatomical data preprocessing steps:

1. Transform slices (from oblique to axial orientation and) to FSL std orientation
2. Skull strip (Can be done with BET before entering the pipeline. In that case, leave out.)
3. Tissue segmentation 
4. Register to MNI152 standard template with linear and nonlinear registration

Running in terminal:
ipython
(in python terminal):
%run preprocess_anatomical_data_Silja.py

Hanna Halme // 2015
Questions and comments: hanna.halme@hus.fi

Updated version: Riikka Ruuth // 2019

"""
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl	     # fsl
import nipype.interfaces.afni as afni	     # afni
import copy
from nipype.interfaces.base import File, traits
import nipype.pipeline.engine as pe          # pypeline engine
import os
import numpy as np
import multiprocessing
import traceback

data_path = '/opt1/MR_data/Silja_Raty/New_rs/image_data/'
results_path = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data_test2/'

subj_dirs = ['Revis_0002_rs']

# Reference brain images
ref_brain = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
ref_mask = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
reference_skull = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'
fnirt_config = '/usr/share/fsl/5.0/etc/flirtsch/T1_2_MNI152_2mm.cnf'

owd = os.getcwd() # original working directory, added by Riikka Ruuth

def prepro_anat(k):
    try:

	subj = k
	

        for s in (['session2']):

		if (not os.path.isdir(data_path +subj+'/'+s)):
			continue
		    	
		if (os.path.isfile(results_path +subj +'/'+s+'/anat/nonlinear_reg/anat_HR_reoriented_warped.nii.gz')):
			print "Skipping "+ subj +'/' + s 
			continue
		'''
		if (os.path.isfile(pilot_path +subj +'/anat/nonlinear_reg/anat_reoriented_skullstrip_warped.nii.gz')):
			print "Skipping "+ subj +'/' + s 
			continue
		'''
		try:
			os.stat(results_path)
		except:
			os.mkdir(results_path)
		try:
			os.stat(results_path+subj)
		except:
			os.mkdir(results_path+subj)

		try:
			os.stat(results_path+subj+'/'+s)
		except:
			os.mkdir(results_path+subj+'/'+s)
		os.chdir(results_path+subj+'/'+s)
		print "Currently processing subject: ", subj+'/'+s

		anat = data_path + subj +'/'+ s + '/anat_HR.nii.gz'

		# Initialize workflow
		workflow = pe.Workflow(name='anat')
		workflow.base_dir = '.'

		# Reorient to FSL standard orientation
		deoblique = pe.Node(interface=afni.Warp(in_file=anat, deoblique=True, outputtype='NIFTI_GZ'), name='deoblique') #leave out if you don't need this
		reorient = pe.Node(interface=fsl.Reorient2Std(output_type='NIFTI_GZ'), name='reorient')
		workflow.connect(deoblique, 'out_file', reorient, 'in_file')

		# AFNI skullstrip
		skullstrip = pe.Node(interface=afni.SkullStrip(outputtype='NIFTI_GZ'), name='skullstrip')
		workflow.connect(reorient, 'out_file', skullstrip, 'in_file')

		# Segment with FSL FAST
		segmentation = pe.Node(interface=fsl.FAST(number_classes=3, use_priors=True, img_type=1), name='segmentation')
		segmentation.segments = True
		segmentation.probability_maps = True		
		workflow.connect(skullstrip, 'out_file', segmentation, 'in_files')


		# Register to HR anatomical
		hranat = results_path + subj+'/session1/anat/reorient/anat_HR_brain_reoriented.nii.gz'
		#anat2hr = pe.Node(interface=fsl.FLIRT(no_search=True, reference=hranat), name='anat2hr')
		anat2hr = pe.Node(interface=fsl.FLIRT(dof=6, reference=hranat), name='anat2hr')
		workflow.connect(reorient, 'out_file', anat2hr, 'in_file')


		# Register to standard MNI template
		#1. linear
		linear_reg = pe.Node(interface=fsl.FLIRT(cost='corratio', reference=ref_brain), name='linear_reg')

		#2.nonlinear
		nonlinear_reg = pe.Node(interface=fsl.FNIRT(fieldcoeff_file=True, jacobian_file=True, ref_file=ref_brain, refmask_file=ref_mask), name='nonlinear_reg')

		inv_flirt_xfm = pe.Node(interface=fsl.utils.ConvertXFM(invert_xfm=True), name='inv_linear_xfm')

		workflow.connect(skullstrip, 'out_file', linear_reg, 'in_file')
		#workflow.connect(anat2hr, 'out_matrix_file', linear_reg, 'in_matrix_file')
		workflow.connect(linear_reg, 'out_matrix_file', nonlinear_reg, 'affine_file')
		workflow.connect(skullstrip, 'out_file', nonlinear_reg, 'in_file')
		workflow.connect(linear_reg, 'out_matrix_file', inv_flirt_xfm, 'in_file')

		# Run workflow
		workflow.write_graph()
		workflow.run()

		print "ANATOMICAL PREPROCESSING DONE! Results in ", results_path+subj+'/'+s

    except:
	print "Error with patient: ", subj
	traceback.print_exc()

os.chdir(owd) # back to owd, added by Riikka Ruuth

if __name__ == '__main__':
	pool_size=10
	pool = multiprocessing.Pool(processes=pool_size)
	try:
		pool.map(prepro_anat, subj_dirs)
	finally:
		pool.close()
		pool.join()




