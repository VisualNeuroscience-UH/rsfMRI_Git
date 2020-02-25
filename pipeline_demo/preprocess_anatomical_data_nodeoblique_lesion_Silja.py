""" 
Nipype-pipeline: anatomical preprocessing

Anatomical data preprocessing steps:

1. Transform slices (from oblique to axial orientation and) to FSL std orientation
(2. Skull strip, done with BET pefore pipeline)
(3. Tissue segmentation, comment the lines out if you wish to do this)
4. Register to MNI152 standard template with linear and nonlinea registration

Running in terminal:
ipython
(in python terminal):
%run preprocess_anatomical_data

Hanna Halme // 2015
Questions and comments: hanna.halme@hus.fi

"""
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.fsl as fsl	     # fsl
import nipype.interfaces.afni as afni	     # afni
import copy
from nipype.interfaces.base import File, traits
import nipype.pipeline.engine as pe          # pypeline engine
import os
import numpy as np

data_path = '/opt1/MR_data/Silja_Raty/New_rs/image_data/'
results_path = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data_lesion/'


subj_dirs = os.listdir(data_path)
subj_dirs = ['Revis_0009_rs']


for k in range(0,len(subj_dirs)):
	subj = subj_dirs[k]
	try:
	    os.stat(results_path+subj)
	except:
	    os.mkdir(results_path+subj)
	os.chdir(results_path+subj)
	print "Currently processing subject: ", subj

	if os.path.isfile(data_path + subj_dirs[k] + '/session1/lesion_seg.nii.gz'):
	    anat = data_path + subj_dirs[k] + '/session1/lesion_seg.nii.gz'
	else:
	    print "No anatomical images found!"
	

	# Initialize workflow
	workflow = pe.Workflow(name='anat')
	workflow.base_dir = '.'

	# Reference brain images
	ref_brain = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
	ref_mask = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
	reference_skull = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'
	fnirt_config = '/usr/share/fsl/5.0/etc/flirtsch/T1_2_MNI152_2mm.cnf'
	anat = data_path + subj_dirs[k] + '/session1/lesion_seg.nii.gz'

	# Reorient to FSL standard orientation
	#deoblique = pe.Node(interface=afni.Warp(in_file=anat, deoblique=True, outputtype='NIFTI_GZ'), name='deoblique')
	reorient = pe.Node(interface=fsl.Reorient2Std(in_file=anat, output_type='NIFTI_GZ'), name='reorient')
	#workflow.connect(deoblique, 'out_file', reorient, 'in_file')

	# AFNI skullstrip
	#skullstrip = pe.Node(interface=afni.SkullStrip(args='-no_use_edge -ld 20', outputtype='NIFTI_GZ'), name='skullstrip')
	#workflow.connect(reorient, 'out_file', skullstrip, 'in_file')

	# Segment with FSL FAST
	#tissue priors
	#tissue_path = '/usr/share/fsl/5.0/data/standard/tissuepriors/2mm/'
	#csf_prior = tissue_path + 'avg152T1_csf_bin.nii.gz'
	#white_prior = tissue_path + 'avg152T1_white_bin.nii.gz'
	#gray_prior = tissue_path + 'avg152T1_gray_bin.nii.gz'

	#segmentation = pe.Node(interface=fsl.FAST(number_classes=3, use_priors=True, img_type=1), name='segmentation')
	#workflow.connect(skullstrip, 'out_file', segmentation, 'in_files')

	# Register to standard MNI template
	#1. linear
	linear_reg = pe.Node(interface=fsl.FLIRT(cost='corratio', reference=ref_brain), name='linear_reg')

	#2.nonlinear
	nonlinear_reg = pe.Node(interface=fsl.FNIRT(fieldcoeff_file=True, jacobian_file=True, ref_file=ref_brain, refmask_file=ref_mask), name='nonlinear_reg')

	inv_flirt_xfm = pe.Node(interface=fsl.utils.ConvertXFM(invert_xfm=True), name='inv_linear_xfm')

	workflow.connect(reorient, 'out_file', linear_reg, 'in_file')
	workflow.connect(linear_reg, 'out_matrix_file', nonlinear_reg, 'affine_file')
	workflow.connect(reorient, 'out_file', nonlinear_reg, 'in_file')
	workflow.connect(linear_reg, 'out_matrix_file', inv_flirt_xfm, 'in_file')

	# Run workflow
	workflow.write_graph()
	workflow.run()

	print "ANATOMICAL PREPROCESSING DONE! Results in ", results_path+subj
