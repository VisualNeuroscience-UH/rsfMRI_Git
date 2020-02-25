""" 
Skullstrip with default parameters 
Tissue segmentation of anatomical data
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

results_path = '/opt1/MR_data/Silja_Raty/New_rs/results/Patient_data/'

reskullstrip = False

#subj_dirs = ['Revis_0021_rs']
subj_dirs = os.listdir(results_path)

owd = os.getcwd() # original working directory

def prepro_anat(k):
    try:

	subj = k

        for s in (['session2', 'session3']):

		if (not os.path.isdir(results_path +subj+'/'+s)):
			continue
		
		os.chdir(results_path+subj+'/'+s) # go to patient directory
		print "Currently processing subject: ", subj+'/'+s

		if reskullstrip:
		    anat = results_path + subj +'/'+ s + '/anat/reorient/anat_reoriented.nii.gz'

		    # Initialize workflow
		    workflow = pe.Workflow(name='seg_anat')
		    workflow.base_dir = '.'

		    # AFNI skullstrip
		    skullstrip = pe.Node(interface=afni.SkullStrip(in_file = anat, outputtype='NIFTI_GZ'), name='skullstrip_default')

		    # Segment with FSL FAST
		    segmentation = pe.Node(interface=fsl.FAST(number_classes=3, use_priors=True, img_type=1), name='segmentation')
		    segmentation.segments = True
		    segmentation.probability_maps = True
		    workflow.connect(reorient, 'out_file', segmentation, 'in_files')

		    # Run workflow
		    workflow.write_graph()
		    workflow.run()

		else:
		    if not os.path.isdir('seg_anat'):
		    	os.mkdir('seg_anat')
		    if not os.path.isdir('seg_anat/segmentation'):
		    	os.mkdir('seg_anat/segmentation')
		    os.chdir('seg_anat/segmentation')
		    # Segment with FSL FAST
		    segmentation = fsl.FAST()
		    segmentation.inputs.number_classes=3
		    segmentation.inputs.use_priors=True
		    segmentation.inputs.img_type=1
		    segmentation.inputs.segments = True
		    segmentation.inputs.probability_maps = True
		    segmentation.inputs.in_files = results_path + subj +'/'+ s + '/anat/reorient/anat_LR_brain_reoriented.nii.gz'
		    segmentation.inputs.out_basename = 'anat_LR_brain_reoriented'
		    segmentation.run()

		print "ANATOMICAL SEGMENTATION DONE! Results in ", results_path+subj+'/'+s

    except:
	print "Error with patient: ", subj
	traceback.print_exc()

os.chdir(owd) # back to original working directory

if __name__ == '__main__':
	pool_size=10
	pool = multiprocessing.Pool(processes=pool_size)
	try:
		pool.map(prepro_anat, subj_dirs)
	finally:
		pool.close()
		pool.join()




