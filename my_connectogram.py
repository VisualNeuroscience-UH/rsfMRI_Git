import sys; sys.dont_write_bytecode = True # Prevent pyc file creation during import. These bimary files stay in memory in ipython etc
import make_connectogram 
import os

import pandas as pd
import pdb

# Cumbersome reloading to avoid ipython memory after Make_Connectogram changes
reload(make_connectogram)
Make_Connectogram = make_connectogram.Make_Connectogram

'''
This module creates Make_Connectogram instance which read excel workbook containing SVR model connection weights. These weights
are created with Craddock (Neuroimage 2013) analysis of functional MRI resting state measurement. The MakeConnectogram instance 
builds link file for connectogram and runs circos, a perl program which builds the graphical circular representation.

You can call the instance with do_histogram() method to just view the histogram.

Excel file names should start with GrpNum_sessionN_ , eg "sub011_session2_"

You need to define
work_path:  the path to your connectogram root, including folders "data" and "etc" and actual data folders 
threshold = (min,max) eg (-0.00002,0.00002): values between min and max will be set to 0
histogram = 0 or 1 # If 1, show histogram and stop program execution. 
distances = 0 or 1 # If 1, will scale the data to connection weight scale for easier comparison with same colors etc. 
heatmaps = 0 or 1 # if 1, do heatmaps for circos. Note, if N heatmaps is not 5, the circos_heatmap.conf needs manual tuning
	Heatmaps contain:	1) Voxel counts in each ROI
						2) Average shortest path
						3) Centrality degree
						4) Centrality eigenvector
						5) Clustering

Necessary files, position starting from connectogram root
'data\structure.label.txt'
'data\segments.txt'
'map.roi.names.xlsx'
'etc\circos.conf'
'etc\circos_heatmap.conf'
'etc\bands.conf'
'etc\color.brain.conf'
'etc\heatmap.conf'
'etc\ideogram.conf'
'etc\ideogram.label.conf'
'etc\ideogram.position.conf'
'etc\segment.order.conf'
'etc\ticks.conf'
'etc\ticks_heatmap.conf'

The program need to be in the path or you need to run it from the folder containting the "my_connectogram.py" and "make_connectogram.py". 
However, if it stops with exception, and the program files are not in your path, you may need to cd back to the folder containing 
the aforementioned files. 

usage from command interpreter eg:
cmd> ipython
[1] cd C:\Users\vanni\Laskenta\Control_Room
[2] run my_connectogram

Simo Vanni 2018-2019
'''
# Note, you have to run this where the my_connectogram.py is or put their position to your path

cwd = os.getcwd()	
work_path = u'/opt/Laskenta/Models/rsfMRI_Git/MTBI_Connectogram' # The perl probably has problems with spaces in the path.

# Leave this '' if you want all xls files from the whole folder
single_file_name = 'TstSub_sessionX_other_data.xls' #'ver001_session2_median_prediction_model_weights.xls' #'HO_ROI_atlas_ROI_distances.xls' # Put here your file name in case you try only one file. This must be empty in case you want to run folder
# single_file_name = 'mean_voxelmedian_prediction_model_weights_session2.xls' #'ver001_session2_median_prediction_model_weights.xls' #'HO_ROI_atlas_ROI_distances.xls' # Put here your file name in case you try only one file. This must be empty in case you want to run folder

data_folder = 'example_project' # eg 'mTBI_weights/controls_median', starting from your work_path. In case eg a single file in work_path use ''

# Set these if you are making connectograms with voxelcount heatmaps. Otherwise they are ignored. Not tested without voxelcount.
voxel_heatmap_data_folder = 'example_project'
voxel_heatmap_in_file_end = '_voxel_count.xls' # The excel file name eg 'HO_ROI_atlas_voxelcount.xls'


threshold = () # () # tuple values between min and max will be set to 0. Leave () for no threshold.
histogram = 1 # Flag to 1 to show histogram and stop program execution. Shows thresholded values in case the tuple is not empty
distances = 0 # Flag to 1 if you have distance or other data instead of weights. Assuming sheet name "Distances" for 1 and "Weights" for 0.
heatmap = 1 # If the N heatmaps is not 5, the circos_heatmap.conf needs manual tuning
network_analysis = 1 # after thresholding. Creates heatmap excel file to "network" subfolder with _nw.xls suffix
n_communities_in_network = 6 # The current community algo might not function very well in our case. Maybe thresholding helps
group_analysis = 0 # Do only group analysis. Requires the individual network analysis, ie in a separate run. 

# pseudo_seed_ROI = 'Right Cingulate Gyrus; posterior division' # If not empty string, shows connections only for this ROI.
pseudo_seed_ROI = '' 

#################################################
#### No need to change code below this level ####
#################################################


# Preparation for running the software and running it

# Prepare a list of all filenames
if single_file_name:
	all_filenames = [single_file_name]
else:
	os.chdir(work_path)
	all_filenames=[each for each in os.listdir(data_folder) if each.endswith('.xls')] # Read all filenames from folder
	os.chdir(cwd)

all_fullpath_names= [os.path.join(data_folder, each) for each in all_filenames]

# Make connectogram object instance
connectogram_object = Make_Connectogram(work_path, all_fullpath_names, threshold=threshold, histogram=histogram, \
										distances=distances, heatmap=heatmap, network_analysis=network_analysis, \
										n_communities_in_network=n_communities_in_network, group_analysis=group_analysis, \
										pseudo_seed_ROI=pseudo_seed_ROI)										

# Put data to object. 										
if heatmap:

	connectogram_object.voxel_heatmap_in_file_end=voxel_heatmap_in_file_end
	connectogram_object.voxel_heatmap_data_name = 'Voxels'
	connectogram_object.voxel_heatmap_data_folder = voxel_heatmap_data_folder
	connectogram_object.heatmap_color_scales_and_names={}
	connectogram_object.voxel_heatmap_filename_out = os.path.join('data','measure.0.txt') # The csv file for circos. The '0' will be running if network analysis

connectogram_object.scales_for_distances = (-2.51e-4,3.47e-4) # for distances to be scaled to weight data. Fixed min, max weights.

# Do the work
connectogram_object.run()


