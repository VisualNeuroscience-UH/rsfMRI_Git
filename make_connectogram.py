
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numbers
import math
import os
import subprocess
import sys
import networkx as nx
import pickle
import string
import scipy.stats as stats


import pdb


# avaa kansio C:\Users\vanni\OneDrive - University of Helsinki\Tyo\REVIS\Connectogram 
# circos esim http://circos.ca/documentation/tutorials/recipes/cortical_maps/lesson
# avaa ohjelma Anaconda 
# cd C:\Users\vanni\Laskenta\Control_Room 
# ipython 
# run revis_links_and_images

'''
This module reads excel workbook containing prediction accuracy values from functional MRI resting state measurement, 
and some other supplementary data. Then it builds link file for connectogram. Then it runs circos, 
a perl program which builds the graphical circular representation.

Simo Vanni 2018
'''

cwd = os.getcwd()

class Make_Connectogram:

	# We use pandas.IndexSlice to facilitate a more natural syntax. It may give performance warning, which can be ignored.
	idx = pd.IndexSlice

	def __init__(self, work_path, data_folder, all_names, threshold=(), histogram=0, distances=0, heatmap=0, network_analysis=0, \
				 n_communities_in_network=0, group_analysis=0, pseudo_seed_ROI=''):


		self.work_path = work_path
		self.data_folder = data_folder
		self.all_names = all_names
		self.threshold = threshold
		self.histogram = histogram
		self.distances = distances
		self.heatmap = heatmap
		self.network_analysis = network_analysis
		self.n_communities_in_network = n_communities_in_network
		self.group_analysis=group_analysis
		self.pseudo_seed_ROI = pseudo_seed_ROI

				
		os.chdir(self.work_path)

		# Read structure.label.txt to dataframe
		# Build dictionary of lobe and roi labels with their start and end positions

		label_file_name = os.path.join('data','structure.label.txt')
		
		lobe_roi_position_df=pd.read_csv(label_file_name,sep='\s+',names = ["Lobe", "Start", "End", "ROI"])
		lobe_roi_position_df=lobe_roi_position_df[["Lobe", "ROI", "Start", "End"]]

		# Pick names and turn to capital letters
		lobe_names=lobe_roi_position_df.Lobe.unique()
		lobe_names.tolist()
		
		self.lobe_names = list(map(str.upper,lobe_names))
		self.lobe_roi_position_df = lobe_roi_position_df
		
		# Read lobe and roi name mapping to df
		filename_lobe_roi = 'map.roi.names.xlsx'
		filename_lobe_roi_sheet_name='Sheet1'

		lobe_roi_map_df = pd.read_excel(filename_lobe_roi, sheet_name=filename_lobe_roi_sheet_name, index_col=0) # p3 mod, probably due to the xlrd excel reading package which was necessary for p3
		lobe_roi_map_df.index = lobe_roi_map_df.index.str.strip() # Strip whitespaces
		
		self.lobe_roi_map_df = lobe_roi_map_df
		
		# Init heatmap counter, heatmap filename. Needs a counter because excel files a re coming from more than one directions
		self.heatmap_counter = 0
		self.measure_filename=[]

		os.chdir(cwd)
		
	def save_obj(self, obj, filename):
		with open(filename + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_obj(self, filename):
		with open(filename + '.pkl', 'rb') as f:
			return pickle.load(f)
		
	def WriteConnectionList(self, connection_dataframe,  working_on_connections):
		# Here we define a function WriteConnectionList. Input arguments are [connection_dataframe,  working_on_connections, is_local].
		# We flag the local connection calls for clarity, rather than try to infer them from the file or variable names.
		# Interareal connections have separate references for origin and termination, and two connection types, feedforward and feedback. 
		# Local connections have only one reference list and one connection type.
		
		links_file_name = os.path.join('data','links_{0}.txt'.format(working_on_connections))

		# # Pick connection values
		# # We are indexing columns, thus the axis=1
		# connection_values_df = connection_dataframe.loc(axis=1)[idx[:,:,'Exists']]
		connection_values_df = connection_dataframe
		connection_values_df.index = connection_values_df.index.str.strip() # Strip whitespaces
		connection_values_df.columns = connection_values_df.columns.str.strip() # Strip whitespaces
		
		# Create mask for existing connections
		# mask = connection_values_df > 0
		mask = connection_values_df != 0
		# presynaptic_list = mask.index.get_values().tolist()
		presynaptic_list = mask.index.values.tolist()

		connection_list = []

		for presynaptic_ROI in presynaptic_list:
			# pick series object of column names for existing connections in this row, df.loc[row_indexer,column_indexer]
			postsynaptic_s = mask.loc[presynaptic_ROI,mask.loc[presynaptic_ROI,:]]
			postsynaptic_list = postsynaptic_s.index.values.tolist()

			for postsynaptic_lobe_and_ROI in postsynaptic_list:
				connection =[] # init temporary connection list

				# get presynaptic area and layer
				connection.append(presynaptic_ROI)
				
				# get postsynaptic area and layer
				connection.append(postsynaptic_lobe_and_ROI)

				# get strength
				strengh = connection_values_df.loc[presynaptic_ROI,postsynaptic_lobe_and_ROI]
				# connection.append("{:.1f}".format(strengh)) # turn to string
				connection.append("{:.2E}".format(strengh)) # turn to string
				
				connection_list.append(connection)

			
		# Write final connection list to links.txt file
		
		lobe_roi_map_df = self.lobe_roi_map_df
		lobe_roi_position_df = self.lobe_roi_position_df
		
		with open(links_file_name,'w') as f:
			for connection in connection_list:
				connection_to_file = []
				# For each connection, insert pre-areaname, turn pre-layer to start and stop, insert post-areaname, turn post layer to start and stop
				# add type, score. Finally write to file and go for next iteration.
				
				# Get pre-areaname
				lobe_name = lobe_roi_map_df.loc[connection[0],'LobeChr']
				connection_to_file.append(lobe_name)
				
				# Get pre-layer start and stop, and shift 25% towards band beginning to separate from post connections
				ROI_name = lobe_roi_map_df.loc[connection[0],'ROIName']

				lobe_roi_position = lobe_roi_position_df[(lobe_roi_position_df['Lobe'] == lobe_name.lower()) & (lobe_roi_position_df['ROI'] == ROI_name)]
				position_start = lobe_roi_position['Start'].values # numpy.ndarray	
				connection_to_file.append(position_start.astype(str)[0])
				position_end = lobe_roi_position['End'].values # numpy.ndarray
				position_end_shifted = int(np.mean([position_start,position_end])) # shift 25% left by setting end to mean of start and end
				connection_to_file.append(str(position_end_shifted))

				# Get post-areaname
				lobe_name = lobe_roi_map_df.loc[connection[1],'LobeChr']
				connection_to_file.append(lobe_name)
				
				# Get post-layer start and stop, and shift 25% towards band end to separate from pre connections
				ROI_name = lobe_roi_map_df.loc[connection[1],'ROIName']
				# Warning, the lobe_roi_position_df is coming from outer scope. I am sorry for such sloppy programming
				lobe_roi_position = lobe_roi_position_df[(lobe_roi_position_df['Lobe'] == lobe_name.lower()) & (lobe_roi_position_df['ROI'] == ROI_name)]
				position_start = lobe_roi_position['Start'].values # numpy.ndarray
				position_end = lobe_roi_position['End'].values # numpy.ndarray, note that you have to read this now for the following shift
				position_start_shifted = int(np.mean([position_start,position_end])) # shift 25% left by setting end to mean of start and end
				connection_to_file.append(str(position_start_shifted))
				connection_to_file.append(position_end.astype(str)[0])

				if float(connection[2]) > 0:
					connection_type = '1'
					# print "connection: {0} ; score {1}; type: {2}".format(connection_to_file,connection[2],connection_type)

				elif float(connection[2]) < 0:
					connection_type = '2'
				elif np.isnan(float(connection[2])):
					connection_type = '1'
					connection[2] = 0 # Set NaN to zero
				else:
					raise ValueError('Connection strength is not a non-zero number, aborting...')
					
				connection_score = connection[2]
				# type_and_score = 'type={0},score={1},radius1=0.95r,radius2=1.0r'.format(connection_type, connection_score)
				# type_and_score = 'type={0},score={1},radius1=0.95r'.format(connection_type, connection_score)
				# type_and_score = 'type={0},score={1},radius1=0.93r,radius2=0.97r'.format(connection_type, connection_score)
				type_and_score = 'type={0},score={1}'.format(connection_type, connection_score)
				connection_to_file.append(type_and_score)
				
				# if is_local:
					# references_to_file = '#ref {0}'.format(connection[3])
				# else:
					# # Get references, separately for Origin and Termination when interareal connection
					# references_origin = connection[3]
					# references_termination = connection[4]
					# references_to_file = '#ref Origin: {0}; Termination: {1}'.format(references_origin,references_termination)

				# connection_to_file.append(references_to_file)
				
				f.write( '\t'.join( connection_to_file ) )
				f.write('\n')
			f.close()
			
		print("Finished creating connectionlist and writing the links file for {0}.".format(working_on_connections))

	def do_actual_work(self, connection_dataframe,  working_on_connections='interareal', circos_out_path=''):

		self.WriteConnectionList(connection_dataframe, working_on_connections)

		# Run Circos
		print("Starting Circos for {0}. Wait some 10 seconds".format(working_on_connections))

		# conf_path = 'etc\circos_{0}.conf'.format(working_on_connections)
		if self.heatmap: # This just bypasses network heatmaps if heatmap flag is off
			conf_path = os.path.join('etc','circos_heatmap.conf') 
		else: 
			conf_path = os.path.join('etc','circos.conf') 
		
		circos_out_file = 'circos_{0}'.format(working_on_connections)
		links_file_name = os.path.join('data','links_{0}.txt'.format(working_on_connections))
		other_params = '' # other_params = '-param image/radius=15p' # Change parameters on command line
		FNULL = open(os.devnull, 'w')
		## subprocess.call(['powershell',"circos -conf {0} -outputfile {1} {2}".format(conf_path,circos_out_file, other_params)], stdout=FNULL, shell=True)

		# Learned to override parameters, getting link filename from commandline
		if sys.platform[:3] == 'lin':
			if circos_out_path: # circos does not tolerate empty -outputdir
				subprocess.call(["circos -conf {0} -outputdir {1} -outputfile {2} -param links/link/file={3} {4}".format(
									conf_path, circos_out_path, circos_out_file, links_file_name, other_params)], stdout=FNULL, stderr=subprocess.STDOUT, shell=True)
			else:
				subprocess.call(["circos -conf {0} -outputfile {2} -param links/link/file={3} {4}".format(
								conf_path, circos_out_path, circos_out_file, links_file_name, other_params)], stdout=FNULL, stderr=subprocess.STDOUT, shell=True)
		elif sys.platform[:3] == 'win':
			shell_name = 'powershell'
			if circos_out_path: # circos does not tolerate empty -outputdir
				subprocess.call(['{0}'.format(shell_name),"circos -conf {0} -outputdir {1} -outputfile {2} -param links/link/file={3} {4}".format(
									conf_path, circos_out_path, circos_out_file, links_file_name, other_params, shell_name)], stdout=FNULL, stderr=subprocess.STDOUT, shell=True)
			else:
				subprocess.call(['{0}'.format(shell_name),"circos -conf {0} -outputfile {2} -param links/link/file={3} {4}".format(
									conf_path, circos_out_path, circos_out_file, links_file_name, other_params, shell_name)], stdout=FNULL, stderr=subprocess.STDOUT, shell=True)
		
		
		print("Out from Circos for {0}. Connectogram {1} finished".format(working_on_connections, circos_out_file))
	
	def do_histogram(self, connection_dataframe, working_on_connections, outputpath, bins = 10):
		# vectorize dataframe matrix and call histogram visualization
		
		values = connection_dataframe.values # Turn to numpy matrix
		values_flat = values.flatten()
		values_flat_series = pd.Series(data=values_flat)
		stats = values_flat_series.describe().to_string()
		f = plt.figure()
		ax = f.add_subplot(111)

		plt.hist(values_flat, bins=bins)  # arguments are passed to np.histogram
		# plt.text(0.5, 0.5, 'matplotlib', transform=ax.transAxes)
		plt.axvline(values_flat.mean(), color='b', linestyle=':', linewidth=1)
		plt.axvline(0, color='k', linestyle='-', linewidth=1)
		count_th = len(values_flat_series.to_numpy().nonzero()[0])
		count_all = len(values_flat_series.to_numpy())
		
		stats += f'\n\ncount_th\t{count_th}'.expandtabs()
		stats += f'\nnon_zero%\t{count_th*100/count_all:.2f}'.expandtabs()
		if self.threshold:
			threshold = self.threshold[-1] * 1e6 # assumes numerical variable of len 2
			stats += f'\nthreshold\t{threshold:.2f}'.expandtabs()
		else:
			stats += f'\nthreshold\tNone'.expandtabs()
		
		plt.text(0.75, 0.70,stats,
		 horizontalalignment='center',
		 verticalalignment='center',
		 transform = ax.transAxes,fontsize=10)
		# plt.text(0.75, 0.35,threshold_text,
		 # horizontalalignment='center',
		 # verticalalignment='center',
		 # transform = ax.transAxes,fontsize=10)
		plt.title("{0} with {1} bins".format(working_on_connections, str(bins)))
		print(stats)
		#plt.show(block=False)
		# pdb.set_trace()
		# plt.show()
		outputfile = os.path.join(outputpath,f'histogram_{working_on_connections}.eps')
		plt.savefig(outputfile)
		outputfile = os.path.join(outputpath,f'histogram_{working_on_connections}.png')
		plt.savefig(outputfile)

		# stop execution
		# os.chdir(cwd)
		# sys.exit()
		
	def do_threshold(self, connection_dataframe, threshold=(0,0)):
		# set values within threshold limits around zero to zero
		values = connection_dataframe.values 
		mask = (values > threshold[0]) & (values < threshold[1])
		connection_dataframe[mask] = 0
		return connection_dataframe
		
	def do_distances2weight_scale(self, connection_dataframe):
		values = connection_dataframe.values # Turn to numpy matrix
		values_flat = values.flatten()
		dftemp=connection_dataframe
		
		scales_for_distances = self.scales_for_distances
		
		# Find zero values
		mask = values == 0
		target = scales_for_distances[1]-scales_for_distances[0]
		orig = values_flat.max() - values_flat.min()
		scaling_factor = target/orig

		# Put to zero
		dftemp = dftemp - values_flat.min() # There are actually zero min values, but this makes it more explicit
		# Scale
		dftemp = dftemp * scaling_factor
		# Invert
		dftemp = dftemp * -1
		# Set min value
		values_inverted_scaled = dftemp.values
		values_inverted_scaled_flat = values_inverted_scaled.flatten()
		
		dftemp = dftemp - values_inverted_scaled_flat.min() + scales_for_distances[0]
		
		# Set zero values to min values not to highlight in the connectogram. For correlations, u need to get rid of these
		dftemp[mask]=scales_for_distances[0]
		
		connection_dataframe = dftemp
		return connection_dataframe
	
	def do_heatmap(self):
		# Make heatmap file from excel file. Note that the voxelcounts without cx mask are almost identical

		# Load excel file contining heatmap data. If no file, return. Eg missing voxelcount		
		try:
			excel_all_sheets = pd.ExcelFile(self.heatmap_temp_filename)
		except:
			return
		heatmap_data_df_all_values =pd.read_excel(excel_all_sheets,index_col = 0)
		datasets = heatmap_data_df_all_values.columns
		#N_datasets = len(datasets)
		for dataset in datasets:
			heatmap_data_ser = heatmap_data_df_all_values.loc[:,dataset]	

			# Scale data to 0-1
			values = heatmap_data_ser.values # Turn to numpy 
			
			# heatmap_data_ser.loc[:,dataset]=heatmap_data_ser.loc[:,dataset] / float(values.max()) # float to make sure its not p2 integer division
			if dataset == 'average_shortest_path' and np.any(np.isinf(values)): # Circos does not accept inf
				indices = np.where(np.isinf(values))
				values[np.where(np.isinf(values))]=0 # Turn inf to zero 
				values[np.where(values==0)]=values.max() # Turn zero to max remaining heatmap \n
				print(('''\nAverage_shortest_path inf values at \n{0}\n replaced with max heatmap value.
				\n'''.format(heatmap_data_ser.index[indices[0][0]])))
				
			heatmap_data_ser=heatmap_data_ser / float(values.max()) # float to make sure its not p2 integer division
				
			# Reorder columns to match heatmap column order
			heatmap_file_base_df = self.lobe_roi_position_df[["Lobe", "Start", "End", "ROI"]] 
			
			# For each area in heatmap data, find match in self.lobe_roi_map_df
			# Take match abbreviated ROI name and find match in heatmap base file
			# Replace name in "ROI" column of heatmap file base with correct voxel count
			for area_name in heatmap_data_ser.index:
				match_ROI_name, lobe_index = self.lobe_roi_map_df.loc[area_name.strip(),'ROIName'], self.lobe_roi_map_df.loc[area_name.strip(),'LobeChr']
				mask_correct_row = (heatmap_file_base_df.loc[:,'ROI'] == match_ROI_name) & (heatmap_file_base_df.loc[:,'Lobe'] == lobe_index)
				# heatmap_file_base_df.loc[mask_correct_row,'ROI'] = heatmap_data_ser.loc[area_name,dataset]
				heatmap_file_base_df.loc[mask_correct_row,'ROI'] = heatmap_data_ser.loc[area_name]
						
			# Write out heatmap txt file
			heatmap_filename_base, file_extension = os.path.splitext(self.heatmap_temp_filename)
			# voxel_heatmap_filename_out = 'data\measure_{0}.txt'.format(heatmap_filename_base)
			# replace 0 with correct index from counter
			voxel_heatmap_filename_out = self.voxel_heatmap_filename_out
			# voxel_heatmap_filename_out = string.replace(voxel_heatmap_filename_out, '0', str(self.heatmap_counter))
			voxel_heatmap_filename_out =  voxel_heatmap_filename_out.replace('0', str(self.heatmap_counter)) # string.replace() is deprecated on python 3
			heatmap_file_base_df.to_csv(path_or_buf=voxel_heatmap_filename_out, sep="\t", na_rep='0', header=False, index=False)
			
			#self.measure_filename.append(voxel_heatmap_filename_out)
			# set heatmap min max values for colorscale
			self.heatmap_color_scales_and_names['heatmap{0}'.format(self.heatmap_counter)] = [(values.min(),values.max()),'{0}'.format(dataset)]
			
			self.heatmap_counter += 1
				
	def do_network_analysis(self, connection_dataframe):
	
		#Test whether network analysis has already been done to same file
		# if os.path.isfile(self.network_out_filename + '.pkl'):
		if False: # Temporary override to do always new network analysis
			print(('Network analysis for {0} exists, skipping...'.format(self.network_out_filename)))
			return
		else:
			print(('New network analysis for {0}'.format(self.network_out_filename)))
			
		values = connection_dataframe.values # Turn to numpy matrix
		

		# Algorithms are not well characterized for negative values, so we have to ditch them by setting them to 0. 	
		# values = np.where(values>0,values,0)
		values = np.abs(values)
		
		# Algorithms do not tolerate our small values. We normalize them to max 1
		values_flat = values.flatten()
		# f = plt.figure()
		# ax = f.add_subplot(211)
		# ax.imshow(values, cmap="gray")
		# ax2 = f.add_subplot(212)
		# ax2.hist(values_flat,bins=20)
		# plt.show()

		values = values / values_flat.max() # normalize to max 1

		# Create directed graph with weights.
		dg=nx.DiGraph(incoming_graph_data=values) # Check that the directions are correct when importing to nx: pre rows, post columns

		# Centrality
		print('Calculating centrality...')
		# Weighted degree. In Zuo et al. Cerebral Cortex August 2012;22:1862-1875 this is degree centrality.
		degree = [d for n, d in dg.degree(weight='weight')]
		centrality_degree = degree
		
		# Eigenvector centrality
		centrality_eigenvector_dict = nx.eigenvector_centrality_numpy(dg, weight='weight')
		centrality_eigenvector = [centrality_eigenvector_dict[n] for n in centrality_eigenvector_dict]
		
		# # Global centrality
		# global_reaching_centrality = nx.algorithms.centrality.global_reaching_centrality(dg,weight='weight')
		
		# # Local centrality
		# local_reaching_centrality = [nx.algorithms.centrality.local_reaching_centrality(dg,c,weight='weight') for c in np.arange(len(dg.nodes))]
		
		# Clustering
		print('Calculating clustering...')
		# # Check clustering coefficient, local efficiency, transitivity (triangles?, correlated with clustering?), and global efficiency
		# Clustering coefficient, takes a min or so to calculate
		clustering_dict = nx.algorithms.cluster.clustering(dg,weight='weight')
		clustering = [clustering_dict[c] for c in clustering_dict]

		# Shortest path, a surrogate for global efficiency
		print('Calculating shortest path length...')
		# Weights are distances for floyd. We have to invert weights.
		values_inv = np.reciprocal(values)
		dg_inv=nx.DiGraph(incoming_graph_data=values_inv)
				
		# Floyds algorithm is appropriate for finding shortest paths in dense graphs or graphs with negative weights 
		# when Dijkstras algorithm fails. 
		# shortest_path_floyd = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(dg,weight='weight')
		shortest_path_floyd = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(dg_inv,weight='weight')
		shortest_path_floyd_inverse = np.reciprocal(shortest_path_floyd)
		shortest_path_floyd_inverse = shortest_path_floyd_inverse[np.isfinite(shortest_path_floyd_inverse)]
		average_efficiency = (1.0/(len(dg.nodes)*(len(dg.nodes)-1))) *  shortest_path_floyd_inverse.sum()
		average_shortest_path = shortest_path_floyd.mean(axis=1) # This is numpy matrix
		average_shortest_path = np.asarray(np.squeeze(average_shortest_path)) # This is array
		average_shortest_path = average_shortest_path[0]
		
		# Ideal efficiency equals 1 in our case, because we have normalized the weights to 1. 
		
		# # # This efficiency measure is from Burt, Ronald S. Structural Holes: The Social Structure of Competition. 
		# # # Cambridge: Harvard University Press, 1995. It takes some 15 min to calculate for weighted directed graph of 96
		# esize = nx.effective_size(dg,weight='weight')
		# # efficiency = {n: v / dg.degree(n) for n, v in esize.items()} # The problem here is the unweighted degree, which is constant in our case of full connectivity
		# efficiency_dict = {n: v / degree[n] for n, v in esize.items()} # Here we use the weighted degree as the denominator
		# eff = [efficiency_dict[e] for e in efficiency_dict]
		
		# # This algorithm does not allow directed graphs
		# print('Calculating community structure...')
		
		# g=dg.to_undirected()
		# k=self.n_communities_in_network
		# cliques = nx.find_cliques(g)
		# community = nx.algorithms.community.kclique.k_clique_communities(g, k, cliques=cliques)
		# community_list = [fset for fset in community]
		community_list = []
				
		# Build attributes to self
		network={'centrality_degree': centrality_degree, 'centrality_eigenvector':centrality_eigenvector, 'clustering':clustering,
			'shortest_path_floyd':shortest_path_floyd, 'average_shortest_path':average_shortest_path, 
			'average_efficiency':average_efficiency, 'community':community_list}
		self.network = network
		#
		# Save to pickle file 
		self.save_obj(network, self.network_out_filename)
		
		# Save potential heatmaps to excels
		network_df=pd.DataFrame({'average_shortest_path':average_shortest_path, 'centrality_degree':centrality_degree,
								 'centrality_eigenvector':centrality_eigenvector, 'clustering':clustering}) # p3 mod: If data is a dict, argument order is maintained for Python 3.6 and later.
		network_df.index=connection_dataframe.index
		
		network_df.to_excel(self.network_out_filename + '.xls',sheet_name='Network')		
					
	def do_group_analysis(self):
		# Network analysis summary and visualization over all subjects.
	
		#Pick correct path. 
		excel_file_name = self.all_names[0]
		path, filename_w_ext = os.path.split(excel_file_name)
		# network_path_out = os.path.join(path,'network_analysis')
		network_path_out = os.path.join(path,self.nw_analysis_folder_name)

		if not os.path.isdir(network_path_out):
			print("Network path not found. Run network analysis first. Put group_analysis = 0. Aborting...")
			return
			
		# Remove old group analysis. Otherwise it messes up the search for xls files below.
		excel_grp_files = [each for each in os.listdir(network_path_out) if 'GrpData' in each] 
		if excel_grp_files: # if not empty list
			for file_name in excel_grp_files:
				os.remove(os.path.join(network_path_out, file_name))
		
		excel_files = [each for each in os.listdir(network_path_out) if each.endswith('.xls')] # Read all filenames from folder
		
		# Collect all excels to one multi pd.df. Save to network_path_out
		
		# Init multi-index dataframe with zeros. Dims index(ROI,category) column(subject)
		length_of_name = 11 # This should cover the part of filename which provides individual id.
		excel_file_tmp = excel_files[0] # For init of matrices
		pd_tmp = pd.ExcelFile(os.path.join(network_path_out, excel_file_tmp))
		df_tmp = pd.read_excel(pd_tmp, index_col=0) # p3 mod
		index_ROIs = df_tmp.index.tolist()
		columns_categories = df_tmp.columns.tolist()
		columns_subjects = [name[:length_of_name] for name in excel_files]
		mi_df_index = index_ROIs		
		mi_df_columns = pd.MultiIndex.from_product([columns_subjects, columns_categories])
		
		df = pd.DataFrame(0, index=mi_df_index, columns=mi_df_columns)

		for excel_file in excel_files:
			pd_tmp = pd.ExcelFile(os.path.join(network_path_out, excel_file))
			individual_data_df = pd.read_excel(pd_tmp, index_col=0) # p3 mod
			subject = excel_file[:length_of_name]
			df[subject] = individual_data_df

		df_out_name = os.path.join(network_path_out, excel_file_tmp[:3] + '_' + excel_file_tmp[7:15] + "_GrpData" + '.pkl')	
		df.to_pickle(df_out_name)
		df_out_name_xls = os.path.join(network_path_out, excel_file_tmp[:3] + '_' + excel_file_tmp[7:15] + "_GrpData" + '.xls')	
		df.to_excel(df_out_name_xls)

		# Put each column to own excel file for easier processing later. Hierarchical indexing does not work for group ave data, thus the try-except
		try:
			# Init gamma fit params and figure for 4 network parameters
			n_params = len(columns_categories)
			shape = np.zeros([n_params]); loc = np.zeros([n_params]); scale = np.zeros([n_params])
			data_statistics=dict() # N non inf data may vary, thus we go for dict
			x=np.zeros([100,n_params])
			# g=np.zeros([100,n_params])
			distribution='gamma' # If you change this change the stats.gamma.XXXX also below
			model_values=np.zeros([100,n_params])
			ncols=2; nrows=2
			assert n_params == ncols*nrows, 'Check ncols for group analysis parameter histogram' # comment away for odd N histograms
			fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(13,8))
			ax = ax.flatten() # To enable 1-dim iterator below for 2-dim subplot
			
			filenameroot=excel_file_tmp[:3] + '_' + excel_file_tmp[7:15]
			
			for iterator, column in enumerate(columns_categories):
			
				# Prepare filename and write out each network analysis to separate excel files
				df_out_column_name_xls = os.path.join(network_path_out, filenameroot + \
								"_GrpData_" + column + '.xls')	
				# df_column = df.loc[slice(None), (slice(None), slice(column, column))]
				idx=pd.IndexSlice # py3 mod, mod after fliplr
				df_column = df.loc[:, idx[:,column]]# py3 mod, mod after fliplr
				df_column.to_excel(df_out_column_name_xls)

				# Do stats for network analysis parameter
				# Prepare data for fitting and plotting
				df_column_t = df_column.transpose(copy=True)
				data_array_flatten = df_column_t.values.flatten()
				data_array_clean = data_array_flatten[~np.isinf(data_array_flatten)]
				data_N_infs = np.where(np.isinf(data_array_flatten))[0].size
				data_N = len(data_array_clean)

				# Plot data histogram
				bin_values, foo, foo2 = ax[iterator].hist(data_array_clean, density=True, bins=20) # Drop NaN from histogram
				ax[iterator].set_title(column)
				
				# Gamma stats
				shape[iterator], loc[iterator], scale[iterator] = stats.gamma.fit(data_array_clean, loc=0)
				x[:,iterator] = np.linspace(stats.gamma.ppf(0.001, shape[iterator], loc=loc[iterator], scale=scale[iterator]), 
							stats.gamma.ppf(0.999,  shape[iterator], loc=loc[iterator], scale=scale[iterator]), 100)
				
				# Fit and plot model values
				model_values[:,iterator] = stats.gamma.pdf(x=x[:,iterator], a=shape[iterator], loc=loc[iterator], scale=scale[iterator])
				ax[iterator].plot(x[:,iterator], model_values[:,iterator], 'r-', linewidth=6, alpha=.6)
				ax[iterator].annotate(s='shape = {0:.2g}\nloc = {1:.2g}\nscale = {2:.2g}\nN connections = {3}\nN infs = {4}'.format(
										shape[iterator], loc[iterator], scale[iterator], data_N, data_N_infs), xy=(.6,.4),
										xycoords='axes fraction')
				
				# Rescale y axis if model fit goes high. Shows histogram better
				if model_values[:,iterator].max() > 1.5 * bin_values.max():
					ax[iterator].set_ylim([ax[iterator].get_ylim()[0],1.1 * bin_values.max()])
				
				# Set data to dict for saving
				data_statistics[column]={'data':data_array_clean, 'N_infs':data_N_infs, 'distribution':distribution, 'shape':shape[iterator], 
																		'loc':loc[iterator], 'scale':scale[iterator]} 

		except:
			print("Grouping of network analysis params to distinct xls files not done. Group analysis? Wrong file naming?")

		# Save stats
		data_statistics_file_out = os.path.join(network_path_out,filenameroot + "_GrpStats" + '.pkl')
		pickle.dump(data_statistics, open(data_statistics_file_out, "wb")) 

		# Save stats figures
		filenameout=os.path.join(network_path_out,filenameroot + "_GrpStats" + '.png')
		plt.savefig(filenameout,format='png')
		filenameout=os.path.join(network_path_out,filenameroot + "_GrpStats" + '.svg')
		plt.rcParams['svg.fonttype'] = 'none'
		plt.savefig(filenameout,format='svg')		

		# plt.show()
		
	def do_heatmap_colorscales(self, path_to_heatmap_colorscale):
		# Draw polar colorscales and save figure. One for each subject.
		# Create figure with as many subplots as there are heatmaps
		n_subplots = self.heatmap_counter # Python index from 0
		
		# Get colorscale from C:\Users\vanni\prog\circos-0.69-6\etc\colors.brewer.conf. 
		# If colorindex >5, colorindex =0
		hm_colors = {	'0': ['greys-4-seq', np.array([[220,220,220],[170,170,170],[120,120,120],[60,60,60]])],
						'1': ['reds-4-seq', np.array([[254,209,200],[244,154,135],[234,66,64],[203,24,29]])],
						'2': ['greens-4-seq', np.array([[207,248,203],[166,218,159],[106,192,108],[15,129,49]])],
						'3': ['blues-4-seq', np.array([[215,220,255],[169,205,221],[97,164,214],[33,113,181]])],
						'4': ['purples-4-seq', np.array([[232,220,237],[193,191,216],[148,144,190],[96,71,153]])],
						'5': ['oranges-4-seq', np.array([[255,237,180],[237,182,120],[218,127,60],[200,72,1]])]}
	

		fig, ax = plt.subplots(ncols=n_subplots,figsize=(10,6))
		
		for myplot in np.arange(n_subplots):
			shift_plot = 1.5 * np.mod(myplot+1,2) # Shift every second plot by 1
			heatmap_metadata = self.heatmap_color_scales_and_names['heatmap{0}'.format(str(myplot))]
			heatmap_min=heatmap_metadata[0][0]
			heatmap_max=heatmap_metadata[0][1]
			heatmap_name=heatmap_metadata[1]

			# Create colormap for this plot
			cmap_data=hm_colors[str(myplot)]
			cmap_name=cmap_data[0]
			cmap = mpl.colors.ListedColormap(cmap_data[1]/255., name=cmap_name, N=None)
			
			r_array = np.array([0,1])
			phi_array = np.linspace(0, 2 * np.pi, 5) # If u want more round plot, increase the n steps
			phi_array = phi_array + np.pi/2

			r_grid, phi_grid, = np.meshgrid(r_array, phi_array)

			z_grid = r_grid + phi_grid
			x_grid = r_grid * np.cos(phi_grid)
			y_grid = r_grid * np.sin(phi_grid) 

			y_grid = y_grid + shift_plot
			ax[myplot].set_ylim([-1.5,3])
	
			ax[myplot].pcolormesh(x_grid, y_grid, z_grid, cmap=cmap)
			# ax[myplot].axis('equal')
			ax[myplot].axis('off')

			x_anno_array = np.array([-.3, -1.15, 0, 1.15, .3])
			y_anno_array = np.array([.9, 0, -1.05, 0, .9])

			# Create and set subplot title
			underscore_indices = [pos for pos, char in enumerate(heatmap_name) if char == '_']
			title_breaks = [0] + underscore_indices + [len(heatmap_name)]; 
			title_anno=''

			if underscore_indices: # If not empty
				for iterator, wordlimit in enumerate(title_breaks[:-1]):
					if heatmap_name[wordlimit] == '_':
						wordlimit = wordlimit + 1
						title_anno += '\n'
					title_anno += heatmap_name[wordlimit:title_breaks[iterator+1]]
			else:
				title_anno = heatmap_name
	
			ax[myplot].text(0, 1.2 + shift_plot, title_anno.title(), fontsize=10, weight='bold', ha='center', va='center')
			
			# Create and set colorscale limit numbers
			text_anno = ["{:.2g}".format(value) for value in np.linspace(heatmap_min,heatmap_max,5).tolist()]

			ax[myplot].text(x_anno_array[0], y_anno_array[0] + shift_plot, text_anno[0], fontsize=10, ha='right', va='center')
			ax[myplot].text(x_anno_array[1], y_anno_array[1] + shift_plot, text_anno[1], fontsize=10, ha='right', va='center')
			ax[myplot].text(x_anno_array[2], y_anno_array[2] + shift_plot, text_anno[2], fontsize=10, ha='center', va='top')
			ax[myplot].text(x_anno_array[3], y_anno_array[3] + shift_plot, text_anno[3], fontsize=10, ha='left', va='center')
			ax[myplot].text(x_anno_array[4], y_anno_array[4] + shift_plot, text_anno[4], fontsize=10, ha='left', va='center')
						
		# Save heatmap color keys
		filenameout=path_to_heatmap_colorscale + '.png'
		plt.savefig(filenameout,format='png')
		filenameout=path_to_heatmap_colorscale + '.svg'
		plt.rcParams['svg.fonttype'] = 'none'
		plt.savefig(filenameout,format='svg')		
	
	def do_pseudo_seed(self, connection_dataframe, filename):
		
		# Pick pseudo seed ROI from self
		pseudo_seed_ROI = self.pseudo_seed_ROI

		# Leave all incoming and outgoing connections for that ROI, mark all others to zero
		connection_values_df = connection_dataframe
		connection_values_df.index = connection_values_df.index.str.strip() # Strip whitespaces
		connection_values_df.columns = connection_values_df.columns.str.strip() # Strip whitespaces

		assert pseudo_seed_ROI in connection_values_df.columns, 'Cannot locate the ROI you gave as pseudo seed. Are you sure the name is correct?'

		mask = connection_values_df.copy()
		mask[:] = 0
		mask.loc[pseudo_seed_ROI,:] = connection_values_df.loc[pseudo_seed_ROI,:]
		mask.loc[:,pseudo_seed_ROI] = connection_values_df.loc[:,pseudo_seed_ROI]
		connection_values_df=mask.copy()
		
		self.heatmap = False
		self.network_analysis = False

		# Change filename so that the fig does not overwrite the full connectogram
		underscore_indices = [pos for pos, char in enumerate(filename) if char == '_']
		working_on_file_name_begin = filename[0:underscore_indices[1]]
		pseudo_seed_ROI_evolve = pseudo_seed_ROI.replace(";", "_")
		pseudo_seed_ROI_evolve2 = pseudo_seed_ROI_evolve.replace("'", "_")
		pseudo_seed_ROI_evolve3 = pseudo_seed_ROI_evolve.replace(" ", "_")
		filename_new = os.path.join(working_on_file_name_begin + '_pseudoROI_' + pseudo_seed_ROI_evolve3)
		
		# Return DF, filename
		return connection_values_df, filename_new
	
	def do_flip(self, connection_dataframe, filename):
		'''
		Flip data between left and right hemispheres, both row-wise and columnwise
		'''
		df_temp1=connection_dataframe.copy()
		
		# Reverse Left-Right values for rows 
		df_temp1.iloc[0::2, :] = connection_dataframe.iloc[1::2, :].values
		df_temp1.iloc[1::2, :] = connection_dataframe.iloc[0::2, :].values
		
		# Reverse Left-Right values for columns
		df_temp2=df_temp1.copy()
		df_temp2.iloc[:, 0::2] = df_temp1.iloc[:, 1::2].values
		df_temp2.iloc[:, 1::2] = df_temp1.iloc[:, 0::2].values
		
		connection_dataframe = df_temp2.copy()
		
		return connection_dataframe
			
	def run(self):
	
		cwd = os.getcwd()
		os.chdir(self.work_path)
		
		# Do group analysis after first running individuals
		if self.group_analysis:
			self.do_group_analysis()
			os.chdir(cwd)
			return

		# Loop through different excel files ie subjects
		for excel_file_name in self.all_names:

			self.heatmap_counter = 0 # Init heatmap counter
			self.measure_filename=[] # Init heatmap filenames


			# Load excel book
			excel_all_sheets = pd.ExcelFile(excel_file_name)

			# # Read the interareal, intrinsic excitatory and intrinsic inhibitory connections to separate dataframes
			# # Columns are mapped to three multi-index levels by the header parameter
			# # Rows are mapped to two multi-index levels by the index_col parameter
			if self.distances:
				interareal_connections_dfmi = pd.read_excel(excel_all_sheets, 'Distances', index_col=0) # p3 mod
				# Turn to "weight scale"
				interareal_connections_dfmi = self.do_distances2weight_scale(interareal_connections_dfmi)
			else:
				interareal_connections_dfmi = pd.read_excel(excel_all_sheets, 'Weights', index_col=0) # p3 mod
			
			#get folder or relative path to xls
			path, filename_w_ext = os.path.split(excel_file_name)
			#get filenamebase and extension
			filename, file_extension = os.path.splitext(filename_w_ext)
			
			# In REVIS project we flip all occipital lesions to left side, if they are not already there.
			# Check for left-right flip. Call flip method when appropriate. Save with new name.
			if 'fliplr' in filename:
				flipped_folder = os.path.join(self.work_path, self.data_folder, 'flipped_excel_files')
				if not os.path.exists(flipped_folder):
					os.makedirs(flipped_folder)
				print("Flip flag found for {0}. Left and right will be flipped.".format(filename))
				# print("Before flip\n", interareal_connections_dfmi)
				interareal_connections_dfmi = self.do_flip(interareal_connections_dfmi, filename)
				flipped_filename = filename.replace('fliplr', 'FLIPPEDlr')
				df_out_name_xls = os.path.join(flipped_folder,  flipped_filename + '.xls')	
				interareal_connections_dfmi.to_excel(df_out_name_xls)
				# print("After flip\n", interareal_connections_dfmi)
			else:
				print("Flip flag NOT found for {0}.".format(filename))
			
			if self.threshold: # if the tuple is not empty
				interareal_connections_dfmi = self.do_threshold(interareal_connections_dfmi,self.threshold)
				
			if self.pseudo_seed_ROI:
				interareal_connections_dfmi, filename = self.do_pseudo_seed(interareal_connections_dfmi, filename)

			if self.histogram:
				self.do_histogram(interareal_connections_dfmi, filename, path, bins = 50) # activate for checking the histogram. Stops program execution eg for setting thresholds etc
				continue # Loop all histograms in current folder, skip following tasks
                
			if self.network_analysis:
				# network_path_out = os.path.join(path,'network_analysis')
				network_path_out = os.path.join(path,self.nw_analysis_folder_name)
				
				# Create subfolder 'network_analysis' if it does not exist
				if not os.path.exists(network_path_out):
					os.mkdir(network_path_out) 
					
				self.network_out_filename = os.path.join(network_path_out, filename + '_nw') # temp filename for current network filename
				self.do_network_analysis(interareal_connections_dfmi)

			if self.heatmap: # Prepare heatmap file from auxiliary excel file, such as voxels
				# Prepare voxel heatmap name with correct path
				underscore_indices = [pos for pos, char in enumerate(filename) if char == '_']
				working_on_file_name_begin = filename[0:underscore_indices[1]]
				self.working_on_file_name_begin = working_on_file_name_begin
				self.heatmap_temp_filename= os.path.join(self.voxel_heatmap_data_folder, self.working_on_file_name_begin + self.voxel_heatmap_in_file_end)
				self.do_heatmap()

			if self.network_analysis: # Second time here on purpose
				# Prepare network heatmap
				self.heatmap_temp_filename=self.network_out_filename + '.xls'
				self.do_heatmap()

			if self.heatmap: # Second time here on purpose
				# Print out colorscales
				# Create filename
				path_to_heatmap_colorscale = os.path.join(path,'circos_' + working_on_file_name_begin + '_colorscales')
				
				self.do_heatmap_colorscales(path_to_heatmap_colorscale)
				
			self.do_actual_work(interareal_connections_dfmi,  working_on_connections=filename, circos_out_path=path)
			
			
		os.chdir(cwd)

