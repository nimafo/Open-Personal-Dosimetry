#!/usr/bin/env python
# coding: utf-8


import os, sys
import getopt

## get input
(opts, args) = getopt.getopt(sys.argv[1:],"i:o:",[])
for (opt, arg) in opts:
	if opt == '-i': # system type
		input_file = arg
	if opt == '-o': # system type
		output_prefix = arg

try:
	input_file
except: 
	sys.exit("Input file missing. Specify with -i.")

try:
	output_prefix
except:
	sys.exit("Output prefix missing. Specify with -i.")

"""
Dictionary keys
======================
S-cone-opic: 				sc				S-cone normalized action
M-cone-opic: 				mc				M-cone normalized action 
L-cone-opic: 				lc				L-cone normalized action
Rhodopic: 					rh				Rhodopic normalized acton
Melanopic: 					mel				Melanopic normalized action
Photopic:					v				V-lambda 1924
Melanopic Lucas 2014: 		mel_lucas		M-lambda Lucas 2014
"""

calcs = ["sc", "mc", "lc", "rh", "mel"]

### constants
# note these are all for the CIE standard action spectra normalized to 1
dict_lumeff = { "sc": 0.8173 * 1000.0, # from CIE S 026 toolbox - never used...
	"mc": 1.4558 * 1000.0, # from CIE S 026 toolbox - never used...
	"lc": 1.6289 * 1000.0, # from CIE S 026 toolbox - never used...
	"rh": 1.4497 * 1000.0, # from CIE S 026 toolbox - never used...
	"mel": 1.3262 * 1000.0, # from CIE S 026 toolbox - never used...
	"v": 683.002, # V-lambda lumeff for illuminance calculation
	"mel_lucas": 729.8325} # C-lambda Lucas 2014 lumeff for melanopic illuminance calculation

d_wl = 5.0 # delta wavelength in input source from ALFA calc

### read action and d65 spectra
def read_action_spectra(path="action_spectra.csv"):
	"""Read in the action spectra and output a lookup dictionary."""
	dict_actionspectra = {}
	with open(path, 'r') as f:
		headers = f.readline().rstrip().split(',')
		for h in headers[1:]: # skip first wavelenth column
			dict_actionspectra[h] = {} # create dict entry for column header text strings
		
		for line in f:
			parts = line.rstrip().split(',')
			for i in range(7):
				if i == 0:
					wl = int(parts[i])
				else:
					dict_actionspectra[headers[i]][wl] = float(parts[i])
	
	return dict_actionspectra

def read_spectra_file(path="D65.csv"):
	"""Read a nm,irr/nm format spectral file"""
	spectrum = []
	with open(path, 'r') as f:
		for line in f:
			parts = line.rstrip().split(',')
			wl = int(parts[0])
			irr = float(parts[1])
			spectrum.append( (wl, irr) )
	
	return spectrum

dict_actionspectra = read_action_spectra() 

# Read D65 CIE standard illuminant spectrum for normalization per CIE S 026
spectra_d65 = read_spectra_file()

# Read Lucas 2014 spectra for old melanopic illuminance calculation
spectra_mel_lucas = read_spectra_file(path="c-lambda_lucas2014.csv")
dict_actionspectra["mel_lucas"] = {}
for m in spectra_mel_lucas:	
	dict_actionspectra["mel_lucas"][m[0]] = m[1]

def calc_s026(spectra):
	"""An algorithm to calculate the CIE S026 *-opic toolbox values."""
	
	global d_wl, dict_lumeff, spectra_d65, dict_actionspectra, calcs
	
	### calculate v-lambda illuminance and *-opic irradiances
	def opic_irradiance(sample, key, dict_actionspectra, d_wl):
		"""Calculate *-opic irradiances. sum(delta_wavelength * action_spectra * irrad[wl])
		Times 100 to match CIE units of uW/cm^2."""
		irr = 0
		for s in sample:
			wl = s[0]
			spec_irr = s[1]
			
			if wl >= 360:
				irr += spec_irr * d_wl * dict_actionspectra[key][wl]
		
		# irr = irr * 100.0 # X-opic irrad in uW/cm^2
		
		return irr

	# visible illuminance of sample
	ill = opic_irradiance(spectra, "v", dict_actionspectra, d_wl)
	ill = ill * dict_lumeff["v"]
	# print ("v_ill", ill)
	
	if ill < 0.01:
		empty = { "sc": 0.0,
			"mc": 0.0,
			"lc": 0.0,
			"rh": 0.0,
			"mel": 0.0 }
		return (0.0, 0.0, 0.0, empty, empty, empty, empty)
	else:
		mel_ill = opic_irradiance(spectra, "mel_lucas", dict_actionspectra, d_wl)
		mel_ill = mel_ill * dict_lumeff["mel_lucas"] 
		# print ("m_ill", mel_ill)

		m_over_p = mel_ill / ill
		# print ("m/p", m_over_p)

		# normalize D65 distiribution to match illuminance from sample
		ill_d65 = opic_irradiance(spectra_d65, "v", dict_actionspectra, d_wl)
		ill_d65 = ill_d65 * dict_lumeff["v"] 
		# print ("v_ill_d65 pre-adjustment", ill_d65)
		spectra_d65 = [(u[0],u[1] * ill/ill_d65) for u in spectra_d65] # adjust d65 to match measured ill and check this is correct!
		ill_d65 = opic_irradiance(spectra_d65, "v", dict_actionspectra, d_wl)
		ill_d65 = ill_d65 * dict_lumeff["v"] 
		# print ("v_ill_d65", ill_d65)

		# *-opic irradiance for sample and d65-normalization
		opic_irrads = {}
		for type in calcs:
			opic = opic_irradiance(spectra, type, dict_actionspectra, d_wl)
			opic_irrads[type] = opic
		opic_irrads_d65 = {}
		for type in calcs:
			opic = opic_irradiance(spectra_d65, type, dict_actionspectra, d_wl)
			opic_irrads_d65[type] = opic

		### calculate ELR's
		opic_elrs = {}
		opic_elrs_d65 = {}
		for type in calcs:
			opic_elrs[type] = 1000.0 * opic_irrads[type] / ill
			opic_elrs_d65[type] = 1000.0 * opic_irrads_d65[type] / ill

		### calculate DER for sample irradiance
		opic_ders = {}
		for type in calcs:
			opic_ders[type] = opic_elrs[type] / opic_elrs_d65[type]

		### calculate EDI in non-standard CIE units of lx
		opic_edis = {}
		for type in calcs:
			opic_edis[type] = 1000.0 * opic_irrads[type] / opic_elrs_d65[type]
			
		return (ill, mel_ill, m_over_p, opic_irrads, opic_elrs, opic_ders, opic_edis)

class spectral_data:
	def __init__(self):
		self.X = 0.0
		self.Y = 0.0
		self.Z = 0.0
		self.dX = 0.0
		self.dY = 0.0
		self.dZ = 0.0
		
		self.eml = 0.0
		
		self.type = "" # workplane or view 
		
		self.wavelength_data = range(380, 785, 5)
		self.spectral_data = []
	
	def get_spectral_str(self):
		str_list = ["%.12f" % sd for sd in self.spectral_data]
		return ",".join(str_list)
	
	def get_data(self):
		spectral_list = []
		for i in range( len(self.wavelength_data) ):
			spectral_list.append( (self.wavelength_data[i], self.spectral_data[i]) )
		return spectral_list
	
def read_alfa_file(path):
	"""Reads in all of the spectra and associated information from an ALFA export file."""
	global spectral_data
	
	file_spectral_data = []
	
	with open(path, 'r') as f:
# 		line = f.readline() # read first line 
	
# 		while "WORKPLANE SENSORS" not in line: # skip to workplane sensors
# 			line = f.readline()
		
		header = f.readline() # read column header
		header_parts = header.rstrip().split(',')
		wl_values = [int(i) for i in header_parts[19:]]
		
		
# 		while True: # read until blank line
# 			line = f.readline()
# 			if len(line) < 5:
# 				break
# 		f.readline() # skip this column header
		
		for line in f: # read until end 
			parts = line.rstrip().split(',')
   			#eml = float(parts[6])
			xyz_dir = [str(i) for i in parts[:12]]
			sensor_spectra = []
			# for i in parts[19:]:
			# 	print(i)
			# 	sensor_spectra.append(float(i))
			sensor_spectra = [float(i) for i in parts[19:]]
			this_spectral_data = spectral_data()
			this_spectral_data.date = xyz_dir[0]
			this_spectral_data.time = xyz_dir[1]
# # 			this_spectral_data.Z = xyz_dir[2]
			this_spectral_data.dX = xyz_dir[8]
			this_spectral_data.dY = xyz_dir[9]
			this_spectral_data.dZ = xyz_dir[10]
			
# # 			this_spectral_data.eml = eml
			
			this_spectral_data.type = "workplane"
			
# 			this_spectral_data.date = this_spectral_data.date
			
			this_spectral_data.wavelength_data = wl_values
			
			this_spectral_data.spectral_data.extend(sensor_spectra)
			
			file_spectral_data.append(this_spectral_data)
			
# 		while "VIEW SENSORS" not in line: # skip to workplane sensors
# 			line = f.readline()	
		
# 		f.readline() # skip this column header
		
# 		for line in f: # read until end 
# # 			parts = line.rstrip().split(',')
# # 			eml = float(parts[6])
# # 			xyz_dir = [float(i) for i in parts[:6]]
# 			sensor_spectra = [float(i) for i in parts[9:]]
			
# 			this_spectral_data = spectral_data()
			
# # 			this_spectral_data.X = xyz_dir[0]
# # 			this_spectral_data.Y = xyz_dir[1]
# # 			this_spectral_data.Z = xyz_dir[2]
# # 			this_spectral_data.dX = xyz_dir[3]
# # 			this_spectral_data.dY = xyz_dir[4]
# # 			this_spectral_data.dZ = xyz_dir[5]
			
# # 			this_spectral_data.eml = eml
			
# 			this_spectral_data.type = "view"
			
# 			this_spectral_data.wavelength_data = wl_values
			
# 			this_spectral_data.spectral_data.extend(sensor_spectra)
			
# 			file_spectral_data.append(this_spectral_data)
			
	
	return file_spectral_data
    
with open('%s_workplane_cie.csv' % output_prefix, 'w') as result_workplane:
# 	with open('%s_view_cie.csv' % output_prefix, 'w') as result_view:
		
# 		result_view.write("ViewSensor,X,Y,Z,dX,dY,dZ,Illuminance[p.lx],MelanopicIlluminance[m.lx],M/P,S-opicIrrad[W/m2],M-opicIrrad[W/m2],L-opicIrrad[W/m2],RhodopicIrrad[W/m2],MelanopicIrrad[W/m2],S-opicELR[mW/lm],M-opicELR[mW/lm],L-opicELR[mW/lm],RhodopicELR[mW/lm],MelanopicELR[mW/lm],S-opicDER[mW/lm],M-opicDER[mW/lm],L-opicDER[mW/lm],RhodopicDER[mW/lm],MelanopicDER[mW/lm],S-opicEDI[lx],M-opicEDI[lx],L-opicEDI[lx],RhodopicEDI[lx],MelanopicEDI[lx]\n")
		
	result_workplane.write("Date,Time,X.Original,Y.Original,Z.Original,Illuminance[p.lx],MelanopicIlluminance[m.lx],M/P,S-opicIrrad[W/m2],M-opicIrrad[W/m2],L-opicIrrad[W/m2],RhodopicIrrad[W/m2],MelanopicIrrad[W/m2],S-opicELR[mW/lm],M-opicELR[mW/lm],L-opicELR[mW/lm],RhodopicELR[mW/lm],MelanopicELR[mW/lm],S-opicDER[mW/lm],M-opicDER[mW/lm],L-opicDER[mW/lm],RhodopicDER[mW/lm],MelanopicDER[mW/lm],S-opicEDI[lx],M-opicEDI[lx],L-opicEDI[lx],RhodopicEDI[lx],MelanopicEDI[lx]\n")
					
	if os.path.exists(input_file):
		file_spectral_data = read_alfa_file(input_file)
			
		view_cnt = 1
		wp_cnt = 1
		for sd in file_spectral_data:
			ill, mel_ill, m_over_p, opic_irrads, opic_elrs, opic_ders, opic_edis = calc_s026(sd.get_data())
				
			if sd.type == "workplane":
# 				result_workplane.write("Workplane %i," % wp_cnt)
# 				result_workplane.write("%.8f,%.8f,%.8f,%.8f,%.8f,%.8f," % (sd.X, sd.Y, sd.Z, sd.dX, sd.dY, sd.dZ))
				result_workplane.write(sd.date+",")
				result_workplane.write(sd.time+',')
				result_workplane.write(sd.dX+',')
				result_workplane.write(sd.dY+',')
				result_workplane.write(sd.dZ+',')
				result_workplane.write("%.1f,%.1f,%.6f," % (ill, mel_ill, m_over_p))
				result_workplane.write("%.3f,%.3f,%.3f,%.3f,%.3f," % (opic_irrads["sc"], opic_irrads["mc"], opic_irrads["lc"], opic_irrads["rh"], opic_irrads["mel"]))
				result_workplane.write("%.6f,%.6f,%.6f,%.6f,%.6f," % (opic_elrs["sc"], opic_elrs["mc"], opic_elrs["lc"], opic_elrs["rh"], opic_elrs["mel"]))
				result_workplane.write("%.6f,%.6f,%.6f,%.6f,%.6f," % (opic_ders["sc"], opic_ders["mc"], opic_ders["lc"], opic_ders["rh"], opic_ders["mel"]))
				result_workplane.write("%.1f,%.1f,%.1f,%.1f,%.1f\n" % (opic_edis["sc"], opic_edis["mc"], opic_edis["lc"], opic_edis["rh"], opic_edis["mel"]))
				wp_cnt += 1
