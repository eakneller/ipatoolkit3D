# -*- coding: utf-8 -*-
"""

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Erik A. Kneller
"""
import sys
import math
import numpy as np
import map_tools
import fileIO

iuse_args = 1

if iuse_args == 0:
    script_path = "C:\\Users\\eaknell\\Desktop\\IPAtoolkit3D_1.0\\IPA_Python"
else:
    script_path = sys.argv[1]
    
if iuse_args == 0:
    ipa_main_path = "C:\\Users\\eaknell\\Desktop\\IPAtoolkit3D_1.0"
    ipa_python_path = ipa_main_path + "\\IPA_Python"
    model_name = "TorresArch2022"
    input_path = ipa_main_path + "\\Input_Maps\\" + model_name + "\\"
    output_path = "D:\\IPAtoolkit3D\\" + model_name + "\\"
    file_name_new = "background_hf_r5km"
    file_name_ref = "bghf_35mWm2.dat"
    Rsearch = 5000.0
else:
    input_path = sys.argv[2] + "\\"
    output_path = sys.argv[3] + "\\"
    file_name_new = sys.argv[4]
    file_name_ref = sys.argv[5]
    Rsearch = float(sys.argv[6])

# Read reference heat flow map for Zmap structure
AOI_np_ini = np.zeros((1,1))
(
     ref_xy, nx, ny, dx, dy, 
     xmin, xmax, ymin, ymax, AOI_np_L
) = map_tools.read_ZMAP(input_path, file_name_ref, AOI_np_ini)
AOI_np = np.zeros((nx,ny))
for i in range(nx):
    for j in range(ny):
        AOI_np[i][j] = 1.0
print("nx, ny, dx, dy, xmin, xmax, ymin, ymax : ", 
      nx, ny, dx, dy, xmin, xmax, ymin, ymax)

# Read background heat flow calibration file
input_file_path = input_path + "ipa_wells.csv"
nwells, wells_dict = fileIO.read_well_file_csv(input_file_path)
wkeys = list(wells_dict.keys())

# Loop over all wells and calculate average bghf
icount = 0
sumit = 0.0
for key in wkeys:
    well_name = key
    bghf = wells_dict[key][4]    
    sumit = sumit + bghf    
    icount = icount + 1
bghf_avg = sumit / float(icount)

print("Average background heat flow : ", bghf_avg, " mW/m2")
# Loop over all x-y locations and calculate deltas
for i in range(nx):
    for j in range(ny): # Rows
        x = xmin + dx*float(i)
        y = ymin + dy*float(j)
        delta = 0.0
        sumit1 = 0.0
        sumit2 = 0.0
        for key in wkeys:
            well_name = key
            xx = wells_dict[key][0]
            yy = wells_dict[key][1]
            bghf = wells_dict[key][4]
            ddx = x - xx
            ddy = y - yy
            dist = math.sqrt(ddx*ddx + ddy*ddy)
            w_i = math.exp(-dist/Rsearch)
            delta_i = bghf - bghf_avg
            sumit1 = sumit1 + w_i*delta_i
            sumit2 = sumit2 + w_i
        if sumit2 > 0.0:
            delta = sumit1/sumit2
        else:
            delta = 0.0
        bghf_new = bghf_avg + delta
        ref_xy[i][j] = bghf_new

map_tools.make_output_file_ZMAP_v4(
                                input_path, file_name_new, ref_xy, 
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
