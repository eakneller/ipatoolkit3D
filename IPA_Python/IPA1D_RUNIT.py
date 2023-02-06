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
import os
import time
import shutil
import RiskModel


input1D_dict={} 
lith1D_dict={}
rockID_dict={}
lith1D_final_dict={}
irun_basin_maker = 1 # run backstripping and forward heat flow modeling
irun_grid_proc = 0 # run grid processing tools
irun_misc_test = 0 # run tests
script_path = sys.argv[1]
#script_path = "C:\\Users\\eaknell\\Desktop\\Map_Based_IPA_alpha2.5\\IPA_Multi1D\\IPA_Python"
print("script_path: ", script_path)
itype3D = 0
inode = 1
jnode = 1
path1D = script_path +"\\ipa_1D\\" 
input_path = script_path +"\\ipa_1D\\input\\"
output_path = script_path +"\\ipa_1D\\output\\"
print("script_path, input_path, output_path : ", 
      script_path, input_path, output_path)
#model1D_direc = "C:\\Users\\eaknell\\Desktop\\IPA_toolkit_beta3.9\\models\\"
#model1D_name = "TEST1D_noero.csv"
model1D_direc = path1D
model1D_name = "IPA1Dinput.csv"
csv1D_path = model1D_direc + model1D_name
#main_direc = os.path.dirname(script_path)
#print("main_direc : ", main_direc)
icheck = 0
# Check to see if output directory exists
bcheck = os.path.isdir(output_path)
# Remove pre-existing output directory
if bcheck == True:
    shutil.rmtree(output_path)
# Make a new output directory
try:
    os.mkdir(output_path)
except:
    icheck = 1
print("icheck : ", icheck)    
# Check to see if input directory exists
bcheck = os.path.isdir(input_path)
# Remove pre-existing input directory
if bcheck == True:
    shutil.rmtree(input_path)
# Make a new input directory
try:
    os.mkdir(input_path)
except:
    icheck = 1
print("icheck : ", icheck)
if icheck == 0:
    # Copy csv input files to output directory
    f1 = "Risk_Sensitivity.csv"
    shutil.copy(path1D + f1, input_path + f1)
    f1 = "BatchRun.csv"
    shutil.copy(path1D + f1, input_path + f1)    
    
    tt1 = time.clock()
    run_stype = 'single'
    nruns_monte_carlo = 1
    ioutput_type = 0
    src_index_list_bulk_gor = [-1]
    #iuse_max_state = 0
    iuse_max_state = 1
    itype3D = 0
    RiskModel.initiate_risk_runs(
                                script_path, input_path, output_path, 
                                run_stype, nruns_monte_carlo, ioutput_type, 
                                iuse_max_state, src_index_list_bulk_gor, 
                                itype3D
                                )
    tt2 = time.clock()
    print("Total time for risk run cpu(s) : ", tt2-tt1)
