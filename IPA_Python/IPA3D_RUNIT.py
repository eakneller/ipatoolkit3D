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
import RiskModel


""" Main driver script for IPA toolkit

    Version 1.24

    This script can be used to manually run IPA by setting iuse_args = 0. 
    For this option the user must define paths below  to the main scripts 
    directory, an input file directory and an output directory.

    Alternatively, the script can be run via command line with arguments 
    specifying paths.
    
    Input Description
    -----------------
    script_path : string
        Path to python code directory IPA_Python. For example,
            "C:\\Users\\username\\Desktop\\IPA_Toolkit\\IPA_Python"
    
    input_path : string
        Path to the input file directory. For example,
            "C:\\Users\\username\\Desktop\\IPA_Toolkit\\MyModel\\"
        
    output_path : string
        Path where output files will be sent to. For example:
            "C:\\Users\\username\\Desktop\\IPA_Toolkit\\Output\\"
    
    iuse_args : integer
        Controls how paths are defined for the python executable, input 
        directory and output directory
        
         0 = use user defined paths
         1 = use paths defined by system arguments. This option should be used 
             when running code from the Excel user interface.

         When running multi-scenario analysis using multirun_stype = 
         "monte_carlo", "scenario" or "highlow" set iuse_args to 0.

    multirun_stype : string
        Controls the type of run. The following options are available:
        
        "single" : only one run
        
        "highlow" : run high and lowside scenarios using delta_min and delta_
                    max parameters for variable keys in HL_dict. Number of 
                    runs is determined from total number of combinations. 
                    Note that key enteries can be commented out in HL_dict 
                    below to exclude parameters.
        
        "monte_carlo" : run a Monte Carlo simulations using delta_min, 
                        delta_max parameters, pdf type and distribution 
                        parameters for each variable. Number of runs is 
                        defined by nruns.
        
         "scenario" : Run a discrete number of scenarios based on the size of 
                     list associated with the scenario_delta_dict dictionary.
    
    ioutput_type : integer
        Integer that controls the type of output:
            0 = send output to terminal
            1 = send to file log.txt
        
    src_index_list_bulk_gor : list of integers
        Controls source combinations when calculating bulk GOR:
            [-1]  = consider all sources when calculating bulk gor
            [0]   = only the first source starting from the oldest
            [1]   = only the second source
            [0,1] = only first and second sources ...
    
        This list is used when iuse_arg == 0 and for high-low multi-runs 
        (multirun_stype == highlow). When multirun_stype == "scenario" ar 
        "monte_carlo"then src_index_list_bulk_gor is defined from input 
        dictionaries below.
    
    nruns_monte_carlo : integer
        Controls the number of risk runs for Monte Carlo Simulation. 
        Set nruns to 1 for single base case model run. If multirun_stype = 
        "highlow" then nruns will be calculated below. nruns will be set to 
        zero if iuse_args == 1.

    iuse_max_state : integer
        Integer that controls the definition of erosion maps. Are erosion maps
        in the present-day compaction state (option 1) or erosion compaction 
        state (option 0):
             0 : do not compact eroded thicknesses into maximum present-day 
                 compaction state for shift maps
             1 : compact eroded thicknesses into maximum present-day compaction 
                 state for shift maps
"""
iuse_args = 1
multirun_stype = "single"
nruns_monte_carlo = 3
script_path = "C:\\Users\\eaknell\\Desktop\\IPAtoolkit3D_1.0\\IPA_Python"
input_path = ("C:\\Users\\eaknell\\Desktop\\IPAtoolkit3D_1.0\\"
              "Input_Maps\\TorresArch2022\\")
output_path = ("C:\\Users\\eaknell\\Desktop\\IPAtoolkit3D_1.0\\"
               "Output_Maps\\TorresArch2022\\")
ioutput_type = 1
src_index_list_bulk_gor = [-1]
iuse_max_state = 0
itype3D = 1

if __name__ == "__main__":
    if iuse_args == 1:
        script_path = sys.argv[1]  
        input_path = sys.argv[2] + "\\"
        output_path = sys.argv[3] + "\\"
        multirun_stype = sys.argv[4]
        nruns_monte_carlo = int(sys.argv[5])
        ioutput_type = int(sys.argv[6])
    itype3D = 1
    RiskModel.initiate_risk_runs(
                        script_path, input_path, output_path, multirun_stype,
                        nruns_monte_carlo, ioutput_type, iuse_max_state,
                        src_index_list_bulk_gor, itype3D
                    )