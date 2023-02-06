# -*- coding: utf-8 -*-
import os
import RiskModel  
""" IPA3D Batch Run Script

    This script can be used to run IPA3D in batch mode.
    
    Input Description
    -----------------    
    ipa_main_path : string
        Path to main IPA3D directory.
    
    ipa_python_path : string
        Path to python code directory IPA_Python.
    
    input_path : string
        Path to the input file directory.
        
    output_path : string
        Path where output files will be sent to.
    
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
    
        This list is used as default for high-low multi-runs 
        (multirun_stype == highlow). When multirun_stype == "scenario" ar 
        "monte_carlo"then src_index_list_bulk_gor is defined from input 
        files defined by the user.
    
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

if __name__ == "__main__":
    ipa_main_path = r'C:\Users\eaknell\Desktop\IPAtoolkit3D_1.0'
    ipa_python_path = os.path.join(ipa_main_path, "IPA_Python")
    model_name = "ColoradoBasin"
    input_path = os.path.join(ipa_main_path, "Input_Maps", model_name)
    output_path = os.path.join(r'D:\IPAtoolkit3D', model_name + "_test1")
    
    run_stype = 'single'
    nruns_monte_carlo = 1
    ioutput_type = 0
    src_index_list_bulk_gor = [-1]
    iuse_max_state = 0
    itype3D = 1
    
    RiskModel.initiate_risk_runs(
                                 ipa_python_path, input_path, output_path, 
                                 run_stype, nruns_monte_carlo, ioutput_type, 
                                 iuse_max_state, src_index_list_bulk_gor,
                                 itype3D
                                 )