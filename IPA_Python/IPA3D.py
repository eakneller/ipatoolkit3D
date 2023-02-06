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
import os
import time
import psutil
import print_funcs
import data_exporter
import BasinModel
import burial_history
import flexural_backstripping
import flexural_salt_restoration
import stretching_inversion
import litho_therm
import charge_to_trap
import thermal_history
import maturity
import source_kinetics
import hc_generation
import ipa1D


def IPA3D_MAIN(
                iuse_max_state, script_path, input_path, output_path, 
                itype3D, inode, jnode, input1D_dict, riskmodel
):
    # Define local inputs
    # Set to zero to reduce output 
    ioutput_main = 1 
    # Set to zero to not extract at well locations
    iextract_at_wells = 1
    # Change active directory to script directory
    os.chdir(script_path)
    print("Current working directory : ", os.getcwd())
    process = psutil.Process(os.getpid())
    print ("Starting Mem (MB) : ",process.memory_info().rss/1e6)
    # Initialize Model
    tt1 = time.time()     
    model = BasinModel.BasinModel(input_path, output_path, script_path,
                                  itype3D, inode, jnode, input1D_dict)
    model.load_main_input_file()
    model.clean_inputs(riskmodel, ioutput_main)
    model.initialize_model(riskmodel, ioutput_main, process)        
    strat_input_dict = model.strat_input_dict
    param_input_dict = model.param_input_dict    
    tt2 = time.time()
    print_funcs.print_finfo(ioutput_main, process, 
                            "Initialized model and loaded maps", tt2-tt1)
    # Print main model parameters
    print_funcs.print_params(param_input_dict)
    print_funcs.print_strat(strat_input_dict)
    print_funcs.print_deltas(riskmodel)
    # Initialize source kinetics and explusion parameters
    srckinetics = source_kinetics.Srckinetics(model)
    srckinetics.load_kinetics()
    model.imass_gen = srckinetics.imass_gen
    # Calculate burial history
    burial_history.burial_history(ioutput_main, process, model)
    data_exporter.export_initial_burial_history(ioutput_main, process, model)
    # Invert for crustal stretching factors using a single event
    (
     xth_xy, moho_xy_np, 
     moho_twt_xy_np, delta_best_fit_xy
    ) = stretching_inversion.stretching_inversion(ioutput_main, process, model)
    data_exporter.export_stretching_maps(ioutput_main, process, model,
                                        delta_best_fit_xy, xth_xy)       
    data_exporter.export_moho_maps(ioutput_main, process, model, 
                                   moho_xy_np, moho_twt_xy_np)
    # Update rift parameters if in 1D mode
    xth_xy = ipa1D.rift_params_1D(model, ioutput_main, process, xth_xy)
    # Load crustal thickness if inversion is not used
    xth_xy = model.load_crustal_thick(xth_xy, riskmodel, ioutput_main, process)
    # Calculate forward subsidence and heat flow history
    litho_therm.calculate_sub_and_hf_maps(ioutput_main, process, model, xth_xy)   
    data_exporter.export_sub_and_hf_maps(model, ioutput_main, process)   
    data_exporter.export_interpolated_sub_hf_maps(model, ioutput_main, process)
    # Perform flexural backstripping if option is selected
    flexural_backstripping.flexural_backstripping(model, ioutput_main, process)
    # Perform flexural salt restoration if option is selected    
    flexural_salt_restoration.salt_restoration(model, ioutput_main, process)
    # Calculate residual subsidence and update paleo-water depth
    burial_history.residual_subsidence_and_pwd(ioutput_main, process, model)    
    data_exporter.export_updated_burial_history(ioutput_main, process, model)
    # Calculate thermal history in sediments
    thermal_history.thermal_history(model, process, ioutput_main)    
    data_exporter.export_temperature_maps(model, ioutput_main, process)
    # Calculate thermal maturity (EasyRo)
    maturity.maturity_history(model, process, ioutput_main)    
    maturity.calc_all_LOM_maps(model, process, ioutput_main)    
    data_exporter.export_maturity_maps(model, process, ioutput_main)
    # Calculate hydrocarbon mass generation, oil and gas split and expulsion
    (
    OILAPI_list, 
    GASGRAV_list
    ) = hc_generation.hc_generation(srckinetics, model, ioutput_main, process)   
    data_exporter.export_gen_maps_and_expulsion(model, ioutput_main, process)
    # Extract 1D model if in 1D mode
    ipa1D.extract_for_1D_model(model, ioutput_main, process)
    # Extract and export at well locations
    data_exporter.export_at_well_locations(model, ioutput_main, process, 
                                           iextract_at_wells)
    # Move maps to directories
    model.move_maps(ioutput_main)
    # Perform charge to trap calculations using yield history, evolving
    # fetch area polygons and fluid property model
    charge_to_trap.charge_to_trap(model, riskmodel, ioutput_main, process,
                                  OILAPI_list, GASGRAV_list)
                
                