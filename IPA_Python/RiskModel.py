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
import sys
import numpy as np
import time
import IPA3D
import fileIO
import print_funcs
import risk_tools
import map_tools
import ipa1D

class RiskModel():
    
    
    def __init__(self, input_path, output_path, 
                 output_path_root, output_dir_name,
                 ioutput_type, nruns_monte_carlo, 
                 multirun_stype, src_index_list_bulk_gor
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.ioutput_type = ioutput_type
        self.output_dir_name = output_dir_name
        self.output_path_root = output_path_root
        self.nruns_monte_carlo = nruns_monte_carlo
        self.multirun_stype = multirun_stype
        self.src_index_list_bulk_gor = src_index_list_bulk_gor

    def initialize_risk_maps(self):    
        self.HL_dict = {}
        input_file_path = os.path.join(self.input_path, "Risk_Sensitivity.csv")
        (
        self.srtype, 
        nruns_dum, 
        self.pdf_dict, 
        self.discrete_dict, 
        self.src_index_list_bulk_gor_scenario1,
        self.src_index_list_bulk_gor_scenario2, 
        self.src_index_list_bulk_gor_scenario3
        ) = fileIO.read_RiskSensitivty_file_csv(input_file_path)        
        print("Reading main input file")
        # Read main input file
        input_dict = {}
        param_input_dict = {}
        input_file_path = os.path.join(self.input_path, "ipa_input.csv")
        fileIO.read_main_input_file_csv(
                                        input_file_path, 
                                        input_dict, 
                                        param_input_dict
                                    )        
        print("Getting name of PWD map that controls resolution (master grid)")
        # Inititalize AOI numpy array
        AOI_np_ini = np.zeros((1,1))
        event_ID_list_bs = list(input_dict.keys())
        nevents_bs = len(event_ID_list_bs)
        event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
        file_name_master_grid = input_dict[event_ID_last_bs][5]        
        print("Name of master grid: ", file_name_master_grid)
        (
        self.master_xy, self.nx, self.ny, self.dx, self.dy, 
        self.xmin, self.xmax, self.ymin, self.ymax, 
        self.AOI_np
        ) = map_tools.read_ZMAP(
                        self.input_path, file_name_master_grid, AOI_np_ini)
        self.scenario_delta_maps_dict = {}
        if self.multirun_stype == "scenario":
            print("Transforming scenario delta entries into maps")    
            keys = list(self.scenario_delta_dict.keys())
            for key in keys:      
                if key not in ["kinetics", "gas_frac", 
                               "polar_frac", "src_index"
                ]:        
                    nvals = len(self.scenario_delta_dict[key])
                    #print("Working on scenario key: ", key, " nvals ", nvals)        
                    for i in range(nvals):            
                        val = self.scenario_delta_dict[key][i]            
                        #print("     >>>> i, scenario value: ", i, val)            
                        delta_xy_tmp_np = np.copy(self.master_xy)
                        #nx = delta_xy_tmp_np.shape[0]
                        #ny = delta_xy_tmp_np.shape[1]
                        risk_tools.update_scalars(self.nx, self.ny, 
                                                  delta_xy_tmp_np, 
                                                  val, self.AOI_np)            
                        if i == 0:
                            self.scenario_delta_maps_dict[key] = [
                                                    np.copy(delta_xy_tmp_np)
                                                    ]
                        else:
                            self.scenario_delta_maps_dict[key].append(
                                                    np.copy(delta_xy_tmp_np)
                                                    )
        self.risk_maps_path = os.path.join(self.input_path,"risk_maps")
        if self.multirun_stype in ["monte_carlo", "highlow"]:    
            print("Creating risk maps for Monte Carlo and High-Low")
            sname = "bghf"
            (
            self.bghf_min_xy_np, 
            self.bghf_max_xy_np, 
            self.bghf_delta_nxy_np
            ) = self.make_risk_map_set(sname)
            sname = "xth"
            (
            self.xth_min_xy_np, 
            self.xth_max_xy_np, 
            self.xth_delta_nxy_np
            ) = self.make_risk_map_set(sname)   
            sname = "mantlefac"
            (
            self.mantlefac_min_xy_np, 
            self.mantlefac_max_xy_np, 
            self.mantlefac_delta_nxy_np
            ) = self.make_risk_map_set(sname)   
            sname = "def_age"
            (
            self.defage_min_xy_np, 
            self.defage_max_xy_np, 
            self.defage_delta_nxy_np
            ) = self.make_risk_map_set(sname)        
            sname = "hf_red_fac"
            (
            self.hfred_min_xy_np, 
            self.hfred_max_xy_np, 
            self.hfred_delta_nxy_np
            ) = self.make_risk_map_set(sname)  
            sname = "por_decay_depth"
            (
            self.pdecay_min_xy_np, 
            self.pdecay_max_xy_np, 
            self.pdecay_delta_nxy_np
            ) = self.make_risk_map_set(sname)  
            sname = "surf_por"
            (
            self.spor_min_xy_np, 
            self.spor_max_xy_np, 
            self.spor_delta_nxy_np
            ) = self.make_risk_map_set(sname)        
            sname = "TOC"
            (
            self.toc_min_xy_np, 
            self.toc_max_xy_np, 
            self.toc_delta_nxy_np
            ) = self.make_risk_map_set(sname)    
            sname = "HI"
            (
            self.hi_min_xy_np, 
            self.hi_max_xy_np, 
            self.hi_delta_nxy_np
            ) = self.make_risk_map_set(sname)    
            sname = "thick"
            (
            self.srcthick_min_xy_np, 
            self.srcthick_max_xy_np, 
            self.srcthick_delta_nxy_np
            ) = self.make_risk_map_set(sname)    
            sname = "oilapi"
            (
            self.oilapi_min_xy_np, 
            self.oilapi_max_xy_np, 
            self.oilapi_delta_nxy_np
            ) = self.make_risk_map_set(sname)        
            sname = "gasgrav"
            (
            self.gasgrav_min_xy_np, 
            self.gasgrav_max_xy_np, 
            self.gasgrav_delta_nxy_np
            ) = self.make_risk_map_set(sname)     
            sname = "porth"
            (
            self.porth_min_xy_np, 
            self.porth_max_xy_np, 
            self.porth_delta_nxy_np
            ) = self.make_risk_map_set(sname)  
        else:
            nx = self.nx
            ny = self.ny
            self.bghf_min_xy_np = np.zeros((nx,ny))
            self.bghf_max_xy_np = np.zeros((nx,ny))
            self.bghf_delta_nxy_np = np.zeros((1, nx,ny))
            self.xth_min_xy_np = np.zeros((nx,ny))
            self.xth_max_xy_np = np.zeros((nx,ny))
            self.xth_delta_nxy_np = np.zeros((1, nx,ny))
            self.mantlefac_min_xy_np = np.zeros((nx,ny))
            self.mantlefac_max_xy_np = np.zeros((nx,ny))
            self.mantlefac_delta_nxy_np = np.zeros((1, nx,ny))
            self.defage_min_xy_np = np.zeros((nx,ny))
            self.defage_max_xy_np = np.zeros((nx,ny))
            self.defage_delta_nxy_np = np.zeros((1, nx,ny))
            self.hfred_min_xy_np = np.zeros((nx,ny))
            self.hfred_max_xy_np = np.zeros((nx,ny))
            self.hfred_delta_nxy_np = np.zeros((1, nx,ny))
            self.pdecay_min_xy_np = np.zeros((nx,ny))
            self.pdecay_max_xy_np = np.zeros((nx,ny))
            self.pdecay_delta_nxy_np = np.zeros((1, nx,ny))
            self.spor_min_xy_np = np.zeros((nx,ny))
            self.spor_max_xy_np = np.zeros((nx,ny))
            self.spor_delta_nxy_np = np.zeros((1, nx,ny))
            self.toc_min_xy_np = np.zeros((nx,ny))
            self.toc_max_xy_np = np.zeros((nx,ny))
            self.toc_delta_nxy_np = np.zeros((1, nx,ny))
            self.hi_min_xy_np = np.zeros((nx,ny))
            self.hi_max_xy_np = np.zeros((nx,ny))
            self.hi_delta_nxy_np = np.zeros((1, nx,ny))
            self.srcthick_min_xy_np = np.zeros((nx,ny))
            self.srcthick_max_xy_np = np.zeros((nx,ny))
            self.srcthick_delta_nxy_np = np.zeros((1, nx,ny))
            self.oilapi_min_xy_np = np.zeros((nx,ny))
            self.oilapi_max_xy_np = np.zeros((nx,ny))
            self.oilapi_delta_nxy_np = np.zeros((1, nx,ny))
            self.gasgrav_min_xy_np = np.zeros((nx,ny))
            self.gasgrav_max_xy_np = np.zeros((nx,ny))
            self.gasgrav_delta_nxy_np = np.zeros((1, nx,ny))
            self.porth_min_xy_np = np.zeros((nx,ny))
            self.porth_max_xy_np = np.zeros((nx,ny))
            self.porth_delta_nxy_np = np.zeros((1, nx,ny))  
        pdf_stype = self.pdf_dict["adsth"][0]
        a = self.pdf_dict["adsth"][1]
        b = self.pdf_dict["adsth"][2]
        c = self.pdf_dict["adsth"][3]
        vmin_adsth = self.pdf_dict["adsth"][4]
        vmax_adsth = self.pdf_dict["adsth"][5]    
        self.delta_vec_adsth = risk_tools.random_values_from_pdf(
                                            pdf_stype, a, b, c, 
                                            vmin_adsth, vmax_adsth, self.nruns)
        pdf_stype = self.pdf_dict["inert"][0]
        a = self.pdf_dict["inert"][1]
        b = self.pdf_dict["inert"][2]
        c = self.pdf_dict["inert"][3]
        vmin_inert = self.pdf_dict["inert"][4]
        vmax_inert = self.pdf_dict["inert"][5]    
        self.delta_vec_inert = risk_tools.random_values_from_pdf(
                                            pdf_stype, a, b, c, 
                                            vmin_inert, vmax_inert, self.nruns)
        pdf_stype = self.pdf_dict["temp_elastic"][0]
        a = self.pdf_dict["temp_elastic"][1]
        b = self.pdf_dict["temp_elastic"][2]
        c = self.pdf_dict["temp_elastic"][3]
        vmin_Telastic = self.pdf_dict["temp_elastic"][4]
        vmax_Telastic = self.pdf_dict["temp_elastic"][5]    
        self.delta_vec_Telastic = risk_tools.random_values_from_pdf(
                                    pdf_stype, a, b, c,
                                    vmin_Telastic, vmax_Telastic, self.nruns)
        print("Creating high-low dictionary")    
        self.HL_dict["bghf"] = [self.bghf_min_xy_np, 
                                self.bghf_max_xy_np] 
        self.HL_dict["def_age"] = [self.defage_min_xy_np, 
                                   self.defage_max_xy_np]
        self.HL_dict["hf_red_fac"] = [self.hfred_min_xy_np, 
                                      self.hfred_max_xy_np]
        self.HL_dict["temp_elastic"] = [vmin_Telastic, vmin_Telastic]
        self.HL_dict["por_decay_depth"] = [self.pdecay_min_xy_np, 
                                           self.pdecay_max_xy_np]
        self.HL_dict["surf_por"] = [self.spor_min_xy_np, self.spor_max_xy_np]    
        self.HL_dict["TOC"] = [self.toc_min_xy_np, self.toc_max_xy_np]
        self.HL_dict["HI"] = [self.hi_min_xy_np, self.hi_max_xy_np]
        self.HL_dict["thick"] = [self.srcthick_min_xy_np, 
                                 self.srcthick_max_xy_np]
        self.HL_dict["kinetics"] = [-1, 1]
        self.HL_dict["gas_frac"] = [-1, 1]
        self.HL_dict["polar_frac"] = [-1, 1]
        

    def make_risk_map_set(self, sname):
        print(">> Making mapset ", sname, " with nruns = ", self.nruns)
        zeros_xy_np = np.zeros((self.nx, self.ny))
        tt1 = time.clock()
        file_name = self.pdf_dict[sname][4]
        if os.path.isfile(
                        os.path.join(self.risk_maps_path, file_name)) == True:
            smin_xy_np = map_tools.read_ZMAP_interp_to_master(
                                -99999.0, self.master_xy, self.risk_maps_path, 
                                file_name, self.nx, self.ny, self.xmin, 
                                self.ymin, self.dx, self.dy, self.AOI_np, 
                                zeros_xy_np)
        else:
            smin_xy_np = np.zeros((self.nx, self.ny))   
        file_name = self.pdf_dict[sname][5]
        if os.path.isfile(
                        os.path.join(self.risk_maps_path, file_name)) == True:
            smax_xy_np = map_tools.read_ZMAP_interp_to_master(
                               -99999.0, self.master_xy, self.risk_maps_path, 
                                file_name, self.nx, self.ny, self.xmin, 
                                self.ymin, self.dx, self.dy, self.AOI_np, 
                                zeros_xy_np)
        else:
            smax_xy_np = np.zeros((self.nx, self.ny))
        delta_nxy_np = np.zeros((self.nruns, self.nx, self.ny))
        tt2 = time.clock()
        print(">> Read maps and created arrays: cpu(s) ", tt2-tt1)
        tt1 = time.clock()
        risk_tools.assign_random_numbers_to_map(
                self.nx, self.ny, 
                smin_xy_np, smax_xy_np, 
                delta_nxy_np,
                self.nruns, sname, 
                self.pdf_dict, 
                self.AOI_np)
        tt2 = time.clock()
        print(">> Assigned random numbers: cpu(s) : ", tt2-tt1)
        return smin_xy_np, smax_xy_np, delta_nxy_np
    

    def initialize_risk_runs(self):
        print("multirun_stype : ", self.multirun_stype)
        self.fout_multirun = None
        # Multi-scenario dictionaries
        self.pdf_dict = {}
        self.discrete_dict = {}
        self.HL_dict = {}
        self.risk_var_names = [
                          "bghf", 
                          "xth", 
                          "mantlefac", 
                          "def_age", 
                          "hf_red_fac",
                          "por_decay_depth", 
                          "surf_por", 
                          "TOC", 
                          "HI", 
                          "thick", 
                          "oilapi",
                          "gasgrav",
                          "porth",
                          "temp_elastic",
                          "adsth",
                          "inert",
                          "kinetics",
                          "gas_frac","polar_frac"
                         ]
        # scenario_delta_dict
        # -------------------
        # This dictionary is used when multirun_stype == "scenario". Each key 
        # refers to a variable and each dictionary entry is a list of delta 
        # values.  
        input_file_path = os.path.join(self.input_path, "BatchRun.csv")
        if os.path.isfile(input_file_path) == True:
            self.scenario_delta_dict = fileIO.read_BatchRun_file_csv(
                                                            input_file_path)
        else:
            print_funcs.print_warning("BatchRun.csv could not be found")
            print_funcs.print_warning("Switching to single")
            self.multirun_stype = "single"
        if self.multirun_stype == "single":        
            self.nruns = 1
        elif self.multirun_stype == "scenario":        
            keys = list(self.scenario_delta_dict.keys())
            nruns_min = 10
            for key in keys:
                list_tmp = self.scenario_delta_dict[key]
                nruns = len(list_tmp)
                if nruns < nruns_min:
                    nruns_min = nruns
            self.nruns = nruns_min
            print_funcs.print_statement("number of scenario runs : " 
                                        + str(self.nruns))        
        elif self.multirun_stype == "monte_carlo":        
            self.nruns = self.nruns_monte_carlo
        print("Initializing risk maps")
        self.initialize_risk_maps()
        self.HL_delta_dict = {}
        if self.multirun_stype == "highlow":      
            nx = self.bghf_min_xy_np.shape[0]
            ny = self.bghf_min_xy_np.shape[1]
            zeros_xy_np = np.zeros((nx, ny))
            (
            self.HL_delta_dict, 
            self.nruns
            ) = risk_tools.build_high_low_delta_dict(
                    zeros_xy_np,
                    self.risk_var_names,
                    self.HL_dict, 
                    self.pdf_dict, 
                    self.discrete_dict
            )
        if self.nruns > 1 and self.multirun_stype == "monte_carlo":
            #print(">> Defining random numbers for all runs for discrete variables")
            keys = list(self.discrete_dict.keys())
            for key in keys:
                p1 = self.discrete_dict[key][0]
                p2 = self.discrete_dict[key][1]
                p3 = self.discrete_dict[key][2]            
                check_sum = p1 + p2 + p3
                if check_sum != 1:
                    p1 = p1/check_sum
                    p2 = p2/check_sum
                    p3 = p3/check_sum            
                delta_vec = risk_tools.random_values_from_discrete(
                                                            p1, p2, p3, nruns) 
                self.discrete_dict[key].append(delta_vec)
        if self.multirun_stype in ["scenario", "highlow", "monte_carlo"]:
            file_name = os.path.join(self.output_path, 
                                     self.multirun_stype + ".csv")
            self.multirun_file_name = file_name
            fout_multirun = open(self.multirun_file_name, 'w')
            strout = "nruns" + "," + str(nruns)
            fout_multirun.write(strout + " \n")
            strout = "input_dir" + "," + self.input_path
            fout_multirun.write(strout + " \n")
            strout = "root_dir" + "," + self.output_path_root
            fout_multirun.write(strout + " \n")
            if self.multirun_stype in ["scenario", "highlow"]:
                tmp_name_list = [
                                    "sub_dir_name", 
                                    "run#", 
                                    "delta_bghf", 
                                    "delta_xth", 
                                    "delta_mantlefac",
                                    "delta_def_age", 
                                    "delta_hf_red_fac", 
                                    "temp_elastic", 
                                    "por_decay_depth", 
                                    "surf_por",
                                    "delta_toc", 
                                    "delta_HI", 
                                    "delta_thick", 
                                    "delta_oilapi", 
                                    "delta_gasgrav",
                                    "delta_porth", 
                                    "delta_adsth", 
                                    "delta_inert", 
                                    "idelta_kinetics",
                                    "idelta_gas_frac", 
                                    "idelta_polar_frac", 
                                    "delta_sat_gor", 
                                    "src_index_list_bulk_gor"
                                ]
            elif self.multirun_stype in ["monte_carlo"]:
                tmp_name_list = [
                                    "dir_name", 
                                    "run#",
                                    "delta_bghf_min_map",
                                    "delta_bghf_max_map",
                                    "delta_bghf_random_val",
                                    "delta_xth_min_map", 
                                    "delta_xth_max_map",
                                    "delta_xth_random_val",
                                    "delta_mantlefac_min_map", 
                                    "delta_mantlefac_max_map",
                                    "delta_mantlefac_random_val",
                                    "def_age_min_map", 
                                    "def_age_max_map",
                                    "def_age_random_val",
                                    "hf_red_fac_min_map", 
                                    "hf_red_fac_max_map",
                                    "hf_red_fac_random_val",
                                    "por_decay_depth_min_map", 
                                    "por_decay_depth_max_map", 
                                    "por_decay_depth_random_val",
                                    "surf_por_min_map", 
                                    "surf_por_max_map",
                                    "surf_por_random_val",
                                    "delta_toc_min_map", 
                                    "delta_toc_max_map",
                                    "delta_toc_random_val",
                                    "delta_HI_min_map", 
                                    "delta_HI_max_map",
                                    "delta_HI_random_val",
                                    "delta_thick_min_map", 
                                    "delta_thick_max_map",
                                    "delta_thick_random_val",
                                    "delta_oilapi_min_map", 
                                    "delta_oilapi_max_map",
                                    "delta_oilapi_random_val",
                                    "delta_gasgrav_min_map",
                                    "delta_gasgrav_max_map",
                                    "delta_gasgrav_random_val",
                                    "delta_porth_min_map",
                                    "delta_porth_max_map",
                                    "delta_porth_random_val"
                                    "delta_Telastic",
                                    "delta_adsth",
                                    "delta_inert",
                                    "idelta_kinetics",
                                    "idelta_gas_frac",
                                    "idelta_polar_frac",
                                    "delta_sat_gor",
                                    "src_index_list_bulk_gor"
                                ]    
            for ii, str_tmp in enumerate(tmp_name_list):
                if ii == 0:
                    strout = str_tmp
                else:
                    strout = strout + "," + str_tmp
            fout_multirun.write(strout + " \n")
            fout_multirun.close()

    def define_inputs_for_risk_run(self, ir):
        if self.multirun_stype == "monte_carlo":
            self.rname = "riskrun"+str(ir)
        elif self.multirun_stype == "scenario":
            self.rname = "scenario"+str(ir)
        elif self.multirun_stype == "highlow":
            self.rname = "highlow"+str(ir)
        elif self.multirun_stype == "single":
            self.rname = "single"+str(ir)
        print("")
        print("")
        print("**************************************************************")
        print("Working on risk run ", ir, " with name ", self.rname, 
              " nruns = ", self.nruns)
        print("**************************************************************")
        print("")
        print("")    
        if self.nruns > 1:
            self.output_path = os.path.join( 
                                     self.output_path_root, 
                                     self.output_dir_name + "_" + self.rname
                                  )
            print("New output_path : ", self.output_path)
        # Check for output path directory. If it cant be found create it.
        if os.path.isdir(self.output_path) == False:
            os.mkdir(self.output_path)
        if self.nruns == 1: 
            # There is only 1 run, use base case model parameters
            self.delta_bghf_xy_np = self.bghf_delta_nxy_np[ir]
            self.delta_xth_xy_np = self.xth_delta_nxy_np[ir]
            self.delta_mantlefac_xy_np = self.mantlefac_delta_nxy_np[ir]
            self.delta_defage_xy_np = self.defage_delta_nxy_np[ir]
            self.delta_hfred_xy_np = self.hfred_delta_nxy_np[ir]
            self.delta_pdecay_xy_np = self.pdecay_delta_nxy_np[ir]
            self.delta_spor_xy_np = self.spor_delta_nxy_np[ir]
            self.delta_toc_xy_np = self.toc_delta_nxy_np[ir]
            self.delta_HI_xy_np = self.hi_delta_nxy_np[ir]
            self.delta_thick_xy_np = self.srcthick_delta_nxy_np[ir]
            self.delta_oilapi_xy_np = self.oilapi_delta_nxy_np[ir]
            self.delta_gasgrav_xy_np = self.gasgrav_delta_nxy_np[ir]
            self.delta_porth_xy_np = self.porth_delta_nxy_np[ir]
            # Telastic_delta_nxy_np[ir] # Not a map
            self.delta_Telastic = 0.0
            # adsth_delta_nxy_np[ir] # Not a map
            self.delta_adsth = 0.0
            # inert_delta_nxy_np[ir] # Not a map
            self.delta_inert = 0.0
            # -1 = early, 0 = normal, 1 = late, 2 = base case model
            self.idelta_kinetics = 2
            # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
            self.idelta_gas_frac = 2
            # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
            self.idelta_polar_frac = 2
            self.delta_sat_gor = 0.0
        else:
            if self.multirun_stype == "scenario":
                # Define delta values
                self.delta_bghf_val = self.scenario_delta_dict["bghf"][ir]
                self.delta_xth_val = self.scenario_delta_dict["xth"][ir]
                self.delta_mantlefac_val = (
                        self.scenario_delta_dict["mantlefac"][ir])
                self.delta_defage_val = self.scenario_delta_dict["def_age"][ir]
                self.delta_hfred_val = (
                        self.scenario_delta_dict["hf_red_fac"][ir])
                self.delta_pdecay_val = (
                        self.scenario_delta_dict["por_decay_depth"][ir])
                self.delta_spor_val = self.scenario_delta_dict["surf_por"][ir]
                self.delta_toc_val = self.scenario_delta_dict["TOC"][ir]
                self.delta_HI_val = self.scenario_delta_dict["HI"][ir]
                self.delta_thick_val = self.scenario_delta_dict["thick"][ir]
                self.delta_oilapi_val = self.scenario_delta_dict["oilapi"][ir]
                self.delta_gasgrav_val = self.scenario_delta_dict["gasgrav"][ir]
                self.delta_Telastic_val = (
                        self.scenario_delta_dict["temp_elastic"][ir])
                self.delta_porth_val = self.scenario_delta_dict["porth"][ir]
                self.delta_adsth_val = self.scenario_delta_dict["adsth"][ir]
                self.delta_inert_val = self.scenario_delta_dict["inert"][ir]                
                self.idelta_kinetics_val = (
                        self.scenario_delta_dict["kinetics"][ir])
                self.idelta_gas_frac_val = (
                        self.scenario_delta_dict["gas_frac"][ir])
                self.idelta_polar_frac_val = (
                        self.scenario_delta_dict["polar_frac"][ir])
                # Define maps
                self.delta_bghf_xy_np = (
                        self.scenario_delta_maps_dict["bghf"][ir])
                self.delta_xth_xy_np = self.scenario_delta_maps_dict["xth"][ir]                
                self.delta_mantlefac_xy_np = ( 
                        self.scenario_delta_maps_dict["mantlefac"][ir])
                self.delta_defage_xy_np = (
                        self.scenario_delta_maps_dict["def_age"][ir])
                self.delta_hfred_xy_np = (
                        self.scenario_delta_maps_dict["hf_red_fac"][ir])           
                self.delta_pdecay_xy_np = (
                        self.scenario_delta_maps_dict["por_decay_depth"][ir])
                self.delta_spor_xy_np = (
                        self.scenario_delta_maps_dict["surf_por"][ir])
                self.delta_toc_xy_np = self.scenario_delta_maps_dict["TOC"][ir]
                self.delta_HI_xy_np = self.scenario_delta_maps_dict["HI"][ir]
                self.delta_thick_xy_np = (
                        self.scenario_delta_maps_dict["thick"][ir])
                self.delta_oilapi_xy_np = (
                        self.scenario_delta_maps_dict["oilapi"][ir])
                self.delta_gasgrav_xy_np = (
                        self.scenario_delta_maps_dict["gasgrav"][ir])               
                self.delta_porth_xy_np = (
                        self.scenario_delta_maps_dict["porth"][ir])
                self.delta_Telastic = (
                        self.scenario_delta_dict["temp_elastic"][ir])            
                self.delta_adsth = self.scenario_delta_dict["adsth"][ir]
                self.delta_inert = self.scenario_delta_dict["inert"][ir]
                # -1 = early, 0 = normal, 1 = late, 2 = base case model                                
                self.idelta_kinetics = self.scenario_delta_dict["kinetics"][ir]
                # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_gas_frac = self.scenario_delta_dict["gas_frac"][ir]
                 # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_polar_frac = (
                        self.scenario_delta_dict["polar_frac"][ir])                             
                self.src_index_list_bulk_gor = (
                        self.scenario_delta_dict["src_index"][ir])                
                self.delta_sat_gor = 0.0                
            if self.multirun_stype == "monte_carlo":                
                self.delta_bghf_xy_np = self.bghf_delta_nxy_np[ir]
                self.delta_xth_xy_np = self.xth_delta_nxy_np[ir]
                self.delta_mantlefac_xy_np = self.mantlefac_delta_nxy_np[ir]
                self.delta_defage_xy_np = self.defage_delta_nxy_np[ir]
                self.delta_hfred_xy_np = self.hfred_delta_nxy_np[ir]
                self.delta_pdecay_xy_np = self.pdecay_delta_nxy_np[ir]
                self.delta_spor_xy_np = self.spor_delta_nxy_np[ir]
                self.delta_toc_xy_np = self.toc_delta_nxy_np[ir]
                self.delta_HI_xy_np = self.hi_delta_nxy_np[ir]
                self.delta_thick_xy_np = self.srcthick_delta_nxy_np[ir]
                self.delta_oilapi_xy_np = self.oilapi_delta_nxy_np[ir]
                self.delta_gasgrav_xy_np = self.gasgrav_delta_nxy_np[ir]
                self.delta_porth_xy_np = self.porth_delta_nxy_np[ir]
                # Telastic_delta_nxy_np[ir] # Not a map 
                self.delta_Telastic = self.delta_vec_Telastic[ir]               
                self.delta_adsth = self.delta_vec_adsth[ir] # Not a map
                self.delta_inert = self.delta_vec_inert[ir] # Not a map
                # -1 = early, 0 = normal, 1 = late, 2 = base case model                             
                self.idelta_kinetics = self.discrete_dict["kinetics"][3][ir]
                # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_gas_frac = self.discrete_dict["gas_frac"][3][ir]
                # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_polar_frac = (
                        self.discrete_dict["polar_frac"][3][ir])               
                self.idelta_src_scenario = (
                        self.discrete_dict["src_index"][3][ir])             
                if self.idelta_src_scenario == -1:
                    self.src_index_list_bulk_gor = ( 
                            self.src_index_list_bulk_gor_scenario1[:])
                elif self.idelta_src_scenario == 0:
                    self.src_index_list_bulk_gor = (
                            self.src_index_list_bulk_gor_scenario2[:])
                elif self.idelta_src_scenario == 1:
                    self.src_index_list_bulk_gor = (
                            self.src_index_list_bulk_gor_scenario3[:])                
                self.delta_sat_gor = 0.0
                # layer depth uncertainty                
            elif self.multirun_stype == "highlow": 
                self.delta_bghf_val = self.HL_delta_dict["bghf"][ir][1]
                self.delta_xth_val = self.HL_delta_dict["xth"][ir][1]
                self.delta_mantlefac_val = (
                        self.HL_delta_dict["mantlefac"][ir][1])
                self.delta_defage_val = self.HL_delta_dict["def_age"][ir][1]
                self.delta_hfred_val = self.HL_delta_dict["hf_red_fac"][ir][1]
                self.delta_pdecay_val = (
                        self.HL_delta_dict["por_decay_depth"][ir][1])
                self.delta_spor_val = self.HL_delta_dict["surf_por"][ir][1]               
                self.delta_toc_val = self.HL_delta_dict["TOC"][ir][1]
                self.delta_HI_val = self.HL_delta_dict["HI"][ir][1]
                self.delta_thick_val = self.HL_delta_dict["thick"][ir][1]
                self.delta_oilapi_val = self.HL_delta_dict["oilapi"][ir][1]
                self.delta_gasgrav_val = self.HL_delta_dict["gasgrav"][ir][1]
                self.delta_porth_val = self.HL_delta_dict["porth"][ir][1]
                # Not a map                
                self.delta_Telastic_val = (
                        self.HL_delta_dict["temp_elastic"][ir][1])
                # Not a map                            
                self.delta_adsth_val = self.HL_delta_dict["adsth"][ir][1]
                # Not a map
                self.delta_inert_val = self.HL_delta_dict["inert"][ir][1]
                # -1 = early, 0 = normal, 1 = late, 2 = base case model                
                self.idelta_kinetics_val = (
                        self.HL_delta_dict["kinetics"][ir][1])
                # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_gas_frac_val = (
                        self.HL_delta_dict["gas_frac"][ir][1])
                # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_polar_frac_val = (
                        self.HL_delta_dict["polar_frac"][ir][1])          
                self.delta_bghf_xy_np = self.HL_delta_dict["bghf"][ir][0]
                self.delta_xth_xy_np = self.HL_delta_dict["xth"][ir][0]
                self.delta_mantlefac_xy_np = (
                        self.HL_delta_dict["mantlefac"][ir][0])
                self.delta_defage_xy_np = self.HL_delta_dict["def_age"][ir][0]
                self.delta_hfred_xy_np = (
                        self.HL_delta_dict["hf_red_fac"][ir][0])
                self.delta_pdecay_xy_np = (
                        self.HL_delta_dict["por_decay_depth"][ir][0])
                self.delta_spor_xy_np = self.HL_delta_dict["surf_por"][ir] [0]               
                self.delta_toc_xy_np = self.HL_delta_dict["TOC"][ir][0]
                self.delta_HI_xy_np = self.HL_delta_dict["HI"][ir][0]
                self.delta_thick_xy_np = self.HL_delta_dict["thick"][ir][0]
                self.delta_oilapi_xy_np = self.HL_delta_dict["oilapi"][ir][0]
                self.delta_gasgrav_xy_np = self.HL_delta_dict["gasgrav"][ir][0]
                self.delta_porth_xy_np = self.HL_delta_dict["porth"][ir][0]
                # Not a map
                self.delta_Telastic = self.HL_delta_dict["temp_elastic"][ir][0]           
                # Not a map
                self.delta_adsth = self.HL_delta_dict["adsth"][ir][0] 
                # Not a map
                self.delta_inert = self.HL_delta_dict["inert"][ir][0]
                # -1 = early, 0 = normal, 1 = late, 2 = base case model                                
                self.idelta_kinetics = self.HL_delta_dict["kinetics"][ir][0]
                # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_gas_frac = self.HL_delta_dict["gas_frac"][ir][0]
                # -1 = minimum, 0 = base, 1 = maximum, 2 = base case model
                self.idelta_polar_frac = (
                        self.HL_delta_dict["polar_frac"][ir][0])               
                self.delta_sat_gor = 0.0
        # Create input for multirun file and write to file
        if self.multirun_stype in ["scenario", "highlow", "monte_carlo"]:
            if self.multirun_stype in ["scenario", "highlow"]:
                tmp_list = [
                                ir, self.delta_bghf_val,
                                self.delta_xth_val, 
                                self.delta_mantlefac_val,
                                self.delta_defage_val, 
                                self.delta_hfred_val, 
                                self.delta_Telastic_val, 
                                self.delta_pdecay_val,
                                self.delta_spor_val, 
                                self.delta_toc_val, 
                                self.delta_HI_val, 
                                self.delta_thick_val,
                                self.delta_oilapi_val, 
                                self.delta_gasgrav_val,
                                self.delta_porth_val, 
                                self.delta_adsth_val, 
                                self.delta_inert_val, 
                                self.idelta_kinetics_val,
                                self.idelta_gas_frac_val, 
                                self.idelta_polar_frac_val, 
                                self.delta_sat_gor
                            ]
            elif self.multirun_stype in ["monte_carlo"]:
                tmp_list = [ir]                
                for key in self.risk_var_names:
                    if key not in ["temp_elastic", 
                                   "adsth", 
                                   "inert", 
                                   "kinetics",
                                   "gas_frac", 
                                   "polar_frac"]:
                        
                        min_elem = self.pdf_dict[key][4]
                        max_elem = self.pdf_dict[key][5]
                        if self.multirun_stype == "monte_carlo":
                            rv = self.pdf_dict[key][6][ir]
                        else:
                            rv = "NA"
                        tmp_list.append(min_elem)
                        tmp_list.append(max_elem)
                        tmp_list.append(rv)               
                tmp_list.append(self.delta_Telastic)
                tmp_list.append(self.delta_adsth)
                tmp_list.append(self.delta_inert)
                tmp_list.append(self.idelta_kinetics)
                tmp_list.append(self.idelta_gas_frac)
                tmp_list.append(self.idelta_polar_frac)
                tmp_list.append(self.idelta_src_scenario)                       
            strout = self.output_dir_name + "_" + self.rname
            for str_tmp in tmp_list:
                strout = strout + "," + str(str_tmp)
            for srcindex in self.src_index_list_bulk_gor:
                strout = strout + "," + str(srcindex)
            fout_multirun = open(self.multirun_file_name, 'a')                                                        
            fout_multirun.write(strout + " \n")
            fout_multirun.close()


def initiate_risk_runs(
                        script_path, input_path, output_path, multirun_stype,
                        nruns_monte_carlo, ioutput_type, iuse_max_state,
                        src_index_list_bulk_gor, itype3D
):
    print("**Starting Program**")
    print("script_path: " + script_path)
    print("input_path: " + input_path)
    print("output_path: " + output_path)
    print("multirun_stype: " + multirun_stype)
    print("nruns_monte_carlo: " + str(nruns_monte_carlo))
    print("ioutput_type: " + str(ioutput_type))
    if os.path.isdir(output_path) == False:
        print("Creating output directory....")
        os.mkdir(output_path)
    root_path, dir_name = os.path.split(output_path)
    root_path, output_dir_name = os.path.split(root_path)
    print_funcs.print_info(output_dir_name, [output_dir_name])
    output_path_root = output_path
    print_funcs.print_info(output_path_root, [output_path_root])
    print("Looking for directory ", input_path)
    ifind_input = 1
    if os.path.isdir(input_path) == True:
        print_funcs.print_statement("Found input directory")
    else:
        print_funcs.print_warning(
                            "Input directory not found. Skipping execution")
        ifind_input = 0
    if ifind_input == 1:
        if ioutput_type == 1:
            print_funcs.print_statement(
                            "All output will be sent to log.txt starting now")
            original = sys.stdout
            std_output_file = os.path.join(output_path, "log.txt")         
            sys.stdout = open(std_output_file, 'w')
        # 0 = only perform calcs at single node; 1 = all nodes
        #itype3D = 1
        if itype3D == 1:
            inode = 1 # i index of node used for itype3D = 0
            jnode = 1 # i index of node used for itype3D = 0
            input1D_dict = {}
        else:
            inode = 1 # i index of node used for itype3D = 0
            jnode = 1 # i index of node used for itype3D = 0
            path1D = os.path.join(script_path, "ipa_1D")
            model1D_direc = path1D
            model1D_name = "IPA1Dinput.csv"
            csv1D_path = os.path.join(model1D_direc, model1D_name)
            (
                input1D_dict, lith1D_dict, 
                rockID_dict, lith1D_final_dict
            ) = ipa1D.ipa1Dmain(path1D, input_path, output_path, csv1D_path)
        execute_risk_runs(
                        iuse_max_state, script_path, input_path, output_path, 
                        output_path_root, itype3D, inode, jnode,
                        output_dir_name, ioutput_type, input1D_dict,
                        nruns_monte_carlo, multirun_stype,
                        src_index_list_bulk_gor
                    )
    if ioutput_type == 1:
        sys.stdout.close()
        sys.stdout = original
    print("**End of Program**")


def execute_risk_runs(
                        iuse_max_state, script_path, 
                        input_path, output_path, 
                        output_path_root, itype3D, inode, jnode,
                        output_dir_name, ioutput_type, input1D_dict,
                        nruns_monte_carlo, multirun_stype,
                        src_index_list_bulk_gor
):
    riskmodel = RiskModel(
                        input_path, output_path, 
                        output_path_root, output_dir_name, 
                        ioutput_type, nruns_monte_carlo, 
                        multirun_stype, src_index_list_bulk_gor
                        )
    riskmodel.initialize_risk_runs()
    for ir in range(riskmodel.nruns):    
        riskmodel.define_inputs_for_risk_run(ir)
        output_path = riskmodel.output_path
        fileIO.make_riskrun_csv(
            output_path, ir, riskmodel.rname, riskmodel.nruns, 
            riskmodel.src_index_list_bulk_gor
            )
        tt1 = time.time()
        IPA3D.IPA3D_MAIN(
                            iuse_max_state, script_path, 
                            input_path, output_path, 
                            itype3D, inode, jnode, input1D_dict,
                            riskmodel
                        )
        tt2 = time.time()        
        print("Total time for IPA3D_Main cpu(s) : ", tt2-tt1)