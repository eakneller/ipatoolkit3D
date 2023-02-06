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
import time
import source_kinetics
import print_funcs
            
def hc_generation(srckinetics, model, ioutput_main, process):         
    OILAPI_list = []
    GASGRAV_list = []
    imass_gen = srckinetics.imass_gen
    iuse_high_lom_gas = srckinetics.iuse_high_lom_gas
    if imass_gen > 0:
        tt1 = time.time()
        # Primary generation
        imass_calc_type = 0
        ioutput = 1
        (
            OILAPI_list, 
            GASGRAV_list
        ) = source_kinetics.calculate_TR_and_mass_history(
                                                    srckinetics, model, 
                                                    imass_calc_type, ioutput)
        tt2 = time.time()
        print_funcs.print_finfo(ioutput_main, process, 
                                "Calculated TR and primary mass gen", tt2-tt1)        
        tt1 = time.time()
        # Secondary cracking
        imass_calc_type = 1
        ioutput = 1
        source_kinetics.calculate_TR_and_mass_history(
                                                    srckinetics, model, 
                                                    imass_calc_type, ioutput)             
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Calculated secondary cracking", tt2-tt1)            
        if iuse_high_lom_gas == 1:
            tt1 = time.time()
            # High LOM Gas
            imass_calc_type = 2
            ioutput = 1
            source_kinetics.calculate_TR_and_mass_history(
                                                    srckinetics, model, 
                                                    imass_calc_type, ioutput)   
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Calculated high LOM gas", tt2-tt1)
        tt1 = time.time()
        imass_calc_type = 0
        ioutput = 1
        source_kinetics.calculate_expulsion_history(imass_calc_type, model)   
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Calculated primary expulsion rate", tt2-tt1)
        tt1 = time.time()
        imass_calc_type = 1
        ioutput = 1
        source_kinetics.calculate_expulsion_history(imass_calc_type, model)
        tt2 = time.time()
        print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Calculated secondary  expulsion rate", tt2-tt1)
    return OILAPI_list, GASGRAV_list

