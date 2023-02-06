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
import numpy as np
import map_tools
import fileIO
import fluid_props


def trap_masses(
                iclass, Oil_API, sgrav_gas, 
                Vhcp_m3, Vhcp_single_phase_gas_m3,
                Vhcp_single_phase_oil_m3, 
                Vhcp_dual_phase_gas_m3, Vhcp_dual_phase_oil_m3,
                Moil_charge_g, Mdisgas_charge_g, 
                Mfreegas_charge_g, Tres_C, fpress_psi,
                WD_m, TC_surf, res_depth_smm, 
                rho_sea, rho_brine, oil_den_mod_fac,
                gas_den_mod_fac, delta_rho_gas_kg_m3, 
                ilimit_gor, iuse_sGOR_cor,
                sGOR_at_charp, charp, exp_fac,
                a_gas, b_gas, sGOR_scf_bbl_max
):
    #OGR_gas = 0.15 # g/g    
    Tres_F = 9.0/5.0*Tres_C + 32.0  
    cm3_scf = 28316.85
    cm3_bbl = 158987.30
    rho_air_surface = 0.001292 # g/cm3    
    rho_gas_surface = rho_air_surface*sgrav_gas # g/cm3
    sgrav_oil = 141.5/(131.5+Oil_API)
    rho_oil_surface = sgrav_oil # g/cm3
    sGOR_g_g_max = (sGOR_scf_bbl_max/rho_oil_surface
                        *rho_gas_surface/cm3_bbl*cm3_scf)
    OGR_min_g_g = 1/sGOR_g_g_max
    #Mliq_charge_g = Moil_charge_g + Mdisgas_charge_g
    if Moil_charge_g > 0:
        GOR_liq = Mdisgas_charge_g / Moil_charge_g # g/g
    else:
        if Mdisgas_charge_g > 0:
            GOR_liq = sGOR_g_g_max
        else:
            GOR_liq = 0            
    GOR_liq_m3m3 = GOR_liq*rho_oil_surface/rho_gas_surface
    GOR_liq_scf_bbl = GOR_liq_m3m3*cm3_bbl/cm3_scf    
    Mtotgas_charge_g = Mfreegas_charge_g +  Mdisgas_charge_g    
    if Moil_charge_g > 0:        
        bGOR_charge = Mtotgas_charge_g/Moil_charge_g
    else:
        bGOR_charge = sGOR_g_g_max        
    bGOR_charge_m3m3 = bGOR_charge*rho_oil_surface/rho_gas_surface
    bGOR_charge_scf_bbl = bGOR_charge_m3m3*cm3_bbl/cm3_scf   
    if Mtotgas_charge_g > 0:
        bOGR_charge = Moil_charge_g/Mtotgas_charge_g
    else:
        bOGR_charge = OGR_min_g_g        
    #bOGR_charge_m3m3 = bOGR_charge*rho_gas_surface/rho_oil_surface
    #bOGR_charge_scf_bbl = bOGR_charge_m3m3*cm3_scf/cm3_bbl   
    #**************************************************************************
    # Estimate free gas and liquid density using reservoir conditions
    # assuming saturation. Also, estimate saturation GOR for oil and gas
    # in the reservoir.
    #**************************************************************************    
    iuse_sat_gor = 1
    (
        zfactor, sGORoil_scf_bbl_res, 
        sGORoil_g_g_res, sGORgas_scf_bbl_res,
        sGORgas_g_g_res, GFVF_rbbl_scf, 
        GFVF_rcf_scf, rho_rgas_g_cc, LFVF_rbbl_stb,
        rho_rfluid_g_cc, dum1, dum2
    ) = fluid_props.calc_fluid_props(
                    Tres_F, fpress_psi, 
                    sgrav_gas, Oil_API, 0,
                    sGOR_scf_bbl_max, iuse_sat_gor, 
                    iuse_sGOR_cor, sGOR_at_charp,
                    charp, exp_fac, a_gas, b_gas
                    )   
    (
        sGOR_int_scf_bbl, fpress_int
    ) = fluid_props.calc_intersection_trap_charge(
            Tres_C, TC_surf, WD_m,
            res_depth_smm, rho_sea, rho_brine, 
            Oil_API, sgrav_gas, rho_gas_surface,
            rho_oil_surface, sGOR_at_charp, 
            charp, exp_fac, sGOR_scf_bbl_max,
            cm3_scf, cm3_bbl, a_gas, b_gas, iuse_sGOR_cor
            )    
    sGOR_int_g_g = (
                    sGOR_int_scf_bbl*
                        cm3_scf/cm3_bbl*(rho_gas_surface/rho_oil_surface)
                   )    
    rho_rgas_kg_m3 = rho_rgas_g_cc*1000
    rho_rfluid_kg_m3 = rho_rfluid_g_cc*1000
    if ilimit_gor == 1:    
        if sGORoil_scf_bbl_res >= sGOR_int_scf_bbl:
            sGORoil_scf_bbl_res = sGOR_int_scf_bbl
            sGORoil_g_g_res = sGOR_int_g_g    
        if sGORgas_scf_bbl_res < sGOR_int_scf_bbl:
            sGORgas_scf_bbl_res = sGOR_int_scf_bbl
            sGORgas_g_g_res = sGOR_int_g_g
    #**************************************************************************
    # Estimate phase state and OGR_g_g using bGOR_charge
    #**************************************************************************
    if bGOR_charge <= 0:        
        #iphase = 0.0 # single phase oil
        sphase = "Single Phase Oil"
        satstate = "Undersaturated"        
        mFGg_charge = 0
        mDOg_charge = 0
        mDGg_charge = Mtotgas_charge_g
        mLg_charge = Moil_charge_g + Mtotgas_charge_g
        OGR_gas_calc = bOGR_charge
    else:        
        if sGORoil_g_g_res >= sGORgas_g_g_res: # single phase below the v            
            #iphase = 1.0
            # Just use the oil branch and the oil correlations for properties
            mFGg_charge = 0
            mDOg_charge = 0
            mDGg_charge = Mtotgas_charge_g
            mLg_charge = Moil_charge_g + Mtotgas_charge_g
            if bGOR_charge <= sGOR_int_g_g:
                sphase = "Deep Single Phase Oil"
                OGR_gas_calc = bOGR_charge
            else:
                sphase = "Deep Single Phase Gas"
                OGR_gas_calc = bOGR_charge
            satstate = "Undersaturated"
        elif bGOR_charge <= sGORoil_g_g_res: # single phase oil               
            #iphase = 0.0
            sphase = "Single Phase Oil"              
            mFGg_charge = 0
            mDOg_charge = 0
            mDGg_charge = Mtotgas_charge_g
            mLg_charge = Moil_charge_g + Mtotgas_charge_g
            OGR_gas_calc = bOGR_charge
            satstate = "Undersaturated"
        elif bGOR_charge >= sGORgas_g_g_res: # single phase gas            
            #iphase = 3.0
            sphase = "Single Phase Gas"
            mFGg_charge = Mtotgas_charge_g
            mDOg_charge = Moil_charge_g
            mDGg_charge = 0
            mLg_charge = 0
            OGR_gas_calc = bOGR_charge
            satstate = "Undersaturated"
        else: # dual phase            
            #iphase = 2.0
            sphase = "Dual Phase"
            (   mDOg_charge, mDGg_charge, 
                mLg_charge, mFGg_charge
            ) = fluid_props.calc_liquid_and_free_gas_mass(
                                            3, Moil_charge_g, Mtotgas_charge_g,
                                            sGORoil_g_g_res, sGORgas_g_g_res
                                            )
            OGR_gas_calc = 1/sGORgas_g_g_res
            satstate = "Saturated"
    # Addding correction factors
    rho_gas_change_kg_m3 = delta_rho_gas_kg_m3*OGR_gas_calc*sGOR_int_g_g
    rho_free_gas_trap_kg_m3 = (
                                rho_rgas_kg_m3*(1+gas_den_mod_fac) 
                                + rho_gas_change_kg_m3
                                )
    rho_satliq_trap_kg_m3 = rho_rfluid_kg_m3 + rho_rfluid_kg_m3*oil_den_mod_fac
    #**************************************************************************
    # Estimating free gas and liquid density using reservoir conditions
    # and charge liquid GOR assuming undersaturation
    #**************************************************************************    
    iuse_sat_gor = 0    
    (
        zfactor, sGOR_scf_bbl, sGOR_g_g, 
        sGORgas_scf_bbl, sGORgas_g_g,
        GFVF_rbbl_scf, GFVF_rcf_scf, 
        rho_rgas_g_cc, LFVF_rbbl_stb_charge,
        rho_rfluid_g_cc_charge, 
        sGOR_int_scf_bbl, sGOR_int_g_g
    ) = fluid_props.calc_fluid_props(
                Tres_F, fpress_psi, 
                sgrav_gas, Oil_API,
                GOR_liq_scf_bbl, sGOR_scf_bbl_max, 
                iuse_sat_gor, iuse_sGOR_cor,
                sGOR_at_charp, charp, exp_fac,
                a_gas, b_gas
                )
    if ilimit_gor == 1:
        if sGOR_scf_bbl >= sGOR_int_scf_bbl:
            sGOR_scf_bbl = sGOR_int_scf_bbl
            #sGOR_g_g = sGOR_int_g_g
        if sGORgas_scf_bbl_res < sGOR_int_scf_bbl:
            sGORgas_scf_bbl_res = sGOR_int_scf_bbl
            sGORgas_g_g_res = sGOR_int_g_g      
    rho_rfluid_kg_m3_charge = rho_rfluid_g_cc_charge*1000
    # Approximate density of unsaturated liquid
    rho_unsatliq_trap_kg_m3 = (
                                rho_rfluid_kg_m3_charge +
                                rho_rfluid_kg_m3_charge*oil_den_mod_fac
                                )
    #**************************************************************************
    # Calculate End-member fill-to-spill scenarios
    #**************************************************************************
    if iclass == 1:
        Munsatliq_trap_max_g = Vhcp_m3*rho_unsatliq_trap_kg_m3*1000
#        if GOR_liq > 0:
#            Munsatdisgas_trap_max_g = Munsatliq_trap_max_g/(1+1/GOR_liq)
#        else:
#            Munsatdisgas_trap_max_g = 0
#        Msatliq_trap_max_g = Vhcp_m3*rho_satliq_trap_kg_m3*1000
#        if sGORoil_g_g_res > 0:
#            Msatdisgas_trap_max_g = Msatliq_trap_max_g/(1+1/sGORoil_g_g_res)
#        else:
#            Msatdisgas_trap_max_g = 0
        Mfluid_gas_trap_max_g =  Vhcp_m3*rho_free_gas_trap_kg_m3*1000
    else:
        Munsatliq_trap_max_g = (
                        Vhcp_single_phase_oil_m3*rho_unsatliq_trap_kg_m3*1000)
#        if GOR_liq > 0:
#            Munsatdisgas_trap_max_g = Munsatliq_trap_max_g/(1+1/GOR_liq)
#        else:
#            Munsatdisgas_trap_max_g = 0
#        Msatliq_trap_max_g = (
#                       Vhcp_single_phase_oil_m3*rho_satliq_trap_kg_m3*1000)
#        if sGORoil_g_g_res > 0:
#            Msatdisgas_trap_max_g = Msatliq_trap_max_g/(1+1/sGORoil_g_g_res)
#        else:
#            Msatdisgas_trap_max_g = 0
        Mfluid_gas_trap_max_g =  (
                        Vhcp_single_phase_gas_m3*rho_free_gas_trap_kg_m3*1000)
    Msatliq_tot_g = 0.0
    Munsatliq_tot_g = 0.0      
    Mfluid_gas_tot_g = 0.0          
    Mfluid_gas_trap_g = 0.0        
    Mdisoil_trap_g = 0.0            
    Munsatliq_trap_g = 0.0
    Msatliq_trap_g = 0.0        
    Mdisgas_trap_g = 0.0        
    Mfree_oil_trap_g = 0.0        
    Mfree_gas_trap_g = 0.0
    Vsatlid_trap_m3 = 0.0
    Vforliq_m3 = 0.0
    Vfluid_gas_trap_m3 = 0.0
    Moil_min_g = 0.0
    Moil_tot_g = 0.0
    if sphase == "Single Phase Oil" or sphase == "Deep Single Phase Oil":
        Munsatliq_tot_g = Moil_charge_g + Mtotgas_charge_g
        Munsatliq_trap_g = min(Munsatliq_tot_g, Munsatliq_trap_max_g)
        if bGOR_charge > 0:
            Mdisgas_trap_g = Munsatliq_trap_g/(1+1/bGOR_charge)
        else:
            Mdisgas_trap_g = 0
        Mfree_oil_trap_g = Munsatliq_trap_g - Mdisgas_trap_g
        Vhc_final_m3 = Munsatliq_trap_g/(rho_unsatliq_trap_kg_m3*1000)        
        perc_fill = Vhc_final_m3/Vhcp_m3
        if perc_fill >= 1.0:
            sfill = "Filled-to-Spill"
        else:
            sfill = "Underfilled"
        liq_fill_frac = perc_fill
        fluid_gas_fill_frac = 0
        if rho_unsatliq_trap_kg_m3 > 0 and Vhcp_m3 > 0:
            trap_mul =  (Munsatliq_tot_g)/1000/rho_unsatliq_trap_kg_m3/Vhcp_m3
        else:
            trap_mul = 0
        Moil_trap_g = Mfree_oil_trap_g
        Mgas_trap_g = Mdisgas_trap_g        
        if Moil_trap_g > 0 and Mgas_trap_g > 0:
            bGOR_trap_g_g = Mgas_trap_g/Moil_trap_g
        else:
            if Mgas_trap_g > 0:
                bGOR_trap_g_g = sGOR_g_g_max
            else:
                bGOR_trap_g_g = 0.0        
        GOR_res_oil_g_g = bGOR_trap_g_g
        GOR_res_gas_g_g = 0        
    elif (sphase == "Single Phase Gas" 
          or sphase == "Deep Single Phase Gas"
          ): # Under-saturated gas        
        Moil_min_g = Mtotgas_charge_g*OGR_min_g_g        
        if Moil_charge_g == 0:
            Moil_tot_g = Moil_min_g
        else:
            Moil_tot_g = Moil_charge_g        
        Mfluid_gas_tot_g = Mtotgas_charge_g + Moil_tot_g        
        Mfluid_gas_trap_g = min(Mfluid_gas_tot_g, Mfluid_gas_trap_max_g)       
        if OGR_gas_calc > 0:
            Mdisoil_trap_g = Mfluid_gas_trap_g/(1+1/OGR_gas_calc)
        else:
            Mdisoil_trap_g = 0.0        
        Mfree_gas_trap_g = Mfluid_gas_trap_g - Mdisoil_trap_g        
        Vhc_final_m3 = Mfluid_gas_trap_g/(rho_free_gas_trap_kg_m3*1000)
        perc_fill = Vhc_final_m3/Vhcp_m3
        if perc_fill >= 1.0:
            sfill = "Filled-to-Spill"
        else:
            sfill = "Underfilled"
        liq_fill_frac = 0
        fluid_gas_fill_frac = perc_fill
        if rho_free_gas_trap_kg_m3 > 0 and Vhcp_m3 > 0:
            trap_mul =  (Mfluid_gas_tot_g)/1000/rho_free_gas_trap_kg_m3/Vhcp_m3
        else:
            trap_mul = 0
        Moil_trap_g = Mdisoil_trap_g
        Mgas_trap_g = Mfree_gas_trap_g        
        if Moil_trap_g > 0 and Mgas_trap_g > 0:
            bGOR_trap_g_g = Mgas_trap_g/Moil_trap_g
        else:
            if Mgas_trap_g > 0:
                bGOR_trap_g_g = sGOR_g_g_max
            else:
                bGOR_trap_g_g = 0.0
        GOR_res_oil_g_g = 0
        GOR_res_gas_g_g = bGOR_trap_g_g        
    elif sphase == "Dual Phase": # Saturated / Dual Phase
        Msatliq_tot_g = max(Moil_charge_g-mDOg_charge+mDGg_charge,0)
        Mfluid_gas_tot_g = max(Mtotgas_charge_g-mDGg_charge+mDOg_charge,0)
        if iclass == 1: # Class 1
            Vfluid_gas_trap_m3 = min(
                    (Mfluid_gas_tot_g/1000)/rho_free_gas_trap_kg_m3,Vhcp_m3)
            Mfluid_gas_trap_g = Vfluid_gas_trap_m3*rho_free_gas_trap_kg_m3*1000
            if OGR_gas_calc > 0:
                Mdisoil_trap_g = Mfluid_gas_trap_g/(1+1/OGR_gas_calc)
            else:
                Mdisoil_trap_g = 0
            Mfree_gas_trap_g = Mfluid_gas_trap_g - Mdisoil_trap_g
            Vforliq_m3 = Vhcp_m3 - Vfluid_gas_trap_m3
            Msatliq_trap_g = min(
                        Vforliq_m3*rho_satliq_trap_kg_m3*1000, Msatliq_tot_g)
        else: # Class 2/3
            Vsatlid_trap_m3 = min(
              Msatliq_tot_g/1000/rho_satliq_trap_kg_m3, Vhcp_dual_phase_oil_m3)
            Vfluid_gas_trap_m3 = min(
                    Mfluid_gas_tot_g/1000/rho_free_gas_trap_kg_m3,
                    Vhcp_single_phase_gas_m3 
                    - (Vhcp_single_phase_gas_m3-Vhcp_dual_phase_gas_m3)
                                        *Vsatlid_trap_m3/Vhcp_dual_phase_oil_m3
                    )
            Mfluid_gas_trap_g = Vfluid_gas_trap_m3*rho_free_gas_trap_kg_m3*1000
            if OGR_gas_calc > 0:
                Mdisoil_trap_g = Mfluid_gas_trap_g/(1+1/OGR_gas_calc)
            else:
                Mdisoil_trap_g = 0.0
            Mfree_gas_trap_g = Mfluid_gas_trap_g - Mdisoil_trap_g
            Msatliq_trap_g = min(
                                    Vsatlid_trap_m3*rho_satliq_trap_kg_m3*1000,
                                    Msatliq_tot_g
                                    )
        if sGORoil_g_g_res > 0:
            Mdisgas_trap_g = Msatliq_trap_g/(1+1/sGORoil_g_g_res)
        else:
            Mdisgas_trap_g = 0
        Mfree_oil_trap_g = Msatliq_trap_g - Mdisgas_trap_g
        Vliq_m3 = Msatliq_trap_g/1000/rho_satliq_trap_kg_m3
        Vfluid_gas_m3 = Mfluid_gas_trap_g/1000/rho_free_gas_trap_kg_m3
        Vhc_final_m3 = (Vliq_m3 + Vfluid_gas_m3)
        perc_fill = Vhc_final_m3/Vhcp_m3
        if perc_fill >= 1.0:
            sfill = "Filled-to-Spill"
        else:
            sfill = "Underfilled"
        liq_fill_frac = Vliq_m3/Vhcp_m3
        fluid_gas_fill_frac = Vfluid_gas_m3/Vhcp_m3
        if (rho_free_gas_trap_kg_m3 > 0 and rho_satliq_trap_kg_m3 > 0 
            and Vhcp_m3 > 0
            ):
            trap_mul =  (
                        (Mfluid_gas_tot_g)/1000/rho_free_gas_trap_kg_m3/Vhcp_m3
                        + (Msatliq_tot_g)/1000/rho_satliq_trap_kg_m3/Vhcp_m3
                        )
        else:
            trap_mul = 0
        Moil_trap_g = Mfree_oil_trap_g + Mdisoil_trap_g
        Mgas_trap_g = Mfree_gas_trap_g + Mdisgas_trap_g
        if Moil_trap_g > 0 and Mgas_trap_g > 0:
            bGOR_trap_g_g = Mgas_trap_g/Moil_trap_g
        else:
            if Mgas_trap_g > 0:
                bGOR_trap_g_g = sGOR_g_g_max
            else:
                bGOR_trap_g_g = 0.0
        if liq_fill_frac > 0:
            GOR_res_oil_g_g = sGORoil_g_g_res
        else:
            GOR_res_oil_g_g = 0
        
        if fluid_gas_fill_frac > 0:
            GOR_res_gas_g_g = sGORgas_g_g_res
        else:
            GOR_res_gas_g_g = 0   
    bGOR_trap_m3m3 = bGOR_trap_g_g*rho_oil_surface/rho_gas_surface
    bGOR_trap_scf_bbl = bGOR_trap_m3m3*cm3_bbl/cm3_scf
    GOR_res_oil_m3m3 = GOR_res_oil_g_g*rho_oil_surface/rho_gas_surface
    GOR_res_oil_scf_bbl = GOR_res_oil_m3m3*cm3_bbl/cm3_scf
    GOR_res_gas_m3m3 = GOR_res_gas_g_g*rho_oil_surface/rho_gas_surface
    GOR_res_gas_scf_bbl = GOR_res_gas_m3m3*cm3_bbl/cm3_scf
    Vfree_oil_trap_GOB = Mfree_oil_trap_g/(rho_oil_surface*1e6)*6.28981/1e9
    Vfree_fluid_gas_Tcf = Mfree_gas_trap_g/(rho_gas_surface*1e6)*35.3147/1e12 
    return (
            sphase, satstate, 
            bGOR_trap_scf_bbl, GOR_res_oil_scf_bbl,
            GOR_res_gas_scf_bbl, Vhc_final_m3, 
            perc_fill, liq_fill_frac,
            fluid_gas_fill_frac, Vfree_oil_trap_GOB, 
            Vfree_fluid_gas_Tcf,
            trap_mul, sfill, Msatliq_tot_g, 
            Munsatliq_tot_g, Mfluid_gas_tot_g,
            Mfluid_gas_trap_g, Mdisoil_trap_g, 
            Msatliq_trap_g, Munsatliq_trap_g,
            Mdisgas_trap_g, Mfree_oil_trap_g, 
            Mfree_gas_trap_g, Moil_trap_g,
            Mgas_trap_g, rho_free_gas_trap_kg_m3, 
            rho_satliq_trap_kg_m3,
            rho_unsatliq_trap_kg_m3, bGOR_charge_scf_bbl
            )
    
    
def update_trap_masses(
                        Moil_trap_g_ini, Mgas_trap_g_ini, wb_dir, wb_name,
                        res_dir, res_name,
                        temp_dir, temp_name, temp_wb_name, 
                        bGOR_dir, bGOR_name, sgrav_gas,
                        secgas_name, pgas_name, poil_name, 
                        api_name, free_gas_fraction_theshold,
                        single_phase_risk_range, sGOR_at_charp, 
                        charp, exp_fac, a_gas, b_gas,
                        rho_sea, rho_brine, name_elem, 
                        sGORflag, x, y, Vhcp_m3,
                        Vhcp_single_phase_gas_m3, Vhcp_single_phase_oil_m3,
                        Vhcp_dual_phase_gas_m3, Vhcp_dual_phase_oil_m3, 
                        OWC_area_km2, over_press_psi, iclass, 
                        oil_den_mod_fac, gas_den_mod_fac, 
                        delta_rho_gas_kg_m3, sGOR_scf_bbl_max, 
                        mig_eff, idepth_type
):
    direc = "incremental"
    if idepth_type == 0:
        direc_depth = "Initial"
    else:
        direc_depth = "Updated"
    TC_surf = 5    
    ilimit_gor = 1
    iuse_sGOR_cor = 1    
    AOI_np_ini = np.zeros((1,1))
    naflag = -99999.0    
    # read api map and interpolate
    api_xy, nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np = \
    map_tools.read_ZMAP(
                        os.path.join(bGOR_dir, "MIGPOLY", "cumulative"), 
                        api_name, AOI_np_ini)
    api_xy_np = np.asarray(api_xy)    
    Oil_API = map_tools.zmap_interp(
                                    x, y, naflag, api_xy_np, nx, ny, dx, dy,
                                    xmin, xmax, ymin, ymax)
    if Oil_API <= 0:
        Oil_API = 38.0                    
    # read top reservoir map and interpolate   
    (
        res_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(
                            os.path.join(res_dir, direc_depth), 
                            res_name, AOI_np_ini)
    dres = map_tools.zmap_interp(x, y, naflag, res_xy_np, nx, ny, dx, dy,
                                 xmin, xmax, ymin, ymax)    
    # read temp reservoir map and interpolate       
    (
        temp_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(temp_dir, temp_name, AOI_np_ini)
    Tres_C = map_tools.zmap_interp(x, y, naflag, temp_xy_np, nx, ny, dx, dy,
                                   xmin, xmax, ymin, ymax)
    # read temp water bottom map and interpolate       
    (
        temp_wb_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(temp_dir, temp_wb_name, AOI_np_ini)
    TC_surf = map_tools.zmap_interp(x, y, naflag, temp_wb_xy_np, 
                                    nx, ny, dx, dy,
                                    xmin, xmax, ymin, ymax)
    # read primary gas Tg and interpolate    
    (
        pgasTg_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(
                            os.path.join(bGOR_dir, "MIGPOLY", direc), 
                            pgas_name, AOI_np_ini)
    mpgas_Tg = map_tools.zmap_interp(x, y, naflag, pgasTg_xy_np, 
                                     nx, ny, dx, dy,
                                     xmin, xmax, ymin, ymax)
    mpgas_g=mpgas_Tg*1e12*mig_eff
    if mpgas_g < 0: 
        mpgas_g = 0
    # read secondary gas Tg and interpolate    
    (
     secgasTg_xy_np, nx, ny, dx, dy, 
     xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(
                            os.path.join(bGOR_dir, "MIGPOLY", direc), 
                            secgas_name, AOI_np_ini)    
    msecgas_Tg = map_tools.zmap_interp(x, y, naflag, secgasTg_xy_np, 
                                       nx, ny, dx, dy,
                                       xmin, xmax, ymin, ymax)
    msecgas_g = msecgas_Tg*1e12*mig_eff
    if msecgas_g < 0: 
        msecgas_g = 0
    # read primary oil Tg and interpolate    
    (
        poilTg_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(
                            os.path.join(bGOR_dir, "MIGPOLY", direc), 
                            poil_name, AOI_np_ini)
    moil_Tg = map_tools.zmap_interp(x, y, naflag, poilTg_xy_np, nx, ny, dx, dy,
                                    xmin, xmax, ymin, ymax)
    moil_g = moil_Tg*1e12*mig_eff
    if moil_g < 0: 
        moil_g = 0
    mgas_g = mpgas_g+msecgas_g
    # read water bottom map and interpolate   
    (
        wb_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(
                            os.path.join(wb_dir, direc_depth), 
                            wb_name, AOI_np_ini)    
    WD_m = map_tools.zmap_interp(x, y, naflag, wb_xy_np, nx, ny, dx, dy,
                                 xmin, xmax, ymin, ymax)
    res_depth_smm = dres - WD_m
    # Need to add trap masses from previous event
    Mdisgas_charge_g = mgas_g + Mgas_trap_g_ini
    Moil_charge_g = moil_g + Moil_trap_g_ini
    Mfreegas_charge_g = 0.0
    # calculate hydrostatic pressure
    # Need to add the over pressure here
    fpress_psi = (
                    (rho_sea*9.8*WD_m + rho_brine*9.8*(dres-WD_m))/6894.76 
                    + over_press_psi
                 )
    
    (
        sphase, satstate, bGOR_trap_scf_bbl, GOR_res_oil_scf_bbl,
        GOR_res_gas_scf_bbl, Vhc_final_m3, 
        perc_fill, liq_fill_frac,
        fluid_gas_fill_frac, Vfree_oil_trap_GOB,
        Vfree_fluid_gas_Tcf, trap_mul,
        sfill, Msatliq_tot_g, Munsatliq_tot_g, Mfluid_gas_tot_g,
        Mfluid_gas_trap_g, Mdisoil_trap_g, 
        Msatliq_trap_g, Munsatliq_trap_g,
        Mdisgas_trap_g, Mfree_oil_trap_g, 
        Mfree_gas_trap_g, Moil_trap_g,
        Mgas_trap_g, rho_free_gas_trap_kg_m3, 
        rho_satliq_trap_kg_m3, rho_unsatliq_trap_kg_m3, 
        bGOR_charge_scf_bbl
    ) = trap_masses(
                    iclass, Oil_API, sgrav_gas, 
                    Vhcp_m3, Vhcp_single_phase_gas_m3,
                    Vhcp_single_phase_oil_m3, 
                    Vhcp_dual_phase_gas_m3, Vhcp_dual_phase_oil_m3,
                    Moil_charge_g, Mdisgas_charge_g, 
                    Mfreegas_charge_g, Tres_C, fpress_psi,
                    WD_m, TC_surf, res_depth_smm, rho_sea, 
                    rho_brine, oil_den_mod_fac,
                    gas_den_mod_fac, delta_rho_gas_kg_m3, 
                    ilimit_gor, iuse_sGOR_cor,
                    sGOR_at_charp, charp, exp_fac,
                    a_gas, b_gas, sGOR_scf_bbl_max
                    )
    
    return (
            sphase, satstate, bGOR_trap_scf_bbl, GOR_res_oil_scf_bbl,
            GOR_res_gas_scf_bbl, Vhc_final_m3, 
            perc_fill, liq_fill_frac,
            fluid_gas_fill_frac, Vfree_oil_trap_GOB, 
            Vfree_fluid_gas_Tcf, trap_mul,
            sfill, Msatliq_tot_g, Munsatliq_tot_g, Mfluid_gas_tot_g,
            Mfluid_gas_trap_g, Mdisoil_trap_g,
            Msatliq_trap_g, Munsatliq_trap_g,
            Mdisgas_trap_g, Mfree_oil_trap_g, 
            Mfree_gas_trap_g, Moil_trap_g,
            Mgas_trap_g, Tres_C, TC_surf, 
            WD_m, res_depth_smm, fpress_psi,
            rho_free_gas_trap_kg_m3, rho_satliq_trap_kg_m3, 
            rho_unsatliq_trap_kg_m3,
            Oil_API, bGOR_charge_scf_bbl
            )
    
    
def calc_trap_masses_main(
                            output_path, sGORflag, res_age_strs, 
                            res_tID, ev_IDs, res_name_top, wb_names, 
                            idepth_type, sGOR_at_charp, charp, exp_fac, a_gas,
                            b_gas, rho_sea, rho_brine, sgrav_gas, 
                            oil_den_mod_fac, gas_den_mod_fac,
                            delta_rho_gas_kg_m3, sGOR_scf_bbl_max, 
                            ntraps, traps_dict, traps_output_dict
):
    """ Calculate charge history and fill state for all traps
    
    This function calculated the charge history and fill-spill state of
    traps using the integrated charge maps for fetch areas produced by the
    Bulk_GOR.calc_bulk_gor function (i.e. MIGPOLY maps), the pressure and
    temperature history of the Trap layer defined by the user, a parameterized 
    black oil saturation model and trap parameters that describe trap-specific
    properties.
    
    """
    output_path_new = os.path.join(output_path, "Trap_Charge_History")
    if os.path.isdir(output_path_new) != True:
        os.mkdir(output_path_new)    
    trap_names = list(traps_dict.keys())   
    # Used to define gas risk when mass of free 
    # gas exceeds this fraction gas factor is 1
    free_gas_fraction_theshold = 0.25    
    # Used to smooth out gas risk in single phase deep domain
    single_phase_risk_range = 0.2    
    nages = len(res_age_strs)
    output_strs_list = [
                        "age", "sphase", "satstate", 
                        "bGOR_trap_scf_bbl", "GOR_res_oil_scf_bbl",
                        "GOR_res_gas_scf_bbl", "bGOR_charge_scf_bbl", 
                        "Vhc_final_m3", "perc_fill", "liq_fill_frac",
                        "fluid_gas_fill_frac", "Vfree_oil_trap_GOB", 
                        "Vfree_fluid_gas_Tcf", 
                        "trap_mul", "sfill", 
                        "Msatliq_tot_g", "Munsatliq_tot_g", "Mfluid_gas_tot_g",
                        "Mfluid_gas_trap_g", "Mdisoil_trap_g",
                        "Msatliq_trap_g", "Munsatliq_trap_g",
                        "Mdisgas_trap_g", "Mfree_oil_trap_g", 
                        "Mfree_gas_trap_g", "Moil_trap_g", "Mgas_trap_g",  
                        "rho_free_gas_trap_kg_m3", "rho_satliq_trap_kg_m3",
                        "rho_unsatliq_trap_kg_m3", "Tres_C", "TC_surf", 
                        "WD_m", "res_depth_smm",
                        "fpress_psi", "Oil_API", "sgrav_gas"
                        ]   
    for trap_name in trap_names:        
        x = traps_dict[trap_name][0]
        y = traps_dict[trap_name][1]
        Vhcp = traps_dict[trap_name][2]
        Vgas_sp = traps_dict[trap_name][3]
        Vliquid_sp = traps_dict[trap_name][4]
        Vgas_dp = traps_dict[trap_name][5]
        Vliquid_dp = traps_dict[trap_name][6]
        OWC_area_km2 = traps_dict[trap_name][7]
        over_press_psi = traps_dict[trap_name][8]
        iclass = traps_dict[trap_name][9]
        mig_eff = traps_dict[trap_name][10]
        print(">>> Working on trap : name, x, y : ", trap_name, x, y)     
        output_list = []
        all_trap_mul = []
        Moil_trap_g_ini=0.0
        Mgas_trap_g_ini=0.0        
        for irm, age_strm in enumerate(res_age_strs):             
            ir = nages-1-irm            
            age_str = res_age_strs[ir]
            age = float(age_str)            
            name_elem = "res"+age_str            
            print(">>> Working on reservoir age: ", age)           
            if idepth_type == 0:
                prefix = "DEPTH_Initial"
            else:
                prefix = "DEPTH_Updated"                
            # Name of water bottom map
            wb_name = (
                          prefix + "_EV_" + ev_IDs[ir] + "_t" + ev_IDs[ir]
                        + "_AGE_" + age_str + "_TOP_" + wb_names[ir] + ".dat" )           
            # Name of temperature file for water bottom
            temp_wb_name = (
                           "TEMP_C_EV_"+ev_IDs[ir]+"_t"+ev_IDs[ir]
                         + "_AGE_" + age_str + "_TOP_" + wb_names[ir] + ".dat")           
            # Name of reservoir depth map in subsea meters
            res_name = (prefix + "_EV_"+ev_IDs[ir] + "_" + res_tID
                        + "_AGE_" + age_str + "_TOP_" + res_name_top + ".dat")            
            # Name of temperature file for reservoir
            temp_name = ("TEMP_C_EV_" + ev_IDs[ir] + "_" + res_tID
                         + "_AGE_" + age_str + "_TOP_" + res_name_top + ".dat")         
            # Standardized names (you shouldn't have to change these)
            bGOR_name = ("MIGPOLY_GOR_g_g_post_trap" + "_res"
                         + age_str + "_" + "incr" + ".dat")
            secgas_name = ("MIGPOLY_SecondaryGas_Tg_post_trap"
                           + "_res" + age_str + "_" + "incr" + ".dat")
            pgas_name = ("MIGPOLY_PrimaryGas_Tg_post_trap"
                         + "_res" + age_str + "_" + "incr" + ".dat")
            poil_name = ("MIGPOLY_PrimaryOil_Tg_post_trap"
                         + "_res" + age_str + "_" + "incr" + ".dat")
            api_name = ("MIGPOLY_API_post_trap" + "_res" + age_str 
                        + "_" + "cumu" + ".dat")            
            bGOR_dir = os.path.join(output_path, "GOR_Bulk")            
            # Path to depth maps files with same resolution 
            # and zmap structure as inputs
            depth_dir = os.path.join(output_path, "Depth_ZMaps")                
            # path to temperature directory
            temp_dir = os.path.join(output_path, "Temperature_ZMaps")         
            wb_dir = depth_dir
            res_dir = depth_dir            
            # Updates masses for a single event
            (
                sphase, satstate, 
                bGOR_trap_scf_bbl, GOR_res_oil_scf_bbl,
                GOR_res_gas_scf_bbl, Vhc_final_m3, 
                perc_fill, liq_fill_frac,
                fluid_gas_fill_frac, Vfree_oil_trap_GOB, 
                Vfree_fluid_gas_Tcf, trap_mul,
                sfill, Msatliq_tot_g, 
                Munsatliq_tot_g, Mfluid_gas_tot_g,
                Mfluid_gas_trap_g, Mdisoil_trap_g,
                Msatliq_trap_g, Munsatliq_trap_g,
                Mdisgas_trap_g, Mfree_oil_trap_g, 
                Mfree_gas_trap_g, Moil_trap_g,
                Mgas_trap_g, Tres_C, TC_surf, WD_m, 
                res_depth_smm, fpress_psi,
                rho_free_gas_trap_kg_m3, 
                rho_satliq_trap_kg_m3,
                rho_unsatliq_trap_kg_m3, 
                Oil_API, bGOR_charge_scf_bbl
            ) = update_trap_masses(
                                    Moil_trap_g_ini, Mgas_trap_g_ini, wb_dir, 
                                    wb_name, res_dir,
                                    res_name, temp_dir, 
                                    temp_name, temp_wb_name, 
                                    bGOR_dir, bGOR_name, sgrav_gas,
                                    secgas_name, pgas_name, 
                                    poil_name, api_name,
                                    free_gas_fraction_theshold, 
                                    single_phase_risk_range,
                                    sGOR_at_charp, charp, exp_fac, 
                                    a_gas, b_gas, rho_sea,
                                    rho_brine, name_elem, sGORflag, x, y, 
                                    Vhcp, Vgas_sp, Vliquid_sp,
                                    Vgas_dp, Vliquid_dp, OWC_area_km2, 
                                    over_press_psi, iclass,
                                    oil_den_mod_fac, gas_den_mod_fac, 
                                    delta_rho_gas_kg_m3, sGOR_scf_bbl_max, 
                                    mig_eff, idepth_type
                                    )
            ioutput_trap_info = 0
            if ioutput_trap_info == 1:
                print(">> iclass : ", iclass)
                print(">> phase state : ", sphase)
                print(">> saturation state : ", satstate)
                print(">>--------------------------------------------")
                print(">> Msatliq_tot_Tg (Charge): ", Msatliq_tot_g/1e12)        
                print(">> Munsatliq_tot_Tg (Charge): ", Munsatliq_tot_g/1e12)
                print(">> Mfluid_gas_tot_Tg (Charge): ", Mfluid_gas_tot_g/1e12)
                print(">>--------------------------------------------")
                print(">> Moil_trap_Tg initial : ", Moil_trap_g_ini/1e12)        
                print(">> Mgas_trap_Tg iniitial : ", Mgas_trap_g_ini/1e12)
                print(">> Moil_trap_Tg final: ", Moil_trap_g/1e12)
                print(">> Mgas_trap_Tg final: ", Mgas_trap_g/1e12)
                print(">>--------------------------------------------")
                print(">> Mfluid_gas_trap_Tg : ", Mfluid_gas_trap_g/1e12)
                print(">> Mdisoil_trap_Tg : ", Mdisoil_trap_g/1e12)        
                print(">> Msatliq_trap_Tg : ", Msatliq_trap_g/1e12)        
                print(">> Munsatliq_trap_Tg : ", Munsatliq_trap_g/1e12)
                print(">> Mdisgas_trap_Tg : ",  Mdisgas_trap_g/1e12)        
                print(">> Mfree_oil_trap_Tg : ",  Mfree_oil_trap_g/1e12)        
                print(">> Mfree_gas_trap_Tg : ",  Mfree_gas_trap_g/1e12)
                print(">>--------------------------------------------")
                print(">> bGOR_trap_scf_bbl : ", bGOR_trap_scf_bbl)
                print(">> GOR_res_oil_scf_bbl : ", GOR_res_oil_scf_bbl)
                print(">> GOR_res_gas_scf_bbl : ", GOR_res_gas_scf_bbl)
                print(">>--------------------------------------------")
                print(">> Vhc_final_m3 : ",  Vhc_final_m3)
                print(">> perc_fill vol % : ",  perc_fill*100)
                print(">> Liquid fill % : ", liq_fill_frac*100)
                print(">> Fluid gas fill % : ", fluid_gas_fill_frac*100)
                print(">> Vfree_oil_trap_GOB : ", Vfree_oil_trap_GOB)
                print(">> Vfree_fluid_gas_Tcf : ", Vfree_fluid_gas_Tcf)
                print(">> trap multiple : ", trap_mul)
                print(">> fill state : ",  sfill)            
            Moil_trap_g_ini = Moil_trap_g 
            Mgas_trap_g_ini = Mgas_trap_g
            all_trap_mul.append(trap_mul)         
            output_list.append(
                                [
                                age, sphase, satstate, bGOR_trap_scf_bbl, 
                                GOR_res_oil_scf_bbl,
                                GOR_res_gas_scf_bbl, bGOR_charge_scf_bbl, 
                                Vhc_final_m3, perc_fill, liq_fill_frac,
                                fluid_gas_fill_frac, Vfree_oil_trap_GOB, 
                                Vfree_fluid_gas_Tcf, trap_mul,
                                sfill, Msatliq_tot_g, 
                                Munsatliq_tot_g, Mfluid_gas_tot_g,
                                Mfluid_gas_trap_g, Mdisoil_trap_g, 
                                Msatliq_trap_g, Munsatliq_trap_g,
                                Mdisgas_trap_g, Mfree_oil_trap_g, 
                                Mfree_gas_trap_g, Moil_trap_g,
                                Mgas_trap_g, rho_free_gas_trap_kg_m3, 
                                rho_satliq_trap_kg_m3, rho_unsatliq_trap_kg_m3, 
                                Tres_C, TC_surf, WD_m, res_depth_smm,
                                fpress_psi, Oil_API, sgrav_gas
                                ]
                                )
        # Caluclate total integrated trap multiple
        trap_mul_max = max(all_trap_mul)
        ifill = 0
        if trap_mul_max >= 1:
            ifill = 1
        excess_charge = 0.0
        for tmul in all_trap_mul:
            if tmul > 1:
                excess_charge = excess_charge + tmul - 1.0
        if ifill == 0:
            trap_mul_tot = trap_mul_max
        else:
            trap_mul_tot = excess_charge + 1.0
        # Create output file for this trap
        fileIO.make_trap_csv(
                                output_path_new, trap_name, nages, 
                                output_strs_list, output_list, trap_mul_tot
                                )