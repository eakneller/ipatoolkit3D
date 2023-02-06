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
import math
import bulk_props
from numba import jit


@jit(nopython=True, cache=True)
def compact_or_decompact(phi_o, c, z1o, z2o, z1n):
    thick_old = z2o-z1o    
    z2n = z1n + thick_old
    z2n_ini = z2n
    z1n_ini = z1n
    TOL = 1e-6
    vcheck = 1e32
    TOL2 = 100.0    
    icount = 0
    nmax = 5  
    while vcheck > TOL and icount < nmax:
        iskip = 0        
        if abs(c*z1o) > TOL2:
            iskip = 1
        if abs(c*z2o) > TOL2:
            iskip = 1
        if abs(c*z1n) > TOL2:
            iskip = 1
        if abs(c*z2n) > TOL2:
            iskip = 1            
        if iskip == 0:
            exp1 = phi_o/c*(math.exp(-c*z1o)-math.exp(-c*z2o))
            exp2 = phi_o/c*(math.exp(-c*z1n)-math.exp(-c*z2n))
            thick_new = z2o - z1o - exp1 + exp2
            z2n = z1n + thick_new
            vcheck = abs(thick_new-thick_old)
            thick_old = thick_new
            icount = icount + 1
        else:
            vcheck = TOL + 1e32
            z2n = z2n_ini
            z1n = z1n_ini
    decom_thick = z2n - z1n
    if decom_thick < 0.0:
        decom_thick = 0.001
    return decom_thick


def compact_or_decompact_form2(phi_o, lamda, z1o, z2o, z1n):  
    # Define old thicknesses
    thick_orig = z2o - z1o    
    phi_b_orig = bulk_props.bulk_porosity_layer(phi_o, 1.0/lamda, z1o, z2o)
    thick_solid_orig = (1.0-phi_b_orig)*thick_orig    
    # Define new total
    z2n = z1n + thick_orig        
    TOL = 1e-10
    vcheck = 1e32        
    icount = 0
    iter_max = 10
    while vcheck > TOL and icount < iter_max:
        # Bulk porosity for component 1 using old total layer depth       
        # Bulk porosity for component 1 using new information
        phi_b_n = bulk_props.bulk_porosity_layer(phi_o, 1.0/lamda, z1n, z2n)        
        # Calculate new thickness for c1 layer
        thick_new = (1.0-phi_b_orig)/(1.0-phi_b_n)*thick_orig                
        # Calculate the new depth
        z2n = z1n+thick_new
        phi_b_f = bulk_props.bulk_porosity_layer(phi_o, 1.0/lamda, z1n, z2n)
        thick_solid_new = (1.0-phi_b_f)*thick_new                
        vcheck = abs(thick_solid_new - thick_solid_orig)                
        icount = icount + 1
    decom_thick = z2n-z1n
    return decom_thick


def compact_or_decompact_2CompMix(
                                    phi_o_1, lamda_1, phi_o_2, lamda_2, 
                                    z1o_L, z2o_L, z1n_L, frac_c1, frac_c2
):
    # Normalize fractions
    frac_tot = frac_c1 + frac_c2
    frac_c1 = frac_c1/frac_tot
    frac_c2 = frac_c2/frac_tot   
    # Define old thicknesses
    thick_L_orig = z2o_L - z1o_L
    thick_c1_orig = thick_L_orig*frac_c1
    thick_c2_orig = thick_L_orig*frac_c2
    phi_b_c1_orig = bulk_props.bulk_porosity_layer(
                                            phi_o_1, 1.0/lamda_1, z1o_L, z2o_L)
    phi_b_c2_orig = bulk_props.bulk_porosity_layer(
                                            phi_o_2, 1.0/lamda_2, z1o_L, z2o_L)
    thick_solid_c1_orig = (1.0-phi_b_c1_orig)*thick_c1_orig
    thick_solid_c2_orig = (1.0-phi_b_c2_orig)*thick_c2_orig
    thick_L_old = thick_L_orig
    # Define new total and component depths
    z2n_L = z1n_L + thick_L_old
    TOL = 1e-10
    vcheck = 1e32
    icount = 0
    iter_max = 10
    while vcheck > TOL and icount < iter_max:
        # Component 1
        # Bulk porosity for component 1 using new information
        phi_b_c1_new = bulk_props.bulk_porosity_layer(
                                            phi_o_1, 1.0/lamda_1, z1n_L, z2n_L)
        # Calculate new thickness for c1 layer
        thick_c1_new = (1.0-phi_b_c1_orig)/(1.0-phi_b_c1_new)*thick_c1_orig
        # Component 2
        # Bulk porosity for component 2 using new information
        phi_b_c2_new = bulk_props.bulk_porosity_layer(
                                            phi_o_2, 1.0/lamda_2, z1n_L, z2n_L)
        # Calculate new thickness for c2 layer
        thick_c2_new = (1.0-phi_b_c2_orig)/(1.0-phi_b_c2_new)*thick_c2_orig
        # Sum component layers
        thick_L_new = thick_c1_new + thick_c2_new
        # Update layer depth
        z2n_L = z1n_L + thick_L_new
        # Component 1
        # Update bulk porosity with new thickness
        phi_b_f_c1 = bulk_props.bulk_porosity_layer(
                                            phi_o_1, 1.0/lamda_1, z1n_L, z2n_L)
        # Calculate new solid thickness
        thick_solid_c1_new = (1.0-phi_b_f_c1)*thick_c1_new
        # Use new solid thickness to compute convergence criterion
        vcheck_c1 = abs(thick_solid_c1_new - thick_solid_c1_orig)
        # Component 2
        # Update bulk porosity with new thickness
        phi_b_f_c2 = bulk_props.bulk_porosity_layer(
                                            phi_o_2, 1.0/lamda_2, z1n_L, z2n_L)
        # Calculate new solid thickness
        thick_solid_c2_new = (1.0-phi_b_f_c2)*thick_c2_new
        # Use new solid thickness to compute convergence criterion
        vcheck_c2 = abs(thick_solid_c2_new - thick_solid_c2_orig)                
        vcheck = vcheck_c1 + vcheck_c2
        print ("---------------------------------------------------------")
        print ("Iteration, vcheck : ", icount, vcheck)
        print ("z1n_L, z2o_L : ", z1n_L, z2n_L)
        print ("phi_b_c1_orig, phi_b_c1_n : ", phi_b_c1_orig, phi_b_c1_new)
        print ("phi_b_c2_oorig, phi_b_c2_n : ", phi_b_c2_orig, phi_b_c2_new)
        print ("thick_c1_orig, thick_c1_new : ", thick_c1_orig, thick_c1_new)
        print ("thick_c2_orig, thick_c2_new : ", thick_c2_orig, thick_c2_new)
        print ("thick_solid_c1_orig, thick_solid_c1_new : ", 
                                       thick_solid_c1_orig, thick_solid_c1_new)
        print ("thick_solid_c2_orig, thick_solid_c2_new : ", 
                                       thick_solid_c2_orig, thick_solid_c2_new)
        print ("thick_L_orig, thick_L_new : ", thick_L_orig, thick_L_new)
        print ("---------------------------------------------------------")
        print ("")
        icount = icount + 1
    decom_thick = z2n_L-z1n_L
    return decom_thick


def compact_or_decompact_from_bottom(phi_o, c, z1o, z2o, z2n):
    thick_old = z2o-z1o
    z1n = z2n - thick_old
    TOL = 1e-10
    vcheck = 1e32
    icount = 0
    while vcheck > TOL:
        exp1 = phi_o/c*(math.exp(-c*z1o)-math.exp(-c*z2o))
        exp2 = phi_o/c*(math.exp(-c*z1n)-math.exp(-c*z2n))
        thick_new = z2o-z1o-exp1+exp2
        z1n = z2n-thick_new
        vcheck = abs(thick_new-thick_old)
        thick_old = thick_new
        icount = icount + 1
    decom_thick = z2n-z1n
    return decom_thick


def compact_layers(
                    icompact,event_age, event_ID, 
                    event_dict, tops_list, ilast_top_index,\
                    Lx, Ly, nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                    idebug_out, icheck_list, jcheck_list
):
    ETOL = 1e-6
    if icompact == 1:
        ntops = ilast_top_index+1
        for jj in range(ntops): 
            # Looping from youngest to oldest
            kk = ntops - jj - 1
            event_index = tops_list[kk][14][event_ID]
            for i in range(nx):
                for j in range(ny):
                    # Is this an erosion node?
                    e_check = event_dict[event_ID][3][i][j]
                    if e_check <= 0.0+ETOL:
                        # Have we depsoited any sediment?
                        nelem2 = len(tops_list[ilast_top_index][1])
                        event_index2 = nelem2-1
                        depo_thick = tops_list[ilast_top_index][2]\
                                                        [event_index2][i][j]
                        if depo_thick > 0.0:
                            if kk < ilast_top_index and kk > 0: 
                                # Skip the newly deposited layer
                                z_pwd = event_dict[event_ID][5][i][j]
                                z_base_level = event_dict[event_ID][6][i][j]
                                z_sed_surf = z_pwd+z_base_level
                                z_top = tops_list[kk][1]\
                                                [event_index][i][j]-z_sed_surf
                                event_index_older = tops_list[kk-1][14]\
                                                                    [event_ID]
                                t_layer = tops_list[kk][2][event_index][i][j]
                                phi_o = tops_list[kk][9]/100.0
                                c = 1.0/(tops_list[kk][10]*1000)
                                # Check thickness
                                len_z_max_xy = len(tops_list[kk][12])
                                if len_z_max_xy > 0:
                                    z_max = tops_list[kk][12][i][j]
                                else:
                                    z_max = 0.0
                                # use the maximum burial depth
                                z1o = z_max
                                # use the pre-compaction layer thickness
                                z2o = z1o + t_layer
                                z1n = z_top
                                if z1n > z_max: 
                                    # Only compact if submud depth is 
                                    # greater than maximum burial depth
                                    decom_thick = compact_or_decompact(
                                                    phi_o, c, z1o, z2o, z1n)
                                    # Update the thickness associated 
                                    # with this top
                                    tops_list[kk][2][event_index][i][j] = \
                                                                    decom_thick
                                    # Update the depth for the next older 
                                    # top using the new thickness
                                    tops_list[kk-1][1]\
                                                    [event_index_older][i][j]\
                                                = z_top+z_sed_surf+decom_thick
