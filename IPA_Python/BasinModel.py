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
import shutil
import os
import glob
import numpy as np
from numba import jit
import print_funcs
import fileIO
import map_tools
import erosion_funcs
import time2depth


def simple_file_copy(path1, path2, filename):
    p1 = os.path.join(path1, filename)
    p2 = os.path.join(path2, filename)
    shutil.copy(p1, p2)
        

@jit(nopython=True, cache=True)
def calc_thickness_and_clean_hor_loop(nx, ny, z_this_top_xy, z_older_top_xy,
                                      thick_xy, deltadepth):
    for i in range(nx):
        for j in range(ny):
            z_this_top = z_this_top_xy[i][j]
            # depth of next oldest top for this event
            z_older_top = z_older_top_xy[i][j] + deltadepth
            t_new = z_older_top - z_this_top
            # eliminate cross horizons
            if t_new < 0.0:
                z_this_top_xy[i][j] = z_older_top
                t_new = 0.0
            thick_xy[i][j] = t_new
            
            
class BasinModel:
    """ Basin model data structures and initialization
    
    The BasinModel class stores main model parameters for a given run
    including the data structures tops_list_bs and event_dict_bs which 
    store stratigraphic information through time and event information,
    respectively.
    
    Description of PWD_overwrite_flag_dict
    ======================================
    Dictionary with keys for eqch event set equal to an integer ioverwrite 
    indicating if a paleo-water depth map was entered as user input. 
    ioverwrite = 1 indicates that a map was found and any newly calculated pwd 
    maps should be overwritten by the user defined map. 
    
    Description of top_list_bs
    ===========================================================================
    The object top_list_bs is a multi-dimensional list of mixed data types with
    dimensions ntops where ntops is the number of stratigraphic surfaces 
    (i.e. tops) defined in the model including the basement surface. Tops
    are ordered from oldest to youngest.
    
    For each top kk, lists are defined with different information types that
    are denoted with an integer ID. For example, tops_list_bs[kk][itype] refers
    to a list of information for information type itype. An integer flag is 
    used instead of a dictionary string name for improved computational 
    performance.
    
    itype = 0: tops_list_bs[kk][0]
    ==============================
    List of integer event ID's when this top was present. This list includes 
    the event ID's associated with the deposition of the top kk and all events 
    younger than the top kk. Each list for top kk has Nevent integers where 
    Nevent is the number of events when top kk was present. For example, 
    for case with 4 depositional events (i.e. 4 tops)...
    
    tops_list_bs[3][0] = [E3]
    tops_list_bs[2][0] = [E2, E3]
    tops_list_bs[1][0] = [E1, E2, E3]  
    tops_list_bs[0][0] = [E0, E1, E2, E3]

    where E# refers to an integer event ID. E# at 0 for the oldest event and 
    becomes larger with younger events.

    itype = 1: tops_list_bs[kk][1]
    ==============================
    List of numpy float arrays with dimensions (nx,ny) that define the depth 
    in meters of the top kk through time for all events when the top kk was 
    present. nx and ny refer to the number of grid points in the x- and 
    y-directions respectively. Each list for top kk has Nevent numpy 
    arrays where Nevent is the number of events when top kk was present. 
    For example, for case with 4 depositional events (i.e. 4 tops):
    
    tops_list_bs[3][1] = [depth_xy_E3]
    tops_list_bs[2][1] = [depth_xy_E2, depth_xy_E3]
    tops_list_bs[1][1] = [depth_xy_E1, depth_xy_E2, depth_xy_E3] 
    tops_list_bs[0][1] = [depth_xy_E0, depth_xy_E1, depth_xy_E2, depth_xy_E3]

    where depth_xy refers to a numpy array with dimensions (nx,ny) and _E# 
    refers to the event ID.
    
    itype = 2: tops_list_bs[kk][2]
    ==============================
    Same as itype = 1 but with layer thickness in meters of the top kk.
    
    itype = 3: tops_list_bs[kk][3]
    ==============================
    Same as itype = 1 but with erosion magnitude in meters of the top kk.

    itype = 4: tops_list_bs[kk][4]
    ==============================
    Float defining the depositional age (Ma) of top kk. For example, for case 
    with 4 depositional events (i.e. 4 tops):

    tops_list_bs[3][4] = age3
    tops_list_bs[2][4] = age2
    tops_list_bs[1][4] = age1
    tops_list_bs[0][4] = age0

    itype = 5: tops_list_bs[kk][5]
    ==============================
    Float defining a dummy thickness in meters set to 1.0 m by default and 
    used to define a thickness for the basement layer.

    tops_list_bs[3][5] = 1.0
    tops_list_bs[2][5] = 1.0
    tops_list_bs[1][5] = 1.0
    tops_list_bs[0][5] = 1.0

    itype = 6: tops_list_bs[kk][6]
    ==============================
    String defining the name of top kk. For example, for case with 4 
    depositional events (i.e. 4 tops):

    tops_list_bs[3][6] = name3
    tops_list_bs[2][6] = name2
    tops_list_bs[1][6] = name1
    tops_list_bs[0][6] = name0
    
    itype = 7: tops_list_bs[kk][7]
    ==============================
    Unused string.
    
    itype = 8: tops_list_bs[kk][8]
    ==============================
    List of numpy float arrays with dimensions (nx,ny) that define the
    total final erosion in meters of the top kk. nx and ny refer to the number 
    of grid points in the x- and y-directions respectively. Each list for top 
    kk has only one element. For example, for case with 4 depositional events 
    (i.e. 4 tops):

    tops_list_bs[3][8] = [e_tot_xy_3]
    tops_list_bs[2][8] = [e_tot_xy_2]
    tops_list_bs[1][8] = [e_tot_xy_1] 
    tops_list_bs[0][8] = [e_tot_xy_0]

    itype = 9: tops_list_bs[kk][9]
    ==============================
    Unused float

    itype = 10: tops_list_bs[kk][9]
    ===============================
    Unused float

    itype = 11: tops_list_bs[kk][9]
    ===============================
    Unused float

    itype = 12: tops_list_bs[kk][12]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the final maximum 
    burial depth of the top kk. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively. For example, for case with 4 
    depositional events (i.e. 4 tops):

    tops_list_bs[3][12] = maxf_xy_3
    tops_list_bs[2][12] = maxf_xy_2
    tops_list_bs[1][12] = maxf_xy_1
    tops_list_bs[0][12] = maxf_xy_0   

    itype = 13: tops_list_bs[kk][13]
    ================================
    Same as itype = 1 but with maximum burial depth in meters of the top kk.
       
    itype = 14: tops_list_bs[kk][14]
    ================================    
    A dictionary mapping event ID's to list indices referring to numpy map 
    arrays. For example, consider a case with 4 depositional events 
    (i.e. 4 tops) and the following general map history for map information 
    type i:
    
    tops_list_bs[3][i] = [map_xy_E3]
    tops_list_bs[2][i] = [map_xy_E2, map_xy_E3]
    tops_list_bs[1][i] = [map_xy_E1, map_xy_E2, map_xy_E3] 
    tops_list_bs[0][i] = [map_xy_E0, map_xy_E1, map_xy_E2, map_xy_E3]
    
    where map_xy refers to a numpy array with dimensions (nx,ny) and _E# 
    refers to the event ID. The dictionaries for this example will be as
    follows:
        
    tops_list_bs[3][14] = {3:0}
    tops_list_bs[2][14] = {2:0, 3:1}
    tops_list_bs[1][14] = {1:0, 2:1, 3: 2}
    tops_list_bs[0][14] = {0:0, 1:1, 2:2, 3:3}       
    
    These dictionaries are used throughout the code to find the list index
    of an array for a given event. For example, consider top 1 in the example 
    above with the following list of numpy map arrays:
        
    tops_list_bs[1][i] = [map_xy_E1, map_xy_E2, map_xy_E3]
    
    In order to find the map array for event_ID 2 (i.e. E2) first the list 
    index referred to as event_index is determined 
    
        event_index = tops_list_bs[0][14][event_ID]
    
    and then the map can be accessed via
    
        map_xy_E2 =  tops_list_bs[1][i][event_index]  
   
    itype = 15: tops_list_bs[kk][15]
    ================================
    String defining the name zmap file for top kk with total vertical depth 
    in subsea meters.

    itype = 16: tops_list_bs[kk][16]
    ================================
    String defining the name zmap file for top kk with paleo-water depth 
    in subsea meters at the time of deposition.
    
    itype = 17: tops_list_bs[kk][17]
    ================================
    Same as itype = 1 but with  bulk density in kg/m^3 of the layer asscoated 
    with top kk

    itype = 18: tops_list_bs[kk][18]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the TWT 
    (milliseconds) depth map of the top kk. nx and ny refer to the number 
    of grid points in the x- and y-directions respectively.
   
    itype = 19: tops_list_bs[kk][19]
    ================================
    String defining the name zmap file for top kk with depth in TWT (ms).
   
    itype = 20: tops_list_bs[kk][20]
    ================================
    Float defining the interval velocity in m/s for the layer associated with 
    top kk.
    
    itype = 21: tops_list_bs[kk][21]
    ===============================
    Unused float   
    
    itype = 22: tops_list_bs[kk][22]
    ===============================
    Unused float

    itype = 23: tops_list_bs[kk][23]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the depth map in
    meters from time-to-depth conversion of the top kk. nx and ny refer to the 
    number of grid points in the x- and y-directions respectively.

    itype = 24: tops_list_bs[kk][24]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the interval 
    velocity map in m/s for the top kk. nx and ny refer to the number of 
    grid points in the x- and y-directions respectively.
    
    itype = 25: tops_list_bs[kk][25]
    ================================
    Unused float

    itype = 26: tops_list_bs[kk][26]
    ================================
    Numpy float array with dimensions (nx,ny) that defines lithology ID's for 
    the top kk. nx and ny refer to the number of grid points in the x- and 
    y-directions respectively.   
    
    Note that float ID's are converted to integers within functions.
    
    itype = 27: tops_list_bs[kk][27]
    ================================
    String defining the name zmap file for top kk with integer lithology ID's.
    
    itype = 28: tops_list_bs[kk][28]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the matrix 
    velocity map in m/s for the top kk. nx and ny refer to the number of 
    grid points in the x- and y-directions respectively.

    itype = 29: tops_list_bs[kk][29]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the matrix heat 
    production in microW/m^3 for the top kk. nx and ny refer to the number of 
    grid points in the x- and y-directions respectively.

    itype = 30: tops_list_bs[kk][30]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the grain thermal
    conductivity in W/m/K for the top kk. nx and ny refer to the number of 
    grid points in the x- and y-directions respectively.

    itype = 31: tops_list_bs[kk][31]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the grain heat 
    capacity in J/kg/K for the top kk. nx and ny refer to the number of 
    grid points in the x- and y-directions respectively.

    itype = 32: tops_list_bs[kk][32]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the grain density
    in kg/m^3 for the top kk. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively.

    itype = 33: tops_list_bs[kk][33]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the surface porosity
    % for the top kk. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively.

    itype = 34: tops_list_bs[kk][34]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the porosity decay 
    depth in km for the top kk. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively.

    itype = 35: tops_list_bs[kk][35]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the d-parameter for
    Gardner's law for the top kk. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively.

    itype = 35: tops_list_bs[kk][35]
    ================================
    Numpy float array with dimensions (nx,ny) that defines the f-parameter for
    Gardner's law for the top kk. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively.
      	 	    	  	 	 
    itype = 37: tops_list_bs[kk][37]
    ================================
    Same as itype = 1 but with steady-state temperature in Celcius of the top 
    kk.
    
    itype = 38: tops_list_bs[kk][38]
    ================================
    Same as itype = 1 but with transient temperature in Celcius of the top 
    kk.
   
    itype = 39: tops_list_bs[kk][39]
    ================================
    Same as itype = 1 but with vitrinite reflectance (EasyRo Ro%) of the top 
    kk.
    
    itype = 40: tops_list_bs[kk][40]
    ================================
    Same as itype = 1 but with LOM of the top kk.
    
    itype = 41: tops_list_bs[kk][41]
    ================================
    Same as itype = 1 but with transformation ratio of the top kk.

    itype = 42: tops_list_bs[kk][42]
    ================================
    Same as itype = 1 but with total mass of generated hydrocarbons in 
    mgHC/gTOC of the top kk.

    itype = 43: tops_list_bs[kk][43]
    ================================
    Same as itype = 1 but with mass of oil and dissolved gas in mgHC/gTOC 
    expelled from the source layer asscociated with top kk.
    
    itype = 44: tops_list_bs[kk][44]
    ================================
    Same as itype = 1 but with mass of free gas in mgHC/gTOC expelled into 
    pore space from the source layer asscociated with top kk.
    
    itype = 45: tops_list_bs[kk][45]
    ================================
    Same as itype = 1 but with Expulsion rate in mgHC/gOC/Myr from the source 
    layer asscociated with top kk.  

    itype = 46: tops_list_bs[kk][46]
    ================================
    Same as itype = 1 but with forward maximum burial in meters for top kk.
    
    itype = 47: tops_list_bs[kk][47]
    ================================
    List of lists with mixed type. The primary list has dimensions (nx,ny) 
    where nx and ny refer to the number of grid points in the x- and 
    y-directions respectively. Each element of the primary list is a list of 
    mixed type storing source parameters. For example, for coordinate (i,j):
        
        src_params = tops_list_bs[kk][47][i][j]
        
        src_params[0]
            String defining organic matter type:
            "Type_I", "Type_II", "Type_III", or "Type_IIS" 
        src_params[1]
            Float defining the hydrogen index in mgHC/gTOC
        src_params[2]
            Float defining total organic content in wt%
        src_params[3]
            String defining kinetic model scenario:
            "early", "normal", "late"
        src_params[4]
            Float defining source thickness in meters
        src_params[5]
            unused float
        src_params[6]
            Float defining oil api density of oil generated from source
        src_params[7]
            Float defining gas gravity of gas generated from source
        src_params[8]
            Float defining the porosity threshold fraction required for source
            expulsion
        src_params[9]
            String defining the gas fraction scenario:
            "minimum", "base" or "maximum"
        src_params[10]
            String defining the polar fraction scenario:
            "minimum", "base" or "maximum"
    
    itype = 48: tops_list_bs[kk][48]
    ================================
    Same as itype = 1 but with primary residual gas mass (g) from the source 
    layer asscociated with top kk used for secondary cracking calculations.
    
    itype = 49: tops_list_bs[kk][49]
    ================================
    Same as itype = 1 but with primary residual oil mass (g) from the source 
    layer asscociated with top kk used for secondary cracking calculations.   
   
    itype = 50: tops_list_bs[kk][50]
    ================================
    Same as itype = 1 but with primary residual coke mass (g) from the source 
    layer asscociated with top kk used for secondary cracking calculations.
    
    itype = 51: tops_list_bs[kk][51]
    ================================
    Same as itype = 1 but with primary residual liquid (oil + dissolved gas in 
    pores and adsorped) (mg HC / g TOC) from the source layer associated with 
    top kk used for secondary cracking calculations.
    
    itype = 52: tops_list_bs[kk][52]
    ================================
    Same as itype = 1 but with volume of gas (surface conditions, Tcf) 
    expelled from the source layer associated with top kk.
    
    itype = 53: tops_list_bs[kk][53]
    ================================
    Same as itype = 1 but with volume of liquid (oil + dissolved gas) 
    (surface conditions, GOB) expelled from the source layer associated with 
    top kk.
    
    itype = 54: tops_list_bs[kk][54]
    ================================
    Same as itype = 1 but with the density of primary free gas (g/cm3) at 
    source rock conditions.
    
    itype = 55: tops_list_bs[kk][55]
    ================================
    Same as itype = 1 but with the density of primary liquid (oil + dissolved 
    gas) (g/cm3) at source rock conditions.
    
    itype = 56: tops_list_bs[kk][56]
    ================================
    Same as itype = 1 but with the Bg FVF of gas (rcf/scf) for primary 
    generation at source rock conditions.
    
    itype = 57: tops_list_bs[kk][57]
    ================================
    Same as itype = 1 but with the Bo FVF oil (rbbl/stb) for primary generation
    at source rock conditions.
    
    itype = 58: tops_list_bs[kk][58]
    ================================
    Same as itype = 1 but with the mass of secondary gas expelled into pore 
    space (mgHC/gOC) at source rock conditions.
    
    itype = 59: tops_list_bs[kk][59]
    ================================
    Same as itype = 1 but with the Volume of secondary gas  (surface 
    conditions, Tcf) expelled from source associated with top kk.
   
    itype = 60: tops_list_bs[kk][60]
    ================================
    Same as itype = 1 but with the density of secondary free gas (g/cm3) at 
    source rock conditions.
    
    itype = 61: tops_list_bs[kk][61]
    ================================
    Same as itype = 1 but with the Bg FVF of gas (rcf/scf) for secondary 
    cracking at source rock conditions.   
    
    itype = 62: tops_list_bs[kk][62]
    ================================
    Same as itype = 1 but with secondary gas expulsion rate in mgHC/gOC/Myr 
    for the source layer associated with top kk.
   
    itype = 63: tops_list_bs[kk][63]
    ================================
    Same as itype = 1 but with primary generated gas mass in g for the source 
    layer associated with top kk.   
    
    itype = 64: tops_list_bs[kk][64]
    ================================
    Same as itype = 1 but with primary generated oil mass in g for the source 
    layer associated with top kk.  

    itype = 65: tops_list_bs[kk][65]
    ================================
    Same as itype = 1 but with primary generated coke mass in g for the source 
    layer associated with top kk.  
    
    itype = 66: tops_list_bs[kk][66]
    ================================
    Same as itype = 1 but with secondary generation mass (Gas + Coke)  in 
    mgHC/gOC) for the source layer associated with top kk. 
    
    itype = 67: tops_list_bs[kk][67]
    ================================
    Same as itype = 1 but with secondary generated gas mass (generated + 
    residual) in g for the source layer associated with top kk.  
    
    itype = 68: tops_list_bs[kk][68]
    ================================
    Same as itype = 1 but with mass of initial organic carbon (primary) in g 
    for the source layer associated with top kk.  
   
    itype = 69: tops_list_bs[kk][69]
    ================================
    Same as itype = 1 but with mass of initial organic carbon in g used for
    secondary cracking for the source layer associated with top kk. This is 
    set equal to the mass of residual oil from primary generation.
    
    itype = 70: tops_list_bs[kk][70]
    ================================
    Same as itype = 1 but with primary fluid GOR (mass of dissolved gas over 
    mass of oil) in g/g for the source layer associated with top kk.   

       
    Description of event_dict_bs
    ===========================================================================
    Dictionary event_dict_bs contains information for Nevents where Nevents is
    the number of events. Events are ordered from oldest to youngest.
    
    key = 0: event_dict_bs[mm][0]
    =============================
    Float defining the age in Ma of event mm.

    key = 1: event_dict_bs[mm][1]
    =============================
    String defining the type of event for event mm. Options include 
    "Deposition" or "Erosion or Deposition".

    key = 2: event_dict_bs[mm][2]
    =============================
    Integer defining the top index kk for the event.

    key = 3: event_dict_bs[mm][3]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the total eroded
    thickness in meters in the maximum compaction state for event mm. nx and 
    ny refer to the number of grid points in the x- and y-directions 
    respectively.

    key = 4: event_dict_bs[mm][4]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the shift map in 
    meters for event mm. nx and ny refer to the number of grid points in the 
    x- and y-directions respectively.

    key = 5: event_dict_bs[mm][5]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the paleo-water 
    depth in meters for event mm. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively.

    key = 6: event_dict_bs[mm][6]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the base level in 
    meters for event mm. nx and ny refer to the number of grid points 
    in the x- and y-directions respectively. The convention used involves 
    negative values for elevations above present-day sea level and positive 
    below. 

    key = 7: event_dict_bs[mm][7]
    =============================
    Numpy float array with dimensions (nx,ny) that defines where paleo-water
    depth was adjusted for event mm. nx and ny refer to the number of grid 
    points in the x- and y-directions respectively. The convention used 
    involves negative values for elevations above present-day sea level and 
    positive below. 

    key = 8: event_dict_bs[mm][8]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the thermo-tectonic
    subsidence in meters for event mm. nx and ny refer to the number of grid 
    points in the x- and y-directions respectively.
    
    key = 9: event_dict_bs[mm][9]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the total heat 
    flow in mW/m^2 for event mm. nx and ny refer to the number of grid 
    points in the x- and y-directions respectively.
    
    key = 10: event_dict_bs[mm][10]
    =============================
    String defining the name of shift map file for event mm.
    
    key = 11: event_dict_bs[mm][11]
    ===============================
    String defining the name of the PWD file for event mm.

    key = 12: event_dict_bs[mm][12]
    ===============================
    Numpy float array with dimensions (nx,ny) that defines the bulk sediment 
    density (kg/m/m/m) for event mm. nx and ny refer to the number of grid 
    points in the x- and y-directions respectively.
    
    key = 13: event_dict_bs[mm][13]
    ===============================
    Numpy float array with dimensions (nx,ny) that defines the pwd correction 
    in meters (residual or anomalous subsidence) for event mm. nx and ny refer 
    to the number of grid points in the x- and y-directions respectively.
    
    key = 14: event_dict_bs[mm][14]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the forward modeled
    therm-tectonic subsidence in meters for event mm. nx and ny refer to the 
    number of grid points in the x- and y-directions respectively.

    key = 15: event_dict_bs[mm][15]
    =============================
    Numpy float array with dimensions (nx,ny) that defines the anomalous heat 
    flow in mW/m^2 for event mm. nx and ny refer to the number of grid 
    points in the x- and y-directions respectively.

    key = 16: event_dict_bs[mm][16]
    ===============================
    Numpy float array with dimensions (nx,ny) that defines the upper crustal 
    load (above detachment) in MPa for event mm. nx and ny refer to the 
    number of grid points in the x- and y-directions respectively.

    key = 17: event_dict_bs[mm][17]
    ===============================
    Numpy float array with dimensions (nx,ny) that defines the thermal load 
    in MPa for event mm. nx and ny refer to the number of grid points in the 
    x- and y-directions respectively.

    key = 18: event_dict_bs[mm][18]
    ===============================    
    Numpy float array with dimensions (nx,ny) that defines the incremental 
    flexural deflection in meters for event mm. nx and ny refer to the number 
    of grid points in the x- and y-directions respectively.
    
    key = 19: event_dict_bs[mm][19]
    ===============================
    Numpy float array with dimensions (nx,ny) that defines the effective 
    elastic thickness in meters for event mm. nx and ny refer to the number of 
    grid points in the x- and y-directions respectively.

    key = 20: event_dict_bs[mm][20]
    ===============================      
    Numpy float array with dimensions (nx,ny) that defines the sediment load 
    in MPa for event mm. nx and ny refer to the number of grid points in the 
    x- and y-directions respectively.

    key = 21: event_dict_bs[mm][21]
    ===============================      
    Numpy float array with dimensions (nx,ny) that defines the total sediment
    thickness in meters for event mm. nx and ny refer to the number of grid 
    points in the x- and y-directions respectively.
    
    key = 22: event_dict_bs[mm][22]
    ===============================  
    Numpy float array with dimensions (nx,ny) that defines the paleo-water
    depth in meters based on flexural backstripping (no filtering) for event 
    mm. nx and ny refer to the number of grid points in the x- and 
    y-directions respectively.
    
    key = 23: event_dict_bs[mm][23]
    ===============================
    Numpy float array with dimensions (nx,ny) that defines salt thickness 
    in meters for event mm. nx and ny refer to the number of grid points in 
    the x- and y-directions respectively.
    
    key = 24: event_dict_bs[mm][24]
    ===============================  
    String defining the map file name for salt for event mm.
    
    key = 25: event_dict_bs[mm][25]
    ===============================
    Numpy float array with dimensions (nx,ny) that defines paleo-water depth 
    in meters based on flexural backstripping with low-pass filtering for 
    event mm. nx and ny refer to the number of grid points in the x- and 
    y-directions respectively.
    
    key = 26: event_dict_bs[mm][26]
    ===============================  
    Numpy float array with dimensions (nx,ny) that defines basement depth 
    in meters based on flexural backstripping for event mm. nx and ny refer 
    to the number of grid points in the x- and y-directions respectively.
    """ 
    
    def __init__(self, input_path, output_path, script_path,
                 itype3D, inode, jnode, input1D_dict):
        self.input_path = input_path
        self.output_path = output_path
        self.script_path = script_path
        self.itype3D = itype3D
        self.inode = inode
        self.jnode = jnode
        self.input1D_dict = input1D_dict
        
        
    def load_main_input_file(self):
        self.strat_input_dict = {}
        self.param_input_dict = {}
        input_file_path = os.path.join(self.input_path, "ipa_input.csv")
        fileIO.read_main_input_file_csv(
                                         input_file_path, 
                                         self.strat_input_dict, 
                                         self.param_input_dict
                                        )
        for k, v in self.param_input_dict.items():
            setattr(self, k, v)
       
        
    def clean_inputs(self, riskmodel, ioutput_main):
        self.param_input_dict["rad_search_m"] = (
                                self.param_input_dict["rad_search_m"]*1000.0)
        self.rad_search_m = self.param_input_dict["rad_search_m"]
        self.inert_frac = self.inert_frac + riskmodel.delta_inert
        self.adsorption_perc = self.adsorption_perc + riskmodel.delta_adsth
        self.temp_elastic = self.temp_elastic + riskmodel.delta_Telastic    
        self.dist_taper = self.dist_taper*1000/2
        if self.adsorption_perc < 0.0:
            self.adsorption_perc = 0
        if self.inert_frac < 0:
            self.inert_frac = 0
        if self.temp_elastic < 0:
            self.temp_elastic = 1.0        
        if self.imass_gen == 2:
            self.ioutput_LOM = 2
            self.ioutput_TR = 2    
        # If not using flexure do not restore salt in burial history
        if self.iuse_flexure == 0:
            if self.isalt_restore == 1:
                 # Do not update paleo-water depth if you are restoring salt
                self.iupdate_PWD = 0    
        if self.icalc_temp == 0:
            self.imass_gen = 0
            self.icalc_LOM = 0        
        if self.itype3D == 0:
            self.imass_gen = 1
            self.icalc_LOM = 1
            self.icalc_temp = 1
        if self.dt_Myr > self.dt_rift_Myr:
            self.dt_Myr = self.dt_rift_Myr
            if ioutput_main == 1: 
                print('reset dt_Myr equal to dt_rift_Myr')


    def make_ipa_data_structures(self):
        tops_list_bs = []
        event_dict_bs = {}
        ilast_top_index_list_bs = []    
        poly_file_name_list = []
        src_top_names = []
        poly_age_start_list = []
        poly_age_end_list = []
        all_ages = []
        icount = 0        
        # If no trap element is found then assume the trap is always in place
        trap_age = 4567.0
        trap_name = "None"
        trap_ev_ID = -99999    
        keys = self.strat_input_dict.keys() 
        for key in keys:
            # looping from oldest to youngest events
            age = self.strat_input_dict[key][0]
            name = self.strat_input_dict[key][1]
            stype = self.strat_input_dict[key][2]
            file_name_top = self.strat_input_dict[key][3]
            file_name_shift = self.strat_input_dict[key][4]
            file_name_PWD = self.strat_input_dict[key][5]
            file_name_salt = self.strat_input_dict[key][6]
            file_name_top_TWT = self.strat_input_dict[key][7]        
            file_name_lith = self.strat_input_dict[key][8]
            vint =  self.strat_input_dict[key][9]
            file_name_fetch =  self.strat_input_dict[key][11]
            pse = self.strat_input_dict[key][10]        
            all_ages.append(age)        
            if file_name_fetch != "":
                if key == 0:
                    age_prev = age+100.0
                else:
                    na = len(all_ages)
                    age_prev = all_ages[na-2]               
                poly_file_name_list.append(file_name_fetch)
                poly_age_start_list.append(age_prev)
                poly_age_end_list.append(age)            
            if pse == "Source":
                src_top_names.append(name)        
            if pse == "Trap":
                trap_age = age
                trap_name = name
                trap_ev_ID = key        
            if stype in ["Erosion"]:
                icount = icount - 1
            # element index is included with comments for easier lookup using
            # the list in docstring above            
            event_dict_bs[key] = [
                    age, # 0
                    stype, # 1
                    icount, # 2
                    [], # 3
                    [], # 4
                    [], # 5
                    [], # 6
                    [], # 7
                    [], # 8
                    [], # 9
                    file_name_shift, # 10
                    file_name_PWD,  # 11
                    [], # 12
                    [], # 13 
                    [], # 14 
                    [], # 15 
                    [], # 16 
                    [], # 17 
                    [], # 18 
                    [], # 19 
                    [], # 20 
                    [], # 21 
                    [], # 22 
                    [], # 23
                    file_name_salt, # 24 
                    [], # 25
                    [] # 26
                ]        
            ilast_top_index_list_bs.append(icount)
            if stype in ["Deposition","Erosion_and_Deposition"]:
                # element index is included with comments for easier 
                # lookup using
                # the list in module docstring located above
                tops_list_bs.append(
                   [
                    [], # itype = 0
                    [], # itype = 1
                    [], # itype = 2
                    [], # itype = 3
                    age, # itype = 4
                    1.0, # itype = 5
                    name, # itype = 6
                    "Not_Used", # itype = 7: Not used 
                    [], # itype = 8
                     -99999.0, # itype = 9
                     -99999.0, # itype = 10
                     -99999.0, # itype = 11
                    [], # itype = 12
                    [], # itype = 13
                    {}, # itype = 14
                    file_name_top, # itype = 15
                    file_name_PWD, # itype = 16
                    [], # itype = 17
                    [], # itype = 18
                    file_name_top_TWT, # itype = 19
                    vint, # itype = 20
                     -99999.0, # itype = 21
                     -99999.0, # itype = 22
                    [], # itype = 23
                    [], # itype = 24
                     -99999.0, # itype = 25
                    [], # itype = 26
                    file_name_lith, # itype = 27
                    [], # itype = 28
                    [], # itype = 29
                    [], # itype = 30
                    [], # itype = 31
                    [], # itype = 32
                    [], # itype = 33
                    [], # itype = 34
                    [], # itype = 35
                    [], # itype = 36
                    [], # itype = 37
                    [], # itype = 38
                    [], # itype = 39
                    [], # itype = 40
                    [], # itype = 41
                    [], # itype = 42
                    [], # itype = 43
                    [], # itype = 44
                    [], # itype = 45
                    [], # itype = 46
                    [], # itype = 47
                    [], # itype = 48
                    [], # itype = 49
                    [], # itype = 50
                    [], # itype = 51
                    [], # itype = 52
                    [], # itype = 53
                    [], # itype = 54
                    [], # itype = 55
                    [], # itype = 56
                    [], # itype = 57
                    [], # itype = 58
                    [], # itype = 59
                    [], # itype = 60
                    [], # itype = 61
                    [], # itype = 62
                    [], # itype = 63
                    [], # itype = 64
                    [], # itype = 65
                    [], # itype = 66
                    [], # itype = 67
                    [], # itype = 68
                    [], # itype = 69
                    [] # itype = 70
                   ]
                )
            icount = icount + 1
            TTS_FW_dict = {}
        # res ages are from oldest to youngest
        # bulk post trap GOR will be calculated for these ages
        res_ages = []
        wb_names = []
        keys = self.strat_input_dict.keys()
        for key in keys:
            age = self.strat_input_dict[key][0]
            name = self.strat_input_dict[key][1]
            if age < trap_age:
                res_ages.append(age)
                wb_names.append(name)   
        keys = list(event_dict_bs.keys())
        nevents_bs = len(keys)
        event_ID_list_bs = keys[:]
        event_ID_last_bs = event_ID_list_bs[nevents_bs-1]    
        res_name_top = trap_name
        res_tID = "t"+str(trap_ev_ID)
        res_age_strs = []    
        res_ev_IDs = []
        res_wb_names = []
        nrages = len(res_ages)
        for ii in range(nrages):
            i = nrages - 1 - ii
            res_age_strs.append(str(res_ages[i]))
            res_wb_names.append(wb_names[i])
            res_ev_IDs.append(str(event_ID_last_bs-ii))       
        self.event_dict_bs = event_dict_bs
        self.tops_list_bs = tops_list_bs
        self.ilast_top_index_list_bs = ilast_top_index_list_bs
        self.poly_file_name_list = poly_file_name_list
        self.poly_age_start_list = poly_age_start_list
        self.src_top_names = src_top_names
        self.poly_age_end_list = poly_age_end_list
        self.TTS_FW_dict = TTS_FW_dict
        self.all_ages = all_ages
        self.trap_age = trap_age
        self.trap_name = trap_name
        self.trap_ev_ID = trap_ev_ID
        self.res_ages = res_ages
        self.wb_names = wb_names
        self.res_name_top = res_name_top
        self.res_tID = res_tID
        self.res_age_strs = res_age_strs
        self.res_ev_IDs = res_ev_IDs
        self.res_wb_names = res_wb_names
    
    
    def make_file_prefix_and_suffix(self): 
        self.prefix_all = [
                "DEPTH", 
                "EasyRo", 
                "LOM", 
                "mHC_mg_gOC", 
                "EXPRATE_mg_gOC",
                "SEC_EXPRATE_mg_gOC", 
                "HF", "mFG_mg_gOC",
                "mODG_mg_gOC",
                "PWD", 
                "SEC_mFG_mg_gOC",
                "TEMP", 
                "TR", 
                "ResidualSub",
                "Source_Expulsion_History_",
                "Moho", 
                "best_fit_xth",
                "best_fit_deltas", 
                "TTS", 
                "Salt", 
                "Flex", 
                "cYield",
                "iYield",
                "Gen",
                "GOR", 
                "ageLOM", 
                "D2LOM"
        ]
        self.suffix_all = ["_History.csv", "ModelOutput.csv"]


    def make_output_directories(self, ioutput_main):   
        bs_depth_direc = "Depth_ZMaps"
        easyro_direc = "EasyRo_ZMaps"
        lom_direc = "LOM_ZMaps"
        mass_gen_direc = "HC_Mass_Gen_ZMaps"
        pore_exp_rate_direc = "Pore_Expulsion_Rate_ZMaps"
        sec_pore_exp_rate_direc = "Secondary_Pore_Expulsion_Rate_ZMaps"
        hf_direc = "HeatFlow_ZMaps"
        free_gas_pore_exp_direc = "FreeGas_Pore_Expulsion_ZMaps"
        liquid_pore_exp_direc = "Liquid_Pore_Expulsion_ZMaps"
        pwd_direc = "PWD_ZMaps"
        sec_free_gas_pore_exp_direc = "Secondary_FreeGas_Pore_Expulsion_ZMaps"
        temp_direc = "Temperature_ZMaps"
        tr_direc = "Transformation_Ratio_ZMaps"
        res_sub_direc = "Residual_Subsidence"
        well_extraction_direc = "Well_Extractions"
        total_sr_exp_direc = "Total_Source_Expulsion"
        crustal_struc_direc = "Crustal_Structure"
        TTS_direc = "Thermotectonic_Subsidence"
        Salt_direc = "Salt_Reconstruction"
        Flex_direc = "Flexure_Output"
        cYield_direc = "SRC_Yield_Cumulative"
        iYield_direc = "SRC_Yield_Incremental"
        Gen_direc = "Total_Gen_Masses"
        GOR_direc = "GOR_Yield"
        LOMage_direc = "LOM_AGE"
        LOMdepth_direc = "LOM_DEPTH"
        self.output_direcs_list_A = [
            bs_depth_direc, easyro_direc,
            lom_direc, mass_gen_direc,
            pore_exp_rate_direc, sec_pore_exp_rate_direc,
            hf_direc, free_gas_pore_exp_direc,
            liquid_pore_exp_direc, pwd_direc,
            sec_free_gas_pore_exp_direc, temp_direc,
            tr_direc, res_sub_direc,
            total_sr_exp_direc, crustal_struc_direc,
            crustal_struc_direc, crustal_struc_direc,
            TTS_direc, Salt_direc,
            Flex_direc, cYield_direc,
            iYield_direc, Gen_direc,
            GOR_direc, LOMage_direc,
            LOMdepth_direc
        ]    
        self.output_direcs_list_B = [well_extraction_direc, 
                                     well_extraction_direc]    
        for tmp_direc in self.output_direcs_list_A:        
            test_direc = os.path.join(self.output_path, tmp_direc)   
            if os.path.isdir(test_direc) != True:                                    
                os.mkdir(test_direc)
        for tmp_direc in self.output_direcs_list_B:        
            test_direc = os.path.join(self.output_path, tmp_direc)
            if os.path.isdir(test_direc) != True:
                os.mkdir(test_direc)


    def create_master_grid_from_water_bottom(self):
        # Inititalize AOI numpy array
        AOI_np_ini = np.zeros((1,1))    
        if self.idepth_model == 0:
            file_name_master_grid = (
                            self.event_dict_bs[self.event_ID_last_bs][11]
                            )
        else:
            ntops = len(self.tops_list_bs)
            itop = ntops - 1
            file_name_master_grid = self.tops_list_bs[itop][19]    
        print_funcs.print_info(
                "File name master grid", [file_name_master_grid])
        
        (
            self.master_xy, self.nx, self.ny, self.dx, self.dy, 
            self.xmin, self.xmax, self.ymin, self.ymax, 
            AOI_np_L
        ) = map_tools.read_ZMAP(
                                self.input_path, 
                                file_name_master_grid, 
                                AOI_np_ini
                            )    
        self.zeros_xy_np = np.zeros((self.nx,self.ny))    
         # Used to calculate masses for each cell
        self.cell_area_km2 = (self.dx/1000.0)*(self.dy/1000.0)        
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin    
        self.AOI_np = np.zeros((self.nx,self.ny))    
        for i in range(self.nx): # Columns
            for j in range(self.ny): # Rows        
                # Initialize AOI_np from bathymetry map
                self.AOI_np[i][j] = AOI_np_L[i][j]
                

    def load_lithology_maps(self):
        ntops = len(self.tops_list_bs)
        for jj in range(ntops):
            map_file_name = self.tops_list_bs[jj][27]
            lith_ID_xy = map_tools.read_ZMAP_interp_to_master_integers(
                                        -99999.0, self.master_xy, 
                                        self.input_path, map_file_name, 
                                        self.nx, self.ny, self.xmin, self.ymin, 
                                        self.dx, self.dy, self.AOI_np
                                        ) 
            self.tops_list_bs[jj][26] = np.copy(lith_ID_xy)
        

    def load_SurfTemp(self):
        ifind_SWIT_files = 1       
        try:
            input_file_path = os.path.join(self.script_path, "ipa_data", 
                                           "surfaceTCvsMa.csv")
            (
                self.surf_temp_age, 
                self.surf_temp_lat, 
                self.surf_temp_xy
            ) = fileIO.read_surf_temp_file_csv(input_file_path)
        except:
            ifind_SWIT_files = 0
        if ifind_SWIT_files == 0:
            wstr = (
                    "SWIT data files were NOT found in ipa_data directory." 
                    + "Setting icalc_SWIT to zero."
                    )
            print_funcs.print_warning(wstr)
            self.icalc_SWIT = 0


    def load_APWP(self):
        try:  
            input_file_path = os.path.join(self.script_path, "ipa_data", 
                                           "APWP.csv")
            self.APWP_dict = fileIO.read_APWP_file_csv(input_file_path)
        except:
            self.APWP_dict = {}

            
    def initialize_lithology_arrays(self):
        ntops = len(self.tops_list_bs)
        nx = self.nx
        ny = self.ny
        for jj in range(ntops):
             # Source rock params
            A10_xy = []    
            for i in range(nx):          
                A10_row = []            
                for j in range(ny):              
                    # scenarios are minimum, base and maximum
                    A10_row.append(
                            [
                                "Type_II", # OMT
                                500.0, # HI	
                                4.0, # TOC
                                "normal", # Kinetics	
                                1.0, # Source Thickness (m)
                                1e-6, # Source Area Per Cell (km2)	
                                30.0, # OIL
                                0.8, # gas gravity
                                0.06, # porosity threshold
                                "base", # gas frac scenario
                                "base" # polar frac scenario
                            ]
                        )
                
                A10_xy.append(A10_row)        
            #Initialize propery arrays
            A1_xy = np.zeros((nx, ny))
            A2_xy = np.zeros((nx, ny))
            A3_xy = np.zeros((nx, ny))
            A4_xy = np.zeros((nx, ny))
            A5_xy = np.zeros((nx, ny))
            A6_xy = np.zeros((nx, ny))
            A7_xy = np.zeros((nx, ny))
            A8_xy = np.zeros((nx, ny))
            A9_xy = np.zeros((nx, ny))        
            self.tops_list_bs[jj][28] = np.copy(A1_xy)
            self.tops_list_bs[jj][29] = np.copy(A2_xy)
            self.tops_list_bs[jj][30] = np.copy(A3_xy)
            self.tops_list_bs[jj][31] = np.copy(A4_xy)
            self.tops_list_bs[jj][32] = np.copy(A5_xy)
            self.tops_list_bs[jj][33] = np.copy(A6_xy)
            self.tops_list_bs[jj][34] = np.copy(A7_xy)
            self.tops_list_bs[jj][35] = np.copy(A8_xy)
            self.tops_list_bs[jj][36] = np.copy(A9_xy)
            self.tops_list_bs[jj][47] = A10_xy[:]


    def load_rock_prop_maps(self, riskmodel):
        ntops = len(self.tops_list_bs)
        lith_IDs = list(self.lith_dict.keys())
        # Fill arrays with rock properties    
        for jj in range(ntops):
            # Enter property information into maps
            for i in range(self.nx):
                for j in range(self.ny):
                    AOI_flag = self.AOI_np[i][j]
                    if AOI_flag == 1:
                        ID = int(self.tops_list_bs[jj][26][i][j])
                        if ID > 100:
                            ID = 1
                        if ID in lith_IDs:
                            prop_list = self.lith_dict[ID]
                        else:
                            desc = ("problem with lith ID " 
                                    + str(ID) 
                                    + ": top name : " 
                                    + str(self.tops_list_bs[jj][6]) 
                                    + " lithology set to first in list"
                                )
                            print_funcs.print_warning(desc)
                            ID = lith_IDs[0]
                        # 1: Matrix Velocity m/s	
                        # 2: Matrix HP: microW/m^3	
                        # 3: Grain k: W/m/K	
                        # 4: Grain Cp: J/kg/K	
                        # 5: Grain rho: kg/m^3	
                        # 6: Surface Porosity %	
                        # 7: Porosity Decay Depth (km)	
                        # 8: d (Gardner's law)	
                        # 9: f (Gardner's Law)
                        self.tops_list_bs[jj][28][i][j] = prop_list[1]
                        self.tops_list_bs[jj][29][i][j] = prop_list[2]
                        self.tops_list_bs[jj][30][i][j] = prop_list[3]
                        self.tops_list_bs[jj][31][i][j] = prop_list[4]
                        self.tops_list_bs[jj][32][i][j] = prop_list[5]
                        # Surface Porosity %
                        tmp_val = (prop_list[6] 
                                   + riskmodel.delta_spor_xy_np[i][j])
                        if tmp_val < 0:
                            tmp_val = 0
                        self.tops_list_bs[jj][33][i][j] = tmp_val               
                        # Porosity Decay Depth km
                        tmp_val = (prop_list[7] 
                                   + riskmodel.delta_pdecay_xy_np[i][j])
                        if tmp_val < 0:
                            tmp_val = 0
                        self.tops_list_bs[jj][34][i][j] = tmp_val                    
                        self.tops_list_bs[jj][35][i][j] = prop_list[8]
                        self.tops_list_bs[jj][36][i][j] = prop_list[9]                    
                        # OMT
                        self.tops_list_bs[jj][47][i][j][0] = prop_list[11]                    
                        # HI
                        HI_tmp = prop_list[12] + riskmodel.delta_HI_xy_np[i][j]
                        if HI_tmp < 0:
                            HI_tmp = 0
                        self.tops_list_bs[jj][47][i][j][1] = HI_tmp                    
                        # TOC
                        TOC_tmp = (prop_list[13] 
                                   + riskmodel.delta_toc_xy_np[i][j])
                        if TOC_tmp < 0:
                            TOC_tmp = 0
                        self.tops_list_bs[jj][47][i][j][2] = TOC_tmp                    
                        # kinetics: early, normal, late
                        if riskmodel.idelta_kinetics == 2:
                            final_kinetics = prop_list[14]
                        elif riskmodel.idelta_kinetics == 0:
                            final_kinetics = "normal"                    
                        elif riskmodel.idelta_kinetics == -1:
                            final_kinetics = "early"
                        elif riskmodel.idelta_kinetics == 1:
                            final_kinetics = "late"                    
                        self.tops_list_bs[jj][47][i][j][3] = final_kinetics                    
                        # source thickness, m
                        thick_tmp = (prop_list[15] 
                                     + riskmodel.delta_thick_xy_np[i][j])
                        if thick_tmp < 0:
                            thick_tmp = 0
                        self.tops_list_bs[jj][47][i][j][4] = thick_tmp                    
                        # Not used
                        self.tops_list_bs[jj][47][i][j][5] = prop_list[16]                    
                        # Oil API
                        oilapi_tmp = (prop_list[17] 
                                      + riskmodel.delta_oilapi_xy_np[i][j])
                        if oilapi_tmp < 15:
                            oilapi_tmp = 15
                        self.tops_list_bs[jj][47][i][j][6] = oilapi_tmp                    
                        # Gas gravity
                        gasgrav_tmp = (prop_list[18] 
                                       + riskmodel.delta_gasgrav_xy_np[i][j])
                        if gasgrav_tmp < 0.25:
                            gasgrav_tmp = 0.25
                        self.tops_list_bs[jj][47][i][j][7] = gasgrav_tmp                    
                        # Porosity threshold fac
                        porth_tmp = (prop_list[19] 
                                     + riskmodel.delta_porth_xy_np[i][j])
                        if porth_tmp < 0:
                            porth_tmp = 0
                        self.tops_list_bs[jj][47][i][j][8] = porth_tmp                    
                        # gas fraction scenario: minimum, base, maximum
                        if riskmodel.idelta_gas_frac == 2:
                            final_frac = prop_list[20]
                        elif riskmodel.idelta_gas_frac == 0:
                            final_frac = "base"
                        elif riskmodel.idelta_gas_frac == -1:
                            final_frac = "minimum"
                        elif riskmodel.idelta_gas_frac == 1:
                            final_frac = "maximum"                        
                        self.tops_list_bs[jj][47][i][j][9] = final_frac                    
                        # polar fraction scenario: minimum, base, maximum
                        if riskmodel.idelta_polar_frac == 2:
                            final_frac = prop_list[21]
                        elif riskmodel.idelta_polar_frac == 0:
                            final_frac = "base"
                        elif riskmodel.idelta_polar_frac == -1:
                            final_frac = "minimum"
                        elif riskmodel.idelta_polar_frac == 1:
                            final_frac = "maximum"                        
                        self.tops_list_bs[jj][47][i][j][10] = final_frac
                    

    def load_event_present_day_pwd(self):
        # define all event keys (integers from 0 to N) 
        # from oldest to youngest
        keys = list(self.event_dict_bs.keys())
        nevents_bs = len(keys) # number of backstripping events
        event_ID_list_bs = keys[:]
        # event ID of last event. This event ended at present day.
        event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
        
        file_name_PWD = self.event_dict_bs[event_ID_last_bs][11]
        # Read the bathy maps as the first ZMAP and 
        # initialize spatial parameters
        (
             pwd_xy, 
             dum_nx, dum_ny, dum_dx, dum_dy, 
             dum_xmin, dum_xmax, dum_ymin, dum_ymax, 
             dum_AOI_np_L
         ) = map_tools.read_ZMAP(self.input_path, file_name_PWD, self.AOI_np)
        self.event_dict_bs[event_ID_last_bs][5] = np.copy(pwd_xy)


    def load_event_pwd(self):
        """ Read in paleo-water depth and initialize base_level 
        """
        # define all event keys (integers from 0 to N) 
        # from oldest to youngest
        keys = list(self.event_dict_bs.keys())
        nevents_bs = len(keys) # number of backstripping events
        event_ID_list_bs = keys[:] # copy keys
        # event ID of last event. This event ended at present day.
        event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
        zeros_xy_np = np.zeros((self.nx, self.ny))
        self.PWD_overwrite_flag_dict = {}
        for k, key in enumerate(keys):
            ioverwrite = 0
            if key != event_ID_last_bs:
                file_name_PWD = self.event_dict_bs[key][11]
                if file_name_PWD != "NA":
                    try:
                        (
                            pwd_xy_new
                         ) = map_tools.read_ZMAP_interp_to_master(
                                    -99999.0, self.master_xy, self.input_path, 
                                    file_name_PWD, 
                                    self.nx, self.ny, self.xmin, self.ymin, 
                                    self.dx, self.dy, 
                                    self.AOI_np, zeros_xy_np
                                    )                    
                        # overwite if a fill is found    
                        ioverwrite = 1                
                    except:
                        ioverwrite = 0
                        pwd_xy_new = np.zeros((self.nx,self.ny))  
                else:
                    ioverwrite = 0
                    pwd_xy_new = np.zeros((self.nx,self.ny))
                self.event_dict_bs[key][5] = np.copy(pwd_xy_new)
            else:
                ioverwrite = 0
            self.PWD_overwrite_flag_dict[key] = ioverwrite


    def base_level_initialize(self):
        #**************************************
        # Now set base level using deltaSl list
        #**************************************
        keys = list(self.event_dict_bs.keys())
        nevents_bs = len(keys)
        event_ID_list_bs = keys[:]
        event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
        nx = self.nx
        ny = self.ny
        for k, key in enumerate(keys):
            self.event_dict_bs[key][6] = np.zeros((nx, ny))    
        for k, event_ID in enumerate(keys):
            if event_ID != event_ID_last_bs:
                base_level = self.deltaSL_list[event_ID]
                for i in range(nx):
                    for j in range(ny):
                        self.event_dict_bs[event_ID][6][i][j] = base_level


    def add_zero_maps(self, jj, key):
        nx = self.nx
        ny = self.ny
        self.tops_list_bs[jj][0].append(key)
        self.tops_list_bs[jj][14][key] = (len(self.tops_list_bs[jj][0])-1)    
        self.tops_list_bs[jj][1].append(np.zeros((nx,ny)))
        self.tops_list_bs[jj][2].append(np.zeros((nx,ny)))
        self.tops_list_bs[jj][3].append(np.zeros((nx,ny)))
        self.tops_list_bs[jj][13].append(np.zeros((nx,ny)))
        self.tops_list_bs[jj][17].append(np.zeros((nx,ny)))
        self.tops_list_bs[jj][46].append(np.zeros((nx,ny)))
        if self.icalc_temp == 1:
            self.tops_list_bs[jj][37].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][38].append(np.zeros((nx,ny)))                    
            self.tops_list_bs[jj][39].append(np.zeros((nx,ny)))                
        if self.icalc_LOM == 1:
            self.tops_list_bs[jj][40].append(np.zeros((nx,ny)))                
        if self.imass_gen > 0:                    
            self.tops_list_bs[jj][41].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][42].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][43].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][44].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][45].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][48].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][49].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][50].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][51].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][52].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][53].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][54].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][55].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][56].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][57].append(np.zeros((nx,ny)))                    
            self.tops_list_bs[jj][58].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][59].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][60].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][61].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][62].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][70].append(np.zeros((nx,ny)))
        #if self.itype3D == 0:
            self.tops_list_bs[jj][63].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][64].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][65].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][66].append(np.zeros((nx,ny)))    
            self.tops_list_bs[jj][67].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][68].append(np.zeros((nx,ny)))
            self.tops_list_bs[jj][69].append(np.zeros((nx,ny)))


    def top_maps_initialize(self):
        keys = list(self.event_dict_bs.keys())                  
        #*******************************
        # Initialize event maps for tops
        #*******************************
        # Looping over events from oldest to youngest
        for k, key in enumerate(keys):
            itop = self.event_dict_bs[key][2]
            ntops = len(self.tops_list_bs)
            for jj in range(ntops):
                # Define maps for each event when this 
                # top was present
                if jj <= itop:
                    self.add_zero_maps(jj, key)


    def initialize_total_final_erosion_and_total_max_burial(self):               
        ntops = len(self.tops_list_bs)
        nx = self.nx
        ny = self.ny
        for kk in range(ntops):
            # Initialize total final erosion map for top
            e_tot_xy = np.zeros((nx,ny))            
            self.tops_list_bs[kk][8].append(e_tot_xy)        
            # Initialize total final maximum burial depth map
            maxf_xy = np.zeros((nx,ny))
            self.tops_list_bs[kk][12] = maxf_xy


    def load_depth_maps(self):
        """ Load depth maps
        """
        # define all event keys (integers from 0 to N) from oldest to youngest
        keys = list(self.event_dict_bs.keys())
        nevents_bs = len(keys)
        event_ID_list_bs = keys[:]
        # event ID of last event. This event ended at present day.
        # Only working on last event (i.e. present-day)
        event_ID = event_ID_list_bs[nevents_bs - 1]
        zeros_xy_np = np.zeros((self.nx, self.ny))
        ntops = len(self.tops_list_bs)
        for kk in range(ntops):
            # Name of top file (TVD, m)
            map_file_name = self.tops_list_bs[kk][15]
            depth_xy = map_tools.read_ZMAP_interp_to_master(
                                    -99999.0, self.master_xy, 
                                    self.input_path, map_file_name, 
                                    self.nx, self.ny, self.xmin, self.ymin, 
                                    self.dx, self.dy, self.AOI_np, zeros_xy_np
                                    )
            # list index for eventID
            event_index = self.tops_list_bs[kk][14][event_ID]
            # Note that lists were already initialized
            self.tops_list_bs[kk][1][event_index] = depth_xy


    def calc_thickness_and_clean_hor(self):
        """ Calculate thickness and clean all layers
        """
        # define all event keys (integers from 0 to N) 
        # from oldest to youngest
        keys = list(self.event_dict_bs.keys())
        nevents_bs = len(keys)
        event_ID_list_bs = keys[:]
        # event ID of last event. This event ended at present day.
        # Only working on last event (i.e. present-day)
        event_ID = event_ID_list_bs[nevents_bs-1]
        # Looping from youngest to oldest
        ntops = len(self.tops_list_bs)
        for jj in range(ntops):
            # tops are list from oldest to youngest, 
            # we need to start with the 
            # youngest and work our way down
            kk = ntops - jj - 1
            evindex = self.tops_list_bs[kk][14][event_ID]
            z_this_top_xy =  self.tops_list_bs[kk][1][evindex]
            thick_xy = self.tops_list_bs[kk][2][evindex]
            if kk > 0:
                nelem = len(self.tops_list_bs[kk-1][0])
                evindex2 = nelem - 1
                z_older_top_xy = self.tops_list_bs[kk-1][1][evindex2]
                deltadepth = 0.0
            else:
                z_older_top_xy = self.tops_list_bs[kk][1][evindex]
                deltadepth = self.tops_list_bs[kk][5]
            calc_thickness_and_clean_hor_loop(
                                            self.nx, self.ny, z_this_top_xy, 
                                            z_older_top_xy, thick_xy, 
                                            deltadepth
                                            )
        

    def initialize_event_arrays(self):
        nx = self.nx
        ny = self.ny
        keys = list(self.event_dict_bs.keys())
        for k, key in enumerate(keys): 
            # Looping over events from oldest to youngest
            self.event_dict_bs[key][8] = np.zeros((nx,ny))
            self.event_dict_bs[key][4] = np.zeros((nx,ny))
            self.event_dict_bs[key][12] = np.zeros((nx,ny))
            self.event_dict_bs[key][13] = np.zeros((nx,ny)) 
            self.event_dict_bs[key][15] = np.zeros((nx,ny)) 
            self.event_dict_bs[key][16] = np.ones((nx,ny))*-99999.0
            self.event_dict_bs[key][17] = np.ones((nx,ny))*-99999.0
            self.event_dict_bs[key][18] = np.zeros((nx,ny))
            self.event_dict_bs[key][19] = np.ones((nx,ny))*-99999.0
            self.event_dict_bs[key][20] = np.ones((nx,ny))*-99999.0
            self.event_dict_bs[key][21] = np.zeros((nx,ny))
            self.event_dict_bs[key][22] = np.zeros((nx,ny))
            self.event_dict_bs[key][23] = np.zeros((nx,ny))
            self.event_dict_bs[key][25] = np.zeros((nx,ny))
            self.event_dict_bs[key][26] = np.zeros((nx,ny))


    def load_shift_maps(self):
        """ Load in shift maps for all events
        """
        nx = self.nx
        ny = self.ny
        zeros_xy_np = np.zeros((nx, ny))
        keys = self.event_dict_bs.keys()
        for key in keys:
            file_name_shift = self.event_dict_bs[key][10]
            if file_name_shift != "NA":            
                ifind_file = 1
                try:
                    shift_xy = map_tools.read_ZMAP_interp_to_master(
                                    -99999.0, self.master_xy, self.input_path, 
                                    file_name_shift, nx, ny, 
                                    self.xmin, self.ymin, self.dx, self.dy,
                                    self.AOI_np, zeros_xy_np
                                    )
                except:                
                    ifind_file = 0                
                if ifind_file == 1:                
                    self.event_dict_bs[key][4] = shift_xy      


    def load_salt_maps(self):
        """ Load salt thickness maps for all events
        """
        nx = self.nx
        ny = self.ny
        zeros_xy_np = np.zeros((nx, ny))
        keys = self.event_dict_bs.keys()
        for key in keys:
            file_name_salt = self.event_dict_bs[key][24]
            if file_name_salt != "NA":
                ifind_file = 1
                try:
                    salt_xy = map_tools.read_ZMAP_interp_to_master(
                                    -99999.0, self.master_xy, self.input_path, 
                                    file_name_salt, nx, ny, 
                                    self.xmin, self.ymin, self.dx, self.dy, 
                                    self.AOI_np, zeros_xy_np
                                    )
                except:
                    ifind_file = 0
                if ifind_file == 1:
                    self.event_dict_bs[key][23] = salt_xy
                    
                else:
                    self.event_dict_bs[key][23] = np.zeros((nx, ny))


    def load_heat_flow_maps(self, riskmodel):       
        self.bghf_xy = map_tools.read_ZMAP_interp_to_master(
                                -99999.0, self.master_xy, self.input_path,
                                self.background_hf_map_file, self.nx, self.ny, 
                                self.xmin, self.ymin, self.dx, self.dy,
                                self.AOI_np, riskmodel.delta_bghf_xy_np
                                )
        self.hf_reduc_fac_xy = map_tools.read_ZMAP_interp_to_master(
                                -99999.0, self.master_xy, self.input_path,
                                self.hf_reduc_fac_map_file, self.nx, self.ny, 
                                self.xmin, self.ymin, self.dx, self.dy, 
                                self.AOI_np, riskmodel.delta_hfred_xy_np
                                )


    def load_rifting_maps(self, riskmodel):
        master_xy = self.master_xy
        input_path = self.input_path
        nx = self.nx
        ny = self.ny
        dx = self.dx
        dy = self.dy
        AOI_np = self.AOI_np
        xmin = self.xmin
        ymin = self.ymin
        zeros_xy_np = self.zeros_xy_np
        
        self.start_age1_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.start_age_rift1_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_defage_xy_np
                                            )    
        self.end_age1_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.end_age_rift1_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_defage_xy_np
                                            )    
        try:
            self.start_age2_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.start_age_rift2_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_defage_xy_np
                                            ) 
        except:
            self.start_age2_xy = np.copy(self.start_age1_xy)
        try:
            self.end_age2_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.end_age_rift2_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_defage_xy_np
                                            )
        except:
            self.end_age2_xy = np.copy(self.end_age1_xy)   
        try:
            self.start_age3_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.start_age_rift3_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_defage_xy_np
                                            )
        except:
            self.start_age3_xy = np.copy(self.start_age1_xy)
        try:
            self.end_age3_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.end_age_rift3_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_defage_xy_np
                                            )
        except:
            self.end_age3_xy = np.copy(self.end_age1_xy)
        self.rift_mag1_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.riftmag_rift1_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            zeros_xy_np
                                            )
        
        try:
            self.rift_mag2_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.riftmag_rift2_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            zeros_xy_np
                                            )
        except:
            self.rift_mag2_xy = np.copy(self.rift_mag1_xy)
        try:
            self.rift_mag3_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.riftmag_rift3_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            zeros_xy_np
                                            )
        except:
            self.rift_mag3_xy = np.copy(self.rift_mag1_xy)
        self.mantle_fac1_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.mantlefac_rift1_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_mantlefac_xy_np
                                            )
        
        try:
            self.mantle_fac2_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.mantlefac_rift2_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_mantlefac_xy_np
                                            )
        except:
            self.mantle_fac2_xy = np.copy(self.mantle_fac1_xy)
        
        try:
            self.mantle_fac3_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.mantlefac_rift3_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            riskmodel.delta_mantlefac_xy_np
                                            )
        except:
            self.mantle_fac3_xy = np.copy(self.mantle_fac1_xy)
        self.nphases_xy = map_tools.read_ZMAP_interp_to_master(
                                                -99999.0, master_xy, 
                                                input_path, self.nphases_name, 
                                                nx, ny, xmin, ymin, 
                                                dx, dy, AOI_np, 
                                                zeros_xy_np
                                                )
        for ii in range(nx):
            for jj in range(ny):
                self.nphases_xy[ii][jj] = int(self.nphases_xy[ii][jj])
        self.age_s_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.start_age_rift1_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            zeros_xy_np
                                            )
        self.age_e_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            self.end_age_rift1_name, 
                                            nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np, 
                                            zeros_xy_np
                                            )  
    

    def load_pd_pwd(self):
        # define all event keys (integers from 0 to N) 
        # from oldest to youngest
        keys = list(self.event_dict_bs.keys())
        nevents_bs = len(keys) # number of backstripping events
        event_ID_list_bs = keys[:]
        # event ID of last event. This event ended at present day.
        event_ID_last_bs = event_ID_list_bs[nevents_bs - 1]
        AOI_np_dum = np.zeros((1,1))
        file_name_PWD = self.event_dict_bs[event_ID_last_bs][11]
        (
            self.pwd_xyz_pd, self.tnx, self.tny, self.tdx, self.tdy,
            self.txmin, self.txmax, self.tymin, self.tymax, 
            self.AOI_np_L
        ) = map_tools.read_ZMAP(self.input_path, file_name_PWD, AOI_np_dum)

        
    def copy_csv_files_to_output(self):
        input_path = self.input_path
        output_path = self.output_path
        # Copy csv input files to output directory
        f1 = "ipa_input.csv"
        simple_file_copy(input_path, output_path, f1)
        f1 = "ipa_calibration.csv"
        simple_file_copy(input_path, output_path, f1)
        f1 = "ipa_lithology.csv"
        simple_file_copy(input_path, output_path, f1)
        f1 = "ipa_loaded_maps.csv"
        simple_file_copy(input_path, output_path, f1)
        f1 = "ipa_wells.csv"
        simple_file_copy(input_path, output_path, f1)
        f1 = "ipa_traps.csv"
        try:
            simple_file_copy(input_path, output_path, f1)    
        except:
            print("Could not copy ", f1)

                   
    def initialize_model(self, riskmodel, ioutput_main, process):
        ttt1 = time.time()
        
        tt1 = time.time()
        self.make_ipa_data_structures()
        tt2 = time.time()
        print_funcs.print_finfo(ioutput_main, process, 
                                "Created data structures", tt2-tt1)
        
        tt1 = time.time()
        self.make_file_prefix_and_suffix()
        self.make_output_directories(ioutput_main)
        tt2 = time.time()
        print_funcs.print_finfo(
                    ioutput_main, process, 
                    "Created output directories and file name lists", tt2-tt1)
        
        tt1 = time.time()
        self.load_SurfTemp()
        self.load_APWP()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded SWIT and APWP info", tt2-tt1)
        
        keys = list(self.event_dict_bs.keys())
        self.nevents_bs = len(keys)
        event_ID_list_bs = keys[:]
        self.event_ID_last_bs = event_ID_list_bs[self.nevents_bs - 1]
        
        tt1 = time.time()
        self.create_master_grid_from_water_bottom()
        tt2 = time.time()
        print_funcs.print_finfo(
                        ioutput_main, process, "Loaded master grid", tt2-tt1)
        
        print_funcs.print_info(
                    "Area per cell (km2) for master grid",[self.cell_area_km2])
        print_funcs.print_info("Number of cells in master grid",
                               [self.nx*self.ny])
        print_funcs.print_info("Master grid dimensions: nx, ny, dx, dy",
                               [self.nx, self.ny, self.dx, self.dy])
        
        
        tt1 = time.time()
        input_file_path = os.path.join(self.input_path, "ipa_lithology.csv")
        self.lith_dict = fileIO.read_lithology_file_csv(input_file_path)
        tt2 = time.time()
        print_funcs.print_finfo(
                        ioutput_main, process, "Read lithology csv", tt2-tt1)
        
        tt1 = time.time()
        self.initialize_lithology_arrays()
        tt2 = time.time()
        print_funcs.print_finfo(
                ioutput_main, process, "Initialized lithology arrays", tt2-tt1)
        
        tt1 = time.time()
        self.load_lithology_maps()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded lithology maps", tt2-tt1)
        
        tt1 = time.time()
        self.load_rock_prop_maps(riskmodel)
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded rock property maps", tt2-tt1)
        
        if self.idepth_model == 1 or self.idepth_model  == 2:
            if self.idepth_model == 1:
                # just use user defined interval velocity
                self.iuse_comp_law = 1
            else:
                # adjust using compaction law
                self.iuse_comp_law = 0
            tt1 = time.time()
            # Read TWT surfaces for tops and create 
            # depth (TVD) files using
            # user defined interval velocity
            time2depth.load_TWT_maps_and_calc_depth_map(
                        self.input_path, self.output_path, self.tops_list_bs, 
                        self.event_dict_bs, self.rho_water, self.iuse_comp_law, 
                        self.niter_t2d_comp, self.AOI_np, 
                        self.master_xy, self.nx, self.ny, self.dx, self.dy, 
                        self.xmin, self.xmax, self.ymin, self.ymax
                        )
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Converted from time to depth", tt2-tt1)
        tt1 = time.time()
        self.load_event_present_day_pwd()
        self.load_event_pwd()
        self.base_level_initialize()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Initialized pwd and baselevel maps", tt2-tt1)    
        
        tt1 = time.time()
        self.top_maps_initialize()
        print_funcs.print_finfo(
                        ioutput_main, process, 
                        "Initialized all maps for tops", tt2-tt1)  
        
        tt1 = time.time()
        self.initialize_total_final_erosion_and_total_max_burial()
        tt2 = time.time()
        print_funcs.print_finfo(
                        ioutput_main, process, 
                        "Initialized tot. final erosion and tot. max burial", 
                        tt2-tt1)
        
        tt1 = time.time()
        self.load_depth_maps()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded depth maps", tt2-tt1)
        
        tt1 = time.time()
        self.calc_thickness_and_clean_hor()
        tt2 = time.time()
        print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Calculated thickness and cleaned maps", tt2-tt1)
        
        tt1 = time.time()
        self.initialize_event_arrays()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Initialized event arrays", tt2-tt1)
        
        tt1 = time.time()
        self.load_shift_maps()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded shift maps", tt2-tt1)
        
        tt1 = time.time()
        self.load_salt_maps()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded salt maps", tt2-tt1)
        
        tt1 = time.time()
        erosion_funcs.calc_total_erosion(
                            self.event_dict_bs, self.tops_list_bs, 
                            self.nx, self.ny)
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Calculated total erosion", tt2-tt1)
        
        tt1 = time.time()
        erosion_funcs.calc_erosion_for_tops(
                                        self.event_dict_bs, self.tops_list_bs, 
                                        self.nx, self.ny
                                        )
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Calculated erosion for tops", tt2-tt1)
        
        tt1 = time.time()
        erosion_funcs.max_burial_initialize(self.event_dict_bs, 
                                            self.tops_list_bs, 
                                            self.nx, self.ny)
        tt2 = time.time()
        print_funcs.print_finfo(
                        ioutput_main, process, 
                        "Initialized maximum burial", tt2-tt1)
        
        if self.idepth_model == 0:
            tt1 = time.time()
            self.iuse_comp_law = 0        
            time2depth.calc_VELOC_TWT_maps(
                            self.input_path, self.output_path, 
                            self.tops_list_bs, self.event_dict_bs, 
                            self.rho_water, self.iuse_comp_law, self.AOI_np, 
                            self.master_xy, self.nx, self.ny, self.dx, self.dy, 
                            self.xmin, self.xmax, self.ymin, self.ymax
                            )
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Calculated velocity and TWT", tt2-tt1)
        tt1 = time.time()
        self.load_heat_flow_maps(riskmodel)
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded BGHF and HF reduction maps", tt2-tt1)
        
        tt1 = time.time()
        self.load_rifting_maps(riskmodel)
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded rifting maps", tt2-tt1)
        
        tt1 = time.time()
        self.load_pd_pwd()
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Loaded PWD present-day map", tt2-tt1)
        
        tt1 = time.time()
        file_name='AIO.dat'
        map_tools.make_output_file_ZMAP_v4(
                                    self.output_path, file_name, self.AOI_np,
                                    self.nx, self.ny, self.dx, self.dy, 
                                    self.xmin, self.xmax, self.ymin, self.ymax, 
                                    self.AOI_np
                                    )
        tt2 = time.time()
        print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Created AOI.dat file", tt2-tt1)
        
        tt1 = time.time()
        self.copy_csv_files_to_output()
        tt2 = time.time()
        print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Copied input files to output directory", tt2-tt1)
        
        ttt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Completed model initialization", ttt2-ttt1)
    

    def load_crustal_thick(self, xth_xy, riskmodel, ioutput_main, process):
        if self.inv_itype < 0 and self.itype3D == 1:            
            tt1 = time.time()            
            try:
                xth_xy = map_tools.read_ZMAP_interp_to_master(
                                        -99999.0, self.master_xy, 
                                        self.input_path, self.xth_file_name, 
                                        self.nx, self.ny, self.xmin, self.ymin, 
                                        self.dx, self.dy, 
                                        self.AOI_np, riskmodel.delta_xth_xy_np
                                        )
            except:                
                print("!!! Crustal thickness map was not found defaulting to "
                      "10 km crust")                
                xth_xy = np.ones((self.nx, self.ny))*10000
            tt2 = time.time()
            print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Finished loading crustal thickness", tt2-tt1)
        return xth_xy

    
    def move_maps(self, ioutput_main):
        if self.itype3D == 1:
            # Move output files to directories
            for kk, odirec in enumerate(self.output_direcs_list_A):
                target_direc = os.path.join(self.output_path, odirec)
                pf = self.prefix_all[kk]
                file_path_list = glob.glob(
                                            os.path.join(self.output_path, 
                                                                      pf + "*")
                                          )
                for file_path1 in file_path_list:  
                    file_name = os.path.basename(file_path1)
                    #file_path2 = target_direc + "\\" + file_name
                    sflags = [
                              "Initial", 
                              "Updated", 
                              "GOR_primary_g_g",
                              "GOR_primary_scf_bbl", 
                              "GOR_total_g_g",
                              "GOR_total_scf_bbl", 
                              "cYield_PDG_Tg",
                              "cYield_PFG_Tg", 
                              "cYield_PO_Tg", 
                              "cYield_SFG_Tg",
                              "iYield_PDG_Tg", 
                              "iYield_PFG_Tg", 
                              "iYield_PO_Tg",
                              "iYield_SFG_Tg", 
                              "HF_ANOM", 
                              "HF_TOT", 
                              "PWD_LocalIsostasy"
                              ]
                    sflag = ""
                    for sflag_check in sflags:
                        if sflag_check in file_name:
                            sflag = sflag_check
                    dir_path = os.path.join(target_direc, sflag)
                    if os.path.isdir(dir_path) != True:
                        os.mkdir(dir_path)
                    file_path2 = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path2) == True:
                        try:
                            os.remove(file_path2)
                        except:
                            if ioutput_main == 1: 
                                print("Unable to delete old file: ", 
                                      file_path2)
                    if os.path.isfile(file_path1) == True:
                        try:
                            os.rename(file_path1, file_path2)
                        except:
                            if ioutput_main == 1: 
                                print("Unable to move file: prefix, "
                                      "file_name, file_path1 : ", 
                                      pf, file_name, file_path1)
                                    
            for kk, odirec in enumerate(self.output_direcs_list_B):
                target_direc = os.path.join(self.output_path, odirec)
                suf = self.suffix_all[kk]
                file_path_list = glob.glob(os.path.join(self.output_path, 
                                                        "*" + suf))
                for file_path1 in file_path_list:  
                    file_name = os.path.basename(file_path1)
                    file_path2 = os.path.join(target_direc, file_name)
                    if os.path.isfile(file_path2) == True:
                        try:
                            os.remove(file_path2)
                        except:
                            if ioutput_main == 1: 
                                print("Unable to delete old file : ", 
                                      file_path2)
                    if os.path.isfile(file_path1) == True:
                        try:
                            os.rename(file_path1, file_path2)        
                        except:
                            if ioutput_main == 1: 
                                print("Unable to move file: suffix, "
                                      "file_name, file_path1 : ", suf, 
                                      file_name, file_path1)
            



