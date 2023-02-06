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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import map_tools
import fileIO


def get_geo_color_layer_count(
                                age, icount_cenozoic, icount_cretaceous, 
                                icount_jurassic, icount_triassic, 
                                icount_paleozoic
                            ):
    if age <= 66.0:
        icount_cenozoic = icount_cenozoic + 1
        if (icount_cenozoic % 2) == 0:
            col = "gold"
        else:
            col = "sandybrown"
    elif 66.0 < age <= 145.0:
        icount_cretaceous = icount_cretaceous + 1
        if (icount_cretaceous % 2) == 0:
            col = "palegreen"
        else:
            col = "green"
    elif 145 < age <= 201.0:
        icount_jurassic = icount_jurassic + 1
        if (icount_jurassic % 2) == 0:
            col = "lightblue"
        else:
            col = "blue"    
    elif 201.0 < age <= 252.0:
        icount_triassic = icount_triassic + 1
        if (icount_triassic % 2) == 0:
            col = "purple"
        else:
            col = "plum" 
    else:
        icount_paleozoic = icount_paleozoic + 1
        if (icount_paleozoic% 2) == 0:
            col = "lightgray"
        else:
            col = "gray"
    return (
            col, icount_cenozoic, icount_cretaceous, 
            icount_jurassic, icount_triassic, 
            icount_paleozoic
        )


def initialize_inputs_for_xs_plot(main_output_path):
    
    input_file_path = main_output_path + "ipa_input.csv"
    strat_input_dict = {} 
    param_input_dict = {}
    fileIO.read_main_input_file_csv(
                                input_file_path, 
                                strat_input_dict, 
                                param_input_dict
                                )
    
    isalt_restore = param_input_dict["isalt_restore"]
    salt_layer_index = param_input_dict["salt_layer_index"]
    keys = list(strat_input_dict.keys())
    ntops = len(keys)
    salt_dict_key_index = ntops - salt_layer_index
    if isalt_restore == 1:
        salt_key = keys[salt_dict_key_index]
        salt_name = strat_input_dict[salt_key][1]
    else:
        salt_name = "None"
    tops_IDS = []
    for i in range(ntops):
        tops_IDS.append(i)
    tops_ages = []
    tops_names = []
    tops_colors = []
    icount_cenozoic = 0
    icount_cretaceous = 0
    icount_jurassic = 0
    icount_triassic = 0
    icount_paleozoic = 0
    icount_source = 0
    for key in keys:
        age = strat_input_dict[key][0]
        name = strat_input_dict[key][1]
        pstype = strat_input_dict[key][10]
        tops_ages.append(age)
        tops_names.append(strat_input_dict[key][1])
        (
         col, icount_cenozoic, icount_cretaceous, 
         icount_jurassic, icount_triassic, 
         icount_paleozoic
         ) = get_geo_color_layer_count(
                                        age, icount_cenozoic, icount_cretaceous, 
                                        icount_jurassic, icount_triassic, 
                                        icount_paleozoic
                                    )
        if pstype == "Source":
            col = "black"
            icount_source = icount_source + 1
            if (icount_source % 2) == 0:
                col = "black"
            else:
                col = "darkgray" 
        if isalt_restore == 1 and name == salt_name:
            col = "pink"
        tops_colors.append(col)
    return (
            ntops, keys,
            tops_ages, tops_names, tops_colors,
            strat_input_dict,
            isalt_restore, salt_name
        )


def define_xs(x1, x2, y1, y2, dL):
    # Define cross section line
    DDx = x2 - x1
    DDy = y2 - y1
    Ltot = math.sqrt(DDx*DDx + DDy*DDy)
    n = int(Ltot/dL)+1
    print("Ltot, dL, n : ", Ltot, dL, n)
    dxx = DDx/float(n)
    x_line = []
    y_line = []
    dist_line = []
    for i in range(n):
        x = x1 + float(i)*dxx 
        y = y1 + DDy/DDx*(x-x1)
        x_line.append(x)
        y_line.append(y)
        dist = math.sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1))
        dist_line.append(dist)
    return n, x_line, y_line, dist_line


def plot_xs_main(zmin_xs, zmax_xs, fig_x_inches, ve_fac, 
                 x1, x2, y1, y2, main_output_path
):  
    xs_length_m = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    output_path = main_output_path
    depth_dir_path = main_output_path + "Depth_ZMaps\\Updated\\"
    (
         ntops, keys,
         tops_ages, tops_names, tops_colors,
         strat_input_dict,
         isalt_restore, salt_name
    ) = initialize_inputs_for_xs_plot(main_output_path)
    nevents = ntops
    np1 = "DEPTH_Updated_EV_"
    #np1 = "DEPTH_Initial_EV_"
    AOI_np_ini = np.zeros((1,1))
    event_lines = {}
    icount_tops = 0
    for i in range(nevents): 
        # Loop over all events from oldest to youngest
        age = tops_ages[i]
        print("Working on event ", i, " with age ", age)
        print("Reading depth maps")
        # Gather all top maps for event
        top_maps = []
        for j in range(ntops): 
            # Loop over all tops associated with event    
            if j <= i: 
                # only consider tops that are older or equal in age to event
                tname = tops_names[j]
                file_name = (
                                  np1+str(i)+"_"
                                + "t"+str(j)
                                + "_AGE_"+str(round(age,2))
                                + "_TOP_"+tname+".dat"
                            )
                (
                    scalars_xy, nx, ny, 
                    dx, dy, xmin, xmax, ymin, ymax, AOI_np
                 ) = map_tools.read_ZMAP(depth_dir_path, file_name, AOI_np_ini)
                top_maps.append(scalars_xy)
                if icount_tops == 0:
                    # define geometric parameters for cross section
                    print("xmin, xmax, ymin, ymax : ", xmin, xmax, ymin, ymax)
                    dL = (dx + dy)/2.0
                    n, x_line, y_line, dist_line = define_xs(
                                                            x1, x2, y1, y2, dL)
                    #print("x_line : ", x_line)
                    #print("y_line : ", y_line)
                    #print("dist_line : ", dist_line)
                    profile_file_name = depth_dir_path + "POLY_profile.poly"
                    fout = open(profile_file_name, 'w')
                    str_out = str(0) + " " + str(x1) + " " + str(y1) + "\n"
                    fout.write(str_out)
                    str_out = str(0) + " " + str(x2) + " " + str(y2) + "\n"
                    fout.write(str_out)
                    fout.close()                    
                icount_tops = icount_tops + 1
        print("Interpolating along profile")
        naflag = -99999.0
        z_lines_all = []
        for mm, top_xy in enumerate(top_maps):
            top_xy_np = np.asarray(top_xy)
            scalars = []
            for ii in range(n):            
                xL = x_line[ii]
                yL = y_line[ii]            
                s_interp = map_tools.zmap_interp(
                                                    xL, yL, naflag, top_xy_np, 
                                                    nx, ny, dx, dy, 
                                                    xmin, xmax, ymin, ymax
                                                )
#                if s_interp < 0:
#                    s_interp = 0
                scalars.append(s_interp)
            z_lines_all.append(scalars)    
        event_lines[i] = z_lines_all[:]
    for i in range(nevents):
        age = tops_ages[i]    
        fdpi = 150
        fig, ax = plt.subplots()
        zlines = event_lines[i]
        for ii, zline in enumerate(zlines):
            ax.plot(dist_line, zline, color='black', 
                    linestyle='-',linewidth=0.5)    
            if ii > 0:            
                poly_pts = []            
                for mm, z  in enumerate(zline):
                    xx = dist_line[mm]
                    poly_pts.append([xx, z])
                imax = len(zlines[ii-1])-1
                for mm, z  in enumerate(zlines[ii-1]):
                    bb = imax - mm 
                    xx = dist_line[bb]
                    z = zlines[ii-1][bb]
                    poly_pts.append([xx, z])
                ax.add_patch(
                                Polygon((poly_pts), closed=True, 
                                fill=True, color=tops_colors[ii])
                            )    
        fig = plt.gcf()
        ddz = zmax_xs - zmin_xs
        #ddx = math.sqrt(delta_x*2*delta_x*2 + delta_y*2*delta_y*2)
        ddx = xs_length_m
        fig_z_inches = fig_x_inches*ddz/ddx*ve_fac  
        fig.set_size_inches(fig_x_inches, fig_z_inches )    
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_ylim( (zmin_xs, zmax_xs))
        ax.set_xlim( (0.0, ddx))
        title_str = (   str(round(age,1)) + " Ma : "
                     +  "x1: " + str(round(x1,2)) + "m"
                     + " y1: " + str(round(y1,2)) + "m"
                     + " x2: " + str(round(x2,2)) + "m"
                     + " y2: " + str(round(y2,2)) + "m"
                    )
        ax.title.set_text(title_str)
        ax.invert_yaxis()    
        filename = depth_dir_path + "profile_"+str(round(age, 2))+"Ma.jpg"
        plt.savefig(filename, dpi=fdpi)

