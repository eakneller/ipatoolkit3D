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
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Polygon
import map_tools
import fileIO


def plot_polys(ax, input_path, poly_plot_dict):
    poly_plot_keys = list(poly_plot_dict.keys())
    for plot_key in poly_plot_keys:
        poly_file_name = poly_plot_dict[plot_key][0]
        size = poly_plot_dict[plot_key][1]
        scolor = poly_plot_dict[plot_key][2]
        poly_file_path = input_path + "\\" + poly_file_name
        if os.path.isfile(poly_file_path) == True:
            poly_dict = fileIO.read_polys_open_vs_closed(poly_file_path)
            keys = list(poly_dict.keys())            
            for key in keys:
                tmp_poly = []
                ncoors = len(poly_dict[key])
                bclosed = True
                for i in range(ncoors):
                    x = poly_dict[key][i][0]/1000
                    y = poly_dict[key][i][1]/1000
                    tmp_poly.append([x, y])
                    if i == ncoors - 1:
                        if poly_dict[key][i][2] == 1:
                            bclosed = False
                ax.add_patch(
                                Polygon(
                                    (tmp_poly), closed=bclosed, fill=False,
                                    color=scolor, linewidth=size
                                    )
                                )

    
def make_color_map_maturity():
    # Oil	0.58
    # Late Oil	0.85
    # Wet Gas	1.25
    # Gas	1.80
    # Super Mature	2.50    
    cmap = matplotlib.colors.ListedColormap(
                [
                 "white", # 1
                 'blue', # 2
                 'darkgreen', # 3
                 'green', # 4
                 'lime', # 5
                 'greenyellow', # 6
                 'yellow', # 7
                 'red', # 8
                 'firebrick', # 9
                ]
                )
    boundaries = [
                  0, # 1
                  0.001, # 2
                  0.65, # 3
                  0.72, # 4
                  0.85, # 5
                  1.03, # 6
                  1.27, # 7
                  2.57, # 8
                  3.61, # 9
                  5.0 # 10
                 ]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm, boundaries

    
def plot_2D_grid(
                    gridx, gridy, gs,
                    xticks, yticks, 
                    V, V2, linewidths_list, 
                    base_name, outpath, 
                    nwells, wells_dict, 
                    poly_plot_dict, plot_dict
):
    # Python font size strings:
    # xx-small, x-small, small, medium, 
    # large, x-large, xx-large, 
    # larger, smaller
    fac1 = 1.0
    fac2 = 1.0 
    fac3 = 1.0
    gs_min = plot_dict["minimum_value"]
    # maximum value
    gs_max = plot_dict["maximum_value"]
    # minimum x-value
    dxmin = plot_dict["xmin_m"]/1000
    # max. x-value
    dxmax = plot_dict["xmax_m"]/1000
    # min. y-value
    dymin = plot_dict["ymin_m"]/1000
    # max. y-value
    dymax = plot_dict["ymax_m"]/1000
    sdata_type = plot_dict["data_type"]
    fs = plot_dict["font_size"]
    fc = plot_dict["font_color"]
    ps = plot_dict["point_size"]
    # color map option string
    color_map = plot_dict["color_map"]
    # 0 = off; 1 = plot contour
    iplot_c = plot_dict["iplot_contour"] 
    # 0 = off; 1 = plot contour labels
    iplot_clab = plot_dict["iplot_clab"] 
    # gouraud = no mesh; flat = plot mesh
    shading_str = plot_dict["shading_str"]
    # dots per inch for figure
    fdpi = plot_dict["fdpi"]
    # fraction of original axes to use for colorbar
    fraction_val = plot_dict["fraction_val"]
    # fraction of original axes between colorbar and new image axes
    pad_val = plot_dict["pad_val"]
    
    plt.ioff()
    cm, norm, boundaries_Ro = make_color_map_maturity()
    xlabel_shift = 2.5 # 150.0/1000*4
    fig, ax = plt.subplots()
    ax.set_xlim(dxmin, dxmax)
    ax.set_ylim(dymin, dymax)
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.xaxis.label.set_size('x-small')
    ax.yaxis.label.set_size('x-small')
    title_str = base_name
    ax.title.set_text(title_str)
    ax.title.set_size('small')
    xx_tmp = []
    yy_tmp = []
    wkeys = list(wells_dict.keys())
    all_txt = []
    for ii, key in enumerate(wkeys):
        xxx = wells_dict[key][0]/1000
        yyy = wells_dict[key][1]/1000
        if xxx >= dxmin and xxx <= dxmax and yyy >= dymin and yyy <= dymax:
            well_name = key
            all_txt.append(well_name)
            xx_tmp.append(xxx)
            yy_tmp.append(yyy)
    nwells_update = len(all_txt)
    xx = np.zeros((nwells_update))
    yy = np.zeros((nwells_update))
    for ii, txt in enumerate(all_txt):
            xx[ii] = xx_tmp[ii]
            yy[ii] = yy_tmp[ii]
    if sdata_type == "Ro":
        nbound = len(boundaries_Ro)
        vmin_Ro = boundaries_Ro[0]
        vmax_Ro = boundaries_Ro[nbound-1] 
        p1 = ax.pcolormesh(
                            gridx*fac1, gridy*fac2, gs*fac3, 
                            shading=shading_str, edgecolors='gray', 
                            linewidth=0.01, cmap=cm,
                            vmin=vmin_Ro, vmax=vmax_Ro, norm=norm
                            )        
    else:
        p1 = ax.pcolormesh(
                            gridx*fac1, gridy*fac2, gs*fac3, 
                            shading=shading_str, edgecolors='gray', 
                            linewidth=0.01, cmap=color_map,
                            vmin=gs_min, vmax=gs_max
                            )
    p1.cmap.set_under('w')
    if iplot_c == 1:
        p2 = ax.contour(
                        gridx*fac1, gridy*fac2, gs*fac3, 
                        V, colors='black', linewidths=linewidths_list
                        )
        if iplot_clab == 1:
            plt.clabel(p2, inline=1, fontsize=5, fmt='%6.1f')    
    #ax.invert_yaxis()
    ax.set_aspect(1.0)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize('xx-small')
        #tick.set_fontname('Times New Roman')
        #tick.set_color('blue')
        #tick.set_weight('bold')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize('xx-small')
        #tick.set_fontname('Times New Roman')
        #tick.set_color('blue')
        #tick.set_weight('bold')
    # fraction 0.15; fraction of original axes to use for colorbar
    # pad 0.05 if vertical, 0.15 if horizontal; fraction of original axes 
    # between colorbar and new image axes
    cbar = plt.colorbar(
                        p1, ticks=V2, orientation='vertical', 
                        fraction=fraction_val, pad=pad_val
                        )
    cbar.ax.set_title(sdata_type, size='xx-small')
    cbar.ax.tick_params(labelsize='xx-small')
    
    plot_polys(ax, input_path, poly_plot_dict)

    ax.scatter(xx, yy, facecolor=fc, edgecolor=fc, linewidth=0.25, s=ps)
    for i, txt in enumerate(all_txt):
        ax.annotate(txt, (xx[i]+xlabel_shift, yy[i]), fontsize=fs, color=fc)
    
    PlotName = outpath + base_name +'.png'
    plt.tight_layout()
    plt.savefig(PlotName, dpi=fdpi, bbox_inches='tight')


def plot_2D_basic(
                    outpath, base_name,
                    gridx, gridy, gs,  
                    nwells, wells_dict,
                    poly_plot_dict, plot_dict
):
    CI = plot_dict["contour_Interval"]
    excluded_val = plot_dict["excluded_value"]
    gs_min = plot_dict["minimum_value"]
    # maximum value
    gs_max = plot_dict["maximum_value"]
    dxmin = plot_dict["xmin_m"]/1000
    # max. x-value
    dxmax = plot_dict["xmax_m"]/1000
    # min. y-value
    dymin = plot_dict["ymin_m"]/1000
    # max. y-value
    dymax = plot_dict["ymax_m"]/1000
    # x spacing for major axis ticks
    xspacing = plot_dict["x-spacing_m"]/1000
    # y spacing for major axis ticks
    yspacing = plot_dict["y-spacing_m"]/1000
    sdata_type = plot_dict["data_type"]
    # conversion factor for length L_conv = L_orig/fconv_len (original in km)
    fconv_len = 1.0
    excluded_vals = [excluded_val]
    gridx = gridx*fconv_len
    gridy = gridy*fconv_len
    if gs_min in excluded_vals:
        print("minimum scalar is in excluded values, set minimum to zero")
        plot_dict["minimum_value"] = 0.0
    Nlevels = int((gs_max - gs_min)/CI) + 1
    if Nlevels > 50:
        Nlevels = 50
        CI = int((gs_max - gs_min)/(Nlevels - 1))
    V=[]
    linewidths_list = []
    icount = 0
    for i in range(Nlevels):
        val = gs_min + CI*float(i)
        if val not in excluded_vals:
            V.append(val)
            if icount == 0:
                lwv  = 0.5
                icount = icount + 1
            else:
                lwv = 0.75
                icount = 0
            linewidths_list.append(lwv) 
    Nlevels2 = int((gs_max-gs_min)/(CI*2.0))+1
    if Nlevels > 50:
        Nlevels = 50
        CI = int((gs_max - gs_min)/(Nlevels - 1))
    V2=[]
    icount = 0
    for i in range(Nlevels2):
        val = gs_min + CI*2.0*float(i)
        V2.append(val)
    # Oil	0.58
    # Late Oil	0.85
    # Wet Gas	1.25
    # Gas	1.80
    # Super Mature	2.50
    if sdata_type == "Ro":
        #V2 = [0.58, 0.85, 1.25, 1.8, 2.5, 3, 5]
        V2 =[
              0.65,
              0.72,
              0.85,
              1.03,
              1.27,
              2.57,
              3.61,
              5.0
            ]
    xspacing_4ticks = xspacing
    yspacing_4ticks = yspacing
    xsize = dxmax - dxmin
    nxticks = int(xsize/xspacing)
    if nxticks > 50:
        nxticks = 50
        xspacing_4ticks = xsize/nxticks  
    ysize = dymax - dymin
    nyticks = int(ysize/yspacing)
    if nyticks > 50:
        nyticks = 50
        yspacing_4ticks = ysize/nyticks
    xticks = []
    for i in range(nxticks+1):
        xtick = dxmin + xspacing_4ticks*float(i)
        xticks.append(xtick)
    yticks = []
    for i in range(nyticks+1):
        ytick = dymin + yspacing_4ticks*float(i)
        yticks.append(ytick)
    plot_2D_grid(
                gridx, gridy, gs, 
                xticks, yticks,
                V, V2, linewidths_list,
                base_name, outpath, 
                nwells, wells_dict, 
                poly_plot_dict, plot_dict
                )

def plotit_map_main(file_path, input_path, itype_option):
    input_file_path = input_path + "ipa_plot_zmaps.csv"
    plot_dict, poly_plot_dict = fileIO.read_zmap_plot_file_csv(input_file_path)
    
    AOI_np_dum = np.zeros((1,1))
    file_input_path = os.path.dirname(file_path) + "\\"
    output_path = file_input_path
    
    all_map_names = []
    for file in os.listdir(file_input_path):
        file = file.replace('.zip','')
        if file.endswith(".dat"):
            all_map_names.append(file)
    
    file_name = os.path.basename(file_path)
    file_name_master = file_name
    if itype_option in [0,1]:
        all_map_names = [file_name]
    
    for file_name in all_map_names:
        print("Working on map file ", file_name)
        file_name_out = file_name.replace('.zip','')
        file_name_out = file_name.replace('.dat','')
        (
         scalars_xy_np, nx, ny, dx, dy, 
         xmin, xmax, ymin, ymax, AOI_np_L
        ) = map_tools.read_ZMAP(file_input_path, file_name, AOI_np_dum)
        
        smax = np.amax(scalars_xy_np)
        smin = np.amin(scalars_xy_np)
        if smin in [-99999.00,-99999]:
            smin = 0.0
        
        if file_name == file_name_master:
            txt_file_name = output_path + file_name_out+".csv"
            fout = open(txt_file_name, 'w')
            str_out = ("nx, ny, dx, dy, xmin, xmax, ymin, "
                       +"ymax, scalar_min, scalar_max \n")
            fout.write(str_out)
            str_out = (
                 str(nx) + "," + str(ny) + "," + str(dx) + "," + str(dy) + "," 
               + str(xmin) + "," + str(xmax) + "," + str(ymin) + "," 
               + str(ymax) + "," + str(smin) + "," + str(smax) + " \n"
               )
            fout.write(str_out)
            fout.close()
        
        gs = np.zeros((ny,nx))
        for j in range(ny):
            for i in range(nx):
                gs[j,i] = scalars_xy_np[i][j]
        gridx = np.zeros((nx))
        for i in range(nx):
            x = xmin + float(i)*dx
            gridx[i] = x/1000
        gridy = np.zeros((ny))
        for i in range(ny):
            y = ymin + float(i)*dy
            gridy[i] = y/1000
            
        if itype_option == 0:
            plot_dict["minimum_value"] = smin
            plot_dict["maximum_value"] = smax
            plot_dict["xmin_m"] = xmin
            plot_dict["xmax_m"] = xmax
            plot_dict["ymin_m"] = ymin
            plot_dict["ymax_m"] = ymax
            # minimum x-value
            dxmin = plot_dict["xmin_m"]
            # max. x-value
            dxmax = plot_dict["xmax_m"]
            # min. y-value
            dymin = plot_dict["ymin_m"]
            # max. y-value
            dymax = plot_dict["ymax_m"]        
            plot_dict["x-spacing_m"] = (dxmax-dxmin)/10.0
            plot_dict["y-spacing_m"] = (dymax-dymin)/10.0
        
        input_file_path = input_path + "ipa_wells.csv"
        nwells, wells_dict = fileIO.read_well_file_csv(input_file_path)
        
        plot_2D_basic(
                        output_path, file_name_out,
                        gridx, gridy, gs, 
                        nwells, wells_dict, 
                        poly_plot_dict, plot_dict
                        )


if __name__ == "__main__":
    file_path = sys.argv[1]
    #output_path = sys.argv[2]
    #file_name_out = sys.argv[3]
    input_path = str(sys.argv[4])
    itype_option = int(sys.argv[5])
    
    plotit_map_main(file_path, input_path, itype_option)
