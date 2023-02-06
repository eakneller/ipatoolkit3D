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
import os
import numpy as np
from numba import jit
import maturity
import compression

    
def read_ZMAP(input_path, file_name, AOI_np):
    """ Read zmap format map file
    
    Comments are from osgeo.org mailing list by David M. Baker. See the 
    following website for original post:
        https://lists.osgeo.org/pipermail/gdal-dev/2011-June/029173.html

    The ZMapPlus is an old format used to store gridded data in an ASCII line 
    format for transport and storage.  It is Landmark Corp. format that many 
    of the Landmark products, including GeoGraphix, used for data exchange.  
    Because it has been around for a while, many industry (Oil and Gas 
    Exploration) applications read and write the format. It is for gridded 
    data. The format can also support point and polygon data, but only one 
    data type is allowed in each file.  There are a specific set of header 
    rows that define how the data is written in the file.

     After the file description line are three more header lines that must 
     exist for a grid file. The first of these three lines has five fields.
     The first field is the field width of each grid node as
     stored in the data section below the last "@".
     The second field is the missing or null data value as it will 
     be found in the data section.
     The third field is a user defined text value used to indicate 
     a missing or null value. 
     The text value is used if field two is blank.
     If field two is defined, then the third field is left blank.
     The forth field indicates the number of decimal places to use
     if no decimal point is found in the data nodes.  
     (Historical note: when disk and tape space was at a premium, 
     this field made it possible to save some space in the
     file by not having to store a "." for each data node.).
     If the data values are written to the file with decimals, 
     this value is ignored.
     The fifth field indicates the starting column of the first 
     grid node on each line in the data section of the file.
    
     If the first character is a "!", the line is a comment.
     The header section starts with the first line that
     has an "@" symbol.  The data starts on the first line
     after the last "@" symbol, and there may only be two.
     Header fields are comma delimited.                   
     On the same line as the first "@" there are three fields.
     The first is user defined but many times is just "GRID FILE",
     the second, for a grid file, must be "GRID", the third is an
     integer that indicates the number of grid nodes per physical line.
     
     The second of the three lines has six fields.
     The first field is the number of rows in the grid.
     The second field is the number of columns in the grid.
     The third is the minimum grid X node value.
     The forth is the maximum grid X node value.
     The fifth is the minimum grid Y node value.
     The sixth is the maximum grid Y node value.
     The last four fields define the bounding box of the grid, and 
     implies, for example, the lower left corner of the of the grid 
     is the lower left corner of the lower left node in the grid.
     The projection and or datum of the bounding box value is not 
     defined.
     
     The third of the three lines is always "0,0, 0.0, 0.0".
     
     After the last header line, there is a single line with a single
     "@", the line after which is the first line of data in the data 
     section of the file. The data section has fixed field widths 
     each field being a single grid node, and is generally right 
     justified.  There will be no more nodes on any physical
     line than that defined in third field of the first header line. 
     A data field may or may not have a decimal point. If none is 
     found, it is implied, and the decimal places are as defined in 
     the second header line in the forth 
     
     The grid nodes in the data section are stored in column major 
     order. That is the first column of data is written first, 
     starting at the upper left corner of the grid.  For example, if 
     the grid has 7 rows and three columns, and the number of nodes
     per line is 4, the first line of the data section will have 4
     nodes, the first four grid nodes going down from the upper left.
     The second line will have three nodes, the last three nodes of 
     the first column. Then the next column is written, four nodes 
     then three.  Then the last column is written in the same pattern.
     
                                                            -- Baker (2011)
    """
    ifind_zip = compression.check_for_zipped_zmap(input_path, file_name)
    if ifind_zip == 1:
        compression.unzip_zmap(input_path, file_name)
    nflag = -99999.0    
    scalars = np.zeros((1))    
    icount = 0
    file_path = os.path.join(input_path, file_name)
    icheck = 0
    icount_col = 0
    icount_col_elem = 0
    with open(file_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            data = line.split()
            p1 = data[0]
            # ignore all lines that begin with !
            if p1 != "!":
                if icount > 4:
                    data = line.split()
                else:
                    data = line.split(",")
                if icount == 0:
                    npL = int(data[2])
                elif icount == 1:
                    #nch = int(data[0])
                    nan_flag = float(data[1])
                    #nan_flag_str = data[2]
                    #ndec = int(data[3])
                    #inode_start = int(data[4])
                elif icount == 2:
                    ny = int(data[0])
                    nx = int(data[1])
                    xmin = float(data[2])
                    xmax = float(data[3])
                    ymin = float(data[4])
                    ymax = float(data[5])
                    npr = int(ny/npL)
                    nrem = ny - npr*npL
                    if nrem > 0:
                        npr = npr + 1
                    scalars = np.ones((nx,ny))*nan_flag
                elif icount == 3:
                    pass
                elif icount == 4:
                    pass
                elif icount > 4:
                    if icount == 5:
                        col_np = np.zeros((ny))
                    if icheck == npr:
                        # Reverse the order of column entries so the first is 
                        # closest to the origin
                        nelem = col_np.size
                        for j in range(nelem):
                            jj = nelem - 1 - j
                            scalars[icount_col, jj] = col_np[j]
                        icount_col = icount_col + 1
                        col_np = np.zeros((ny))
                        icheck = 0
                        icount_col_elem = 0
                    for j, elem in enumerate(data):
                        felem = float(elem)
                        if felem in [99999.0, 0.1000000e31]:
                            felem = -99999.0
                        col_np[icount_col_elem] = felem
                        icount_col_elem = icount_col_elem + 1
                    icheck = icheck + 1
                icount = icount + 1
    # Add last column of data
    # Reverse the order of column entries so the first is closest to the origin
    nelem = col_np.size
    for j in range(nelem):
        jj = nelem -1 - j
        scalars[icount_col, jj] = col_np[j]                                        
    dx = (xmax-xmin)/float(nx-1)
    dy = (ymax-ymin)/float(ny-1)    
    # Only update the master array if sizes are equivalent
    AOI_np_L = np.zeros((nx, ny))    
    AOI_np_L_size = AOI_np_L.size
    AOI_np_size = AOI_np.size
    update_master_AOI = 0
    if AOI_np_size == AOI_np_L_size:
        update_master_AOI = 1        
    for i in range(nx):
        for j in range(ny):
            s = scalars[i][j]
            if s == nflag or s == nan_flag:
                AOI_np_L[i][j] = 0
                if update_master_AOI == 1:
                    AOI_np[i][j]=0
            else:
                AOI_np_L[i][j] = 1
    if ifind_zip == 1:
        # Delete zmap file
        os.remove(os.path.join(input_path, file_name))
    return scalars, nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np_L

       
def make_output_file_ZMAP_v4(
                                output_path, my_file_name, 
                                scalars, nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax, 
                                AOI_np
):    
    """ Export zmap format map file
    
    Comments are from osgeo.org mainling list by David M. Baker. See the 
    following website for original post:
        https://lists.osgeo.org/pipermail/gdal-dev/2011-June/029173.html

    The ZMapPlus is an old format used to store gridded data in an ASCII line 
    format for transport and storage.  It is Landmark Corp. format that many 
    of the Landmark products, including GeoGraphix, used for data exchange.  
    Because it has been around for a while, many industry (Oil and Gas 
    Exploration) applications read and write the format. It is for gridded 
    data. The format can also support point and polygon data, but only one 
    data type is allowed in each file.  There are a specific set of header 
    rows that define how the data is written in the file.
                                                            -- Baker (2011)
    """
    nflag = -99999.0    
    nlines = 5
    # If the first character is a "!", the line is a comment.
    # The header section starts with the first line that
    # has an "@" symbol.  The data starts on the first line
    # after the last "@" symbol, and there may only be two.
    # Header fields are comma delimited.
    file_name = os.path.join(output_path, my_file_name + ".dat")
    fout = open(file_name, 'w')
    str_out = "! \n"
    fout.write(str_out)
    str_out = "! ZMAP file made with ZMAP \n"
    fout.write(str_out)
    str_out = "! \n"
    fout.write(str_out)
    # On the same line as the first "@" there are three fields.
    # The first is user defined but many times is just "GRID FILE",
    # the second, for a grid file, must be "GRID", the third is an
    # integer that indicates the number of grid nodes per physical line.
    str_out = "@GRID FILE, GRID, "+str(nlines)+" \n"
    fout.write(str_out)
    # After the file description line are three more header lines that
    # must exist for a grid file.
    # The first of these three lines has five fields.
    # The first field is the field width of each grid node as
    # stored in the data section below the last "@".
    # The second field is the missing or null data value as it will be found 
    # in the data section.
    # The third field is a user defined text value used to indicate a missing
    # or null value.  The text value is used if field two is blank.
    # If field two is defined, then the third field is left blank.
    # The forth field indicates the number of decimal places to use if no 
    # decimal point is found in the data nodes.  (Historical note: when disk 
    # and tape space was at a premium, this field made it possible to save 
    # some space in the file by not having to store a "." for each data node.)  
    # If the data values are written to the file with decimals, this value is 
    # ignored.
    # The fifth field indicates the starting column of the first grid node on 
    # each line in the data section of the file.
    str_out = str(15)+","+str(nflag)+","+" "+","+str(7)+","+str(1)+"\n"
    fout.write(str_out)
    # The second of the three lines has six fields.
    # The first field is the number of rows in the grid.
    # The second field is the number of columns in the grid.
    # The third is the minimum grid X node value.
    # The forth is the maximum grid X node value.
    # The fifth is the minimum grid Y node value.
    # The sixth is the maximum grid Y node value.
    # The last four fields define the bounding box of the grid, and implies, 
    # for example, the lower left corner of the of the grid is the lower left 
    # corner of the lower left node in the grid.
    # The projection and or datum of the bounding box value is not defined.
    str_out = (
                  str(ny) + ","
                + str(nx) + ","
                + str(xmin) + ","
                + str(xmax) + ","
                + str(ymin) + ","
                + str(ymax) + "\n"
            )
    fout.write(str_out)
    # The third of the three lines is always "0,0, 0.0, 0.0".
    str_out = "0.0, 0.0, 0.0 \n"
    fout.write(str_out)
    #After the last header line, there is a single line with a single "@", the 
    # line after which is the first line of data in the data section of the 
    # file. The data section has fixed field widths each field being a single 
    # grid node, and is generally right justified.  There will be no more nodes 
    # on any physical line than that defined in third field of the first 
    # header line.  A data field may or may not have a decimal point.  If none 
    # is found, it is implied, and the decimal places are as defined in the 
    # second header line in the forth field
    str_out = "@ \n"
    fout.write(str_out)
    # The grid nodes in the data section are stored in column major order.
    # That is the first column of data is written first, starting at the upper
    # left corner of the grid.  For example, if the grid has 7 rows and three
    # columns, and the number of nodes per line is 4, the first line of the 
    # data section will have 4 nodes, the first four grid nodes going down 
    # from the upper left. The second line will have three nodes, the last 
    # three nodes of the first column. Then the next column is written, four 
    # nodes then three.  Then the last column is written in the same pattern.    
    # Columns (x-direction)
    for i, coors in enumerate(scalars):
        ncoors = len(coors)
        icount_out = 1
        # Rows (y-direction)
        for j in range(ncoors):
            jj = ncoors - j - 1 # Reverse order for ZMAP format  
            iZNON = AOI_np[i][jj]
            if iZNON == 0:
                sv = nflag
                str_tmp = "%15i" % sv
            else:
                sv = coors[jj]
                if math.isnan(sv) == True:
                    sv = 0.0
                if math.isinf(sv) == True:
                    sv = 0.0
                str_tmp = "%15.5e" % sv
            if icount_out == 1:
                str_out = str_tmp
            else:
                str_out = str_out+str_tmp
            if icount_out == nlines:
                str_out = str_out + "\n"    
                fout.write(str_out)
                icount_out = 1
            else:
                icount_out = icount_out + 1
        if icount_out > 1:
            str_out = str_out + "\n"    
            fout.write(str_out)
    fout.close()
    icompress = 1
    compression.zip_zmap(icompress, output_path, my_file_name + ".dat")
    

def output_zmap_history(
                        stype, output_path, model, 
                        Lx, Ly, nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                        AOI_np, sflag, imake_all_history, imake_all_tops, 
                        src_top_names
):    
    keys = list(model.event_dict_bs.keys())    
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]    
    ntops = len(model.tops_list_bs)    
    for kk, event_ID in enumerate(keys):    
        if imake_all_history == 0:
            if event_ID == event_ID_last_bs:
                imake_files = 1
            else:
                imake_files = 0
        else:
            imake_files = 1
        
        if imake_files == 1:
            itop_event = model.event_dict_bs[event_ID][2]
            age = model.event_dict_bs[event_ID][0]
            for jj in range(ntops):
                if jj <= itop_event:
                    if imake_all_tops == 0 and event_ID != event_ID_last_bs:
                        if jj == itop_event:
                            imake_top = 1
                        else:
                            imake_top = 0
                    else:
                        imake_top = 1
                    if imake_top == 1:
                        event_index = model.tops_list_bs[jj][14][event_ID]
                        name = model.tops_list_bs[jj][6]
                        ifind_src = 0
                        if name in src_top_names:
                            ifind_src = 1
                        if stype == "TEMP":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][38][event_index])
                        elif stype == "EasyRo":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][39][event_index])
                        elif stype == "LOM":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][40][event_index])
                        elif stype == "TR":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][41][event_index])
                        elif stype == "mHC":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][42][event_index])
                        elif stype == "mODG":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][43][event_index])
                        elif stype == "mFG":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][44][event_index])
                        elif stype == "EXPRATE":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][45][event_index])
                        elif stype == "SEC_EXPRATE":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][62][event_index])                   
                        elif stype == "SEC_mFG":
                            scalar_xy = np.copy(
                                    model.tops_list_bs[jj][58][event_index])                                           
                        file_name = (
                                        stype + "_"
                                        + sflag 
                                        + "_EV_"+str(event_ID)
                                        +"_t"+str(jj)
                                        +"_AGE_"+str(age)
                                        +"_TOP_"+name
                                    )                        
                        imake_file = 1
                        if stype in ["EasyRo","LOM","TR","mHC","mODG","mODG",
                                     "mFG","EXPRATE","SEC_EXPRATE","SEC_mFG"]:
                            if ifind_src != 1:
                                imake_file = 0                        
                        # Only output src related maps if layer 
                        # is classified as a source
                        if imake_file == 1:
                            make_output_file_ZMAP_v4(
                                                        output_path, file_name, 
                                                        scalar_xy, nx, ny, 
                                                        dx, dy, 
                                                        xmin, xmax, ymin, ymax, 
                                                        AOI_np
                                                    ) 


@jit(nopython=True, cache=True)
def low_pass_filter_zmap(
                            dx, dy, nx, ny, AOI_np, rad_search_m, 
                            scalars_xy, scalars_lpf_xy
):
    # area of each element in km^3
    dA = dx/1000*dy/1000
    vol_tot = 0.0
    for i in range(nx): # Columns
        for j in range(ny): # Rows
            imin = i - int(rad_search_m/dx)
            if imin < 0: 
                imin = 0
            imax = i + int(rad_search_m/dx)
            if imax < 1: 
                imax = 0
            jmin = j - int(rad_search_m/dy)
            if jmin < 0: 
                jmin = 0
            jmax = j + int(rad_search_m/dy)
            if jmax < 1: 
                jmax = 1
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                icount = 0
                vsum = 0.0
                for ii in range(imin, imax+1):
                    for jj in range(jmin, jmax+1):
                        AOI_flag = AOI_np[ii][jj]
                        if AOI_flag == 1:
                            if (ii >= imin and ii <= imax 
                                and jj >= jmin and jj <= jmax):
                                vsum = vsum + scalars_xy[ii][jj]
                                icount = icount + 1
                val = vsum / float(icount)
            else:
                val = -99999.0            
            if val >= 0.0:
                vol_tot = vol_tot + dA*val/1000.0                
            scalars_lpf_xy[i][j] = val
    return vol_tot


def extract_at_node_history(
                            inode, jnode, itype_data, well_name, xx, yy, 
                            naflag, output_path, tops_list_bs, event_dict_bs, 
                            deltaSL_list, Lx, Ly, nx, ny, dx, dy, 
                            xmin, xmax, ymin, ymax, AOI_np
):
    nID = "i"+str(inode)+"j"+str(jnode)
    if itype_data == 1:
        stype = "_TVDssm"
        na_num = 0.0    
    elif itype_data == 38:        
        stype = "_TEMP_C_"
        na_num = 0.0
    elif itype_data == 42:
        stype = "_mHC_mg_gOC_"
        na_num = 0.0       
    elif itype_data == 45:
        stype = "_mHC_mg_gOC_Myr_"
        na_num = 0.0          
    elif itype_data == 52:
        stype = "_vFGEXSR_Tcf_"
        na_num = 0.0  
    elif itype_data == 53:
        stype = "_vODGEXSR_GOB_"
        na_num = 0.0
    elif itype_data == 54:
        stype = "_rhog_g_cm3_"
        na_num = 0.0        
    elif itype_data == 55:
        stype = "_rhoL_g_cm3_"
        na_num = 0.0  
    elif itype_data == 56:
        stype = "_Bg_rcf_scf_"
        na_num = 0.0 
    elif itype_data == 57:
        stype = "_Bo_rbbl_sbbl"
        na_num = 0.0
    elif itype_data == 58:
        stype = "_SEC_mHC_mg_gOC_"
        na_num = 0.0 
    elif itype_data == 60:
        stype = "_rhog_SEC_g_cm3_"
        na_num = 0.0
    elif itype_data == 61:
        stype = "_Bg_SEC_rcf_scf"
        na_num = 0.0
    elif itype_data == 62:
        stype = "_SEC_mHC_mg_gOC_Myr_"
        na_num = 0.0 
    elif itype_data == 39:
        stype = "_EasyRo_"
        na_num = 0.0 
    elif itype_data == 40:
        stype = "_LOM_"
        na_num = 0.0 
    elif itype_data == 59:
        stype = "_SEC_vFGEXSR_Tcf_"
        na_num = 0.0 
    elif itype_data == 41:
        stype = "_TR_"
        na_num = 0.0 
    if itype_data in [1, 38, 39, 40, 41, 
                      42, 45, 52, 53, 54, 
                      55, 56, 57, 58, 59, 
                      60, 61, 62]:
        keys = list(event_dict_bs.keys())        
        ntops = len(tops_list_bs)
        file_name = os.path.join(output_path, 
                                 "node"+ nID + "_" + stype + "_History.csv")
        fout = open(file_name, 'w')
        data = ["Age_Ma"]
        pwd_xy_event_list = []
        # Loop over events from old to young
        for event_ID, key in enumerate(keys):
            itop_event = event_dict_bs[event_ID][2]
            age_event = event_dict_bs[event_ID][0]
            data.append(age_event)
            s = event_dict_bs[event_ID][5][inode][jnode]
            pwd_xy_event_list.append(s)
        str_out = ','.join(map(str, data)) + "\n"
        fout.write(str_out)
         # Loop over tops from young to old
        for mm in range(ntops):
            jj = ntops - 1 - mm
            # Name
            name_top = tops_list_bs[jj][6]
            data = [name_top]
            # Loop over events from old to young
            for event_ID, key in enumerate(keys):
                itop_event = event_dict_bs[event_ID][2]
                age_event = event_dict_bs[event_ID][0]
                base_level = deltaSL_list[event_ID]
                # Use water depth for na_num if data type is subsea depth
                if itype_data == 1:
                    na_num = pwd_xy_event_list[event_ID]+base_level
                if jj <= itop_event:
                    event_index = tops_list_bs[jj][14][event_ID]
                    s = tops_list_bs[jj][itype_data][event_index][inode][jnode]
                else:
                    s = na_num
                data.append(s)
            str_out = ','.join(map(str, data)) + "\n"
            fout.write(str_out)
        fout.close()
    else:
        print("extract_at_node_history: itype_data ", itype_data, " not found")


def extract_at_xy_history(
                            itype_data, well_name, xx, yy, naflag, output_path,
                            tops_list_bs, event_dict_bs, deltaSL_list, 
                            Lx, Ly, nx, ny, dx, dy, 
                            xmin, xmax, ymin, ymax, AOI_np
):
    
    if itype_data == 1:
        stype = "_TVDssm"
        na_num = 0.0    
    elif itype_data == 38:        
        stype = "_TEMP_C_"
        na_num = 0.0
    elif itype_data == 39:
        stype = "_EasyRo_"
        na_num = 0.0 
    elif itype_data == 40:
        stype = "_LOM_"
        na_num = 0.0 
    elif itype_data == 41:
        stype = "_TR_"
        na_num = 0.0 
    elif itype_data == 42:
        stype = "_mHC_mg_gOC_"
        na_num = 0.0       
    elif itype_data == 45:
        stype = "_mHC_mg_gOC_Myr_"
        na_num = 0.0          
    elif itype_data == 52:
        stype = "_vFGEXSR_Tcf_"
        na_num = 0.0  
    elif itype_data == 53:
        stype = "_vODGEXSR_GOB_"
        na_num = 0.0
    elif itype_data == 54:
        stype = "_rhog_g_cm3_"
        na_num = 0.0        
    elif itype_data == 55:
        stype = "_rhoL_g_cm3_"
        na_num = 0.0  
    elif itype_data == 56:
        stype = "_Bg_rcf_scf_"
        na_num = 0.0 
    elif itype_data == 57:
        stype = "_Bo_rbbl_sbbl"
        na_num = 0.0
    elif itype_data == 58:
        stype = "_SEC_mHC_mg_gOC_"
        na_num = 0.0 
    elif itype_data == 59:
        stype = "_SEC_vFGEXSR_Tcf_"
        na_num = 0.0
    elif itype_data == 60:
        stype = "_rhog_SEC_g_cm3_"
        na_num = 0.0
    elif itype_data == 61:
        stype = "_Bg_SEC_rcf_scf"
        na_num = 0.0
    elif itype_data == 62:
        stype = "_SEC_mHC_mg_gOC_Myr_"
        na_num = 0.0
        
    if itype_data in [
                      1, 38, 39, 40, 41, 
                      42, 45, 52, 53, 54, 
                      55, 56, 57, 58, 59, 
                      60, 61, 62
                      ]:
        keys = list(event_dict_bs.keys())
        ntops = len(tops_list_bs)
        file_name = os.path.join(output_path, 
                                 well_name + stype + "_History.csv")
        fout = open(file_name, 'w')
        data = ["Age_Ma"]
        pwd_xy_event_list = []
        # Loop over events from old to young
        for event_ID, key in enumerate(keys):
            itop_event = event_dict_bs[event_ID][2]
            age_event = event_dict_bs[event_ID][0]
            data.append(age_event)
            pwd_xy = event_dict_bs[event_ID][5]
            s = zmap_interp(
                            xx, yy, naflag, pwd_xy, nx, ny, dx, dy, 
                            xmin, xmax, ymin, ymax
                        )
            pwd_xy_event_list.append(s)
        str_out = ','.join(map(str, data)) + "\n"
        fout.write(str_out)
        # Loop over tops from young to old
        for mm in range(ntops):
            jj = ntops-1-mm
            # Name
            name_top = tops_list_bs[jj][6]
            data = [name_top]
            # Loop over events from old to young
            for event_ID, key in enumerate(keys):
                itop_event = event_dict_bs[event_ID][2]
                age_event = event_dict_bs[event_ID][0]
                base_level = deltaSL_list[event_ID]
                # Use water depth for na_num if data type is subsea depth
                if itype_data == 1:
                    na_num = pwd_xy_event_list[event_ID]+base_level    
                if jj <= itop_event:    
                    event_index = tops_list_bs[jj][14][event_ID]
                    scalars_xy = tops_list_bs[jj][itype_data][event_index]
                    s = zmap_interp(
                                                xx, yy, naflag, scalars_xy, 
                                                nx, ny, dx, dy, 
                                                xmin, xmax, ymin, ymax
                                            )
                else:
                    s = na_num
                data.append(s)
            str_out = ','.join(map(str, data)) + "\n"
            fout.write(str_out)
        fout.close()   
    else:
        print("extract_at_xy_history: itype_data ", itype_data, " not found")


def extract_at_xy_at_0Ma(
                            well_name, xx, yy, naflag, output_path, 
                            tops_list_bs, event_dict_bs, deltaSL_list,
                            Lx, Ly, nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                            AOI_np, imass_gen, icalc_LOM, icalc_temp
):
    keys = list(event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
    ntops = len(tops_list_bs)
    file_name = os.path.join(output_path, well_name + "_ModelOutput.csv")
    print("Creating well file: ", file_name)
    fout = open(file_name, 'w')
    # pwd_xy (m)
    scalars_xy = event_dict_bs[event_ID_last_bs][5]
    pwd = zmap_interp(
                        xx, yy, naflag, scalars_xy, nx, ny, dx, dy, 
                        xmin, xmax, ymin, ymax
                    )
    data = [
            "TopName", 
            "z_submud_m", 
            "Age_Ma", 
            "T_trans_C", 
            "Ro", 
            "LOM", 
            "TR", 
            "HCGEN_mgHC_gOC",
            "ODGEXP_mgHC_gOC", 
            "FGEXP_mgHC_gOC",
            "EXRate_mgHC_gOC_Myr", 
            "WaterDepth_m",
            "OilCrack_FGEXP_mgHC_gOC", 
            "OilCrack_EXRate_mgHC_gOC_Myr", 
            "FGEXSR_m3", 
            "ODGEXSR_m3",
            "Bg_primary_rcf_scf", 
            "Bo_primary_bbl_bbl",
            "rho_G_primary_g_cm3", 
            "rho_L_primary_g_cm3",
            "OilCrack_FGEXSR_m3", 
            "Bg_OilCrack_cf_cf", 
            "rho_G_OilCrack_g_cm3"
        ]
    str_out = ','.join(data) + "\n"
    fout.write(str_out)
    for mm in range(ntops):
        jj = ntops - 1 - mm
        event_index = tops_list_bs[jj][14][event_ID_last_bs]
        # Depth subsea (m)
        scalars_xy = tops_list_bs[jj][1][event_index]
        s = zmap_interp(
                                    xx, yy, naflag, scalars_xy, nx, ny, 
                                    dx, dy, xmin, xmax, ymin, ymax
                                )
        z_submud_m = s - pwd
        # Age (Ma)
        age = tops_list_bs[jj][4]        
        name = tops_list_bs[jj][6]
        if icalc_temp == 1:
            # Temp (C)
            scalars_xy = tops_list_bs[jj][38][event_index]
            T = zmap_interp(
                            xx, yy, naflag, scalars_xy, nx, ny, 
                            dx, dy, xmin, xmax, ymin, ymax
                        )
            # Ro (%)
            scalars_xy = tops_list_bs[jj][39][event_index]
            Ro = zmap_interp(
                                xx, yy, naflag, scalars_xy, nx, ny, 
                                dx, dy, xmin, xmax, ymin, ymax
                            )
            # Conversion from Walters (2008)
            LOM = maturity.ro_to_LOM(Ro)
        else:
            T = 0.0
            Ro = 0.0
            LOM = 0.0
        
        if imass_gen > 0:
            # TR
            scalars_xy = tops_list_bs[jj][41][event_index]
            TR = zmap_interp(
                                xx, yy, naflag, scalars_xy, nx, ny, 
                                dx, dy, xmin, xmax, ymin, ymax)            
            # mHC
            scalars_xy = tops_list_bs[jj][42][event_index]
            mHC = zmap_interp(
                                xx, yy, naflag, scalars_xy, nx, ny, 
                                dx, dy, xmin, xmax, ymin, ymax)            
            # mODG
            scalars_xy = tops_list_bs[jj][43][event_index]
            mODG = zmap_interp(
                                xx, yy, naflag, scalars_xy, nx, ny, 
                                dx, dy, xmin, xmax, ymin, ymax)            
            # mFG
            scalars_xy = tops_list_bs[jj][44][event_index]
            mFG = zmap_interp(
                                xx, yy, naflag, scalars_xy, nx, ny, 
                                dx, dy, xmin, xmax, ymin, ymax)            
            # Expulsion Rate
            scalars_xy = tops_list_bs[jj][45][event_index]
            scalars_xy_np = np.asarray(scalars_xy)
            ER = zmap_interp(
                                xx, yy, naflag, scalars_xy_np, nx, ny, 
                                dx, dy, xmin, xmax, ymin, ymax)            
            scalars_xy = tops_list_bs[jj][58][event_index]
            scalars_xy_np = np.asarray(scalars_xy)
            SEC_mFG = zmap_interp(
                                    xx, yy, naflag, scalars_xy_np, 
                                    nx, ny, dx, dy, xmin, xmax, 
                                    ymin, ymax)            
            scalars_xy = tops_list_bs[jj][62][event_index]
            SEC_ER = zmap_interp(
                                    xx, yy, naflag, scalars_xy, 
                                    nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax)                        
            scalars_xy = tops_list_bs[jj][52][event_index]
            VFGEXSR = zmap_interp(
                                    xx, yy, naflag, scalars_xy, 
                                    nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax)                        
            scalars_xy = tops_list_bs[jj][53][event_index]
            VLEXSR = zmap_interp(
                                    xx, yy, naflag, scalars_xy, 
                                    nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax)           
            scalars_xy = tops_list_bs[jj][56][event_index]
            Bg_p = zmap_interp(
                                xx, yy, naflag, scalars_xy, 
                                nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax)            
            scalars_xy = tops_list_bs[jj][57][event_index]
            Bo_p = zmap_interp(
                                xx, yy, naflag, scalars_xy, 
                                nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax)
            scalars_xy = tops_list_bs[jj][54][event_index]
            rhog_p = zmap_interp(
                                    xx, yy, naflag, scalars_xy, 
                                    nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax)
            scalars_xy = tops_list_bs[jj][55][event_index]
            rhoL = zmap_interp(
                                xx, yy, naflag, scalars_xy,
                                nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax)
            scalars_xy = tops_list_bs[jj][59][event_index]
            VFGEXSR_SEC = zmap_interp(
                                        xx, yy, naflag, scalars_xy, 
                                        nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax)
            scalars_xy = tops_list_bs[jj][61][event_index]
            Bg_s = zmap_interp(
                                xx, yy, naflag, scalars_xy, 
                                nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax)
            scalars_xy = tops_list_bs[jj][60][event_index]
            rhog_s = zmap_interp(
                                    xx, yy, naflag, scalars_xy, 
                                    nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax)
        else:
            TR = 0
            mHC = 0
            mODG = 0
            mFG = 0
            ER = 0
            
            SEC_mFG = 0
            SEC_ER = 0
            VFGEXSR = 0 
            VLEXSR = 0
            Bg_p = 0
            Bo_p = 0
            rhog_p = 0
            rhoL = 0
            VFGEXSR_SEC = 0
            Bg_s = 0
            rhog_s = 0
        data = [
                name, z_submud_m, age, T, 
                Ro, LOM, TR, mHC, mODG, 
                mFG, ER, pwd, SEC_mFG, 
                SEC_ER, VFGEXSR, VLEXSR, 
                Bg_p, Bo_p, rhog_p, rhoL, 
                VFGEXSR_SEC, Bg_s, rhog_s
            ]
        str_out = ','.join(map(str, data)) + "\n"
        fout.write(str_out)
    fout.close()


@jit(nopython=True, cache=True)
def add_delta(scalar_xy_np, delta, nx, ny, AOI_np):
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                delta_tmp = delta[i][j]                 
                vcheck = scalar_xy_np[i][j] + delta_tmp
                if vcheck < 0:
                    vcheck = 0
                scalar_xy_np[i][j] = vcheck


@jit(nopython=True, cache=True)
def interp_to_master(
                        nx, ny, xmin, ymin, dx, dy, AOI_np, naflag, 
                        nx_ini, ny_ini, dx_ini, dy_ini, 
                        xmin_ini, xmax_ini, ymin_ini, ymax_ini, 
                        scalars_xy_interp_np, scalars_xy_ini_np
):
    for i in range(nx):
        for j in range(ny):            
            xx = xmin + float(i)*dx
            yy = ymin + float(j)*dy
            s = zmap_interp(
                            xx, yy, naflag, scalars_xy_ini_np, 
                            nx_ini, ny_ini, dx_ini, dy_ini, 
                            xmin_ini, xmax_ini, ymin_ini, ymax_ini
                        )
            scalars_xy_interp_np[i][j] = s
            if int(s) == int(naflag):
                AOI_np[i][j] = 0


@jit(nopython=True, cache=True)
def zmap_interp(
                xx, yy, naflag, scalars_xy_np, 
                nx, ny, dx, dy, 
                xmin, xmax, ymin, ymax
):
    TOL=1e-3
    if xx < xmin+TOL or xx > xmax-TOL or yy < ymin+TOL or yy > ymax-TOL:        
        s = naflag        
    else:
        xx = xx - xmin
        yy = yy - ymin
        
        iLL = int(xx/dx)
        jLL = int(yy/dy)
        xLL = float(iLL)*dx
        yLL = float(jLL)*dy
        sLL = scalars_xy_np[iLL][jLL]
        dddx = abs(xx - xLL)/dx
        dddy = abs(yy - yLL)/dy
        fLL = (1-dddx)*(1-dddy)
        
        iUL = iLL
        jUL = jLL+1
        xUL = float(iUL)*dx
        yUL = float(jUL)*dy
        sUL = scalars_xy_np[iUL][jUL]
        dddx = abs(xx - xUL)/dx
        dddy = abs(yy - yUL)/dy
        fUL = (1-dddx)*(1-dddy)
        
        iUR = iLL + 1
        jUR = jLL + 1
        xUR = float(iUR)*dx
        yUR = float(jUR)*dy

        sUR = scalars_xy_np[iUR][jUR]
        dddx = abs(xx - xUR)/dx
        dddy = abs(yy - yUR)/dy
        fUR = (1-dddx)*(1-dddy)
        
        iLR = iLL + 1
        jLR = jLL
        xLR = float(iLR)*dx
        yLR = float(jLR)*dy
        sLR = scalars_xy_np[iLR][jLR]
        dddx = abs(xx - xLR)/dx
        dddy = abs(yy - yLR)/dy
        fLR = (1-dddx)*(1-dddy)
        
        fsum = fLL + fUL + fUR + fLR
        
        if (
               int(sLL) == int(naflag) or int(sUL) == int(naflag) 
            or int(sUR) == int(naflag) or int(sLR) == int(naflag)
        ):
            s = naflag
        else:
            s = (fLL*sLL + fUL*sUL + fUR*sUR + fLR*sLR)/fsum            
    return s

         
def read_ZMAP_interp_to_master(
                                naflag, master_xy, input_path, file_name, 
                                nx, ny, xmin, ymin, dx, dy, 
                                AOI_np, delta_xy_np
):
    # Read raw map
    AOI_dum = np.zeros((1))
    (
         scalars_xy_ini, 
         nx_ini, ny_ini, dx_ini, dy_ini, 
         xmin_ini, xmax_ini, ymin_ini, ymax_ini, AOI_dum2
     ) = read_ZMAP(input_path, file_name, AOI_dum)
    # Make a copy of numpy array
    scalars_xy_ini_np = np.copy(scalars_xy_ini)
    # Initialize interpolation array with master geometry
    scalars_xy_interp_np = np.copy(master_xy)
    interp_to_master(
                        nx, ny, xmin, ymin, dx, dy, 
                        AOI_np, naflag, 
                        nx_ini, ny_ini, dx_ini, dy_ini, 
                        xmin_ini, xmax_ini, ymin_ini, ymax_ini,
                        scalars_xy_interp_np, scalars_xy_ini_np
                    )
    # Add delta where delta has master grid geometry
    add_delta(scalars_xy_interp_np, delta_xy_np, nx, ny, AOI_np)
    return scalars_xy_interp_np


def read_ZMAP_interp_to_master_integers(
                                        naflag, master_xy, input_path, 
                                        file_name, nx, ny, xmin, ymin, 
                                        dx, dy, AOI_np
):
    AOI_dum = np.zeros((1))
    (
         scalars_xy_ini, 
         nx_ini, ny_ini, dx_ini, dy_ini, 
         xmin_ini, xmax_ini, ymin_ini, ymax_ini, AOI_dum2
     ) = read_ZMAP(input_path, file_name, AOI_dum)
    # Mkae copy of numpy array
    scalars_xy_ini_np = np.copy(scalars_xy_ini)
    # Initialize interpolation array with master geometry
    scalars_xy_interp_np = np.copy(master_xy)
    interp_to_master_integers(
                                nx, ny, xmin, ymin, dx, dy, AOI_np, naflag, 
                                nx_ini, ny_ini, dx_ini, dy_ini, 
                                xmin_ini, xmax_ini, ymin_ini, ymax_ini, 
                                scalars_xy_interp_np, scalars_xy_ini_np
                            )
    return scalars_xy_interp_np


@jit(nopython=True, cache=True)
def interp_to_master_integers(
                                nx, ny, xmin, ymin, dx, dy, AOI_np, naflag, 
                                nx_ini, ny_ini, dx_ini, dy_ini, 
                                xmin_ini, xmax_ini, ymin_ini, ymax_ini, 
                                scalars_xy_interp_np, scalars_xy_ini_np
):
    for i in range(nx):
        for j in range(ny):
            xx = xmin + float(i)*dx
            yy = ymin + float(j)*dy
            if i == 24 and j == 24:
                iprint = 1
            else:
                iprint = 0
            s = zmap_interp_integers(
                                        xx, yy, naflag, scalars_xy_ini_np, 
                                        nx_ini, ny_ini, dx_ini, dy_ini, 
                                        xmin_ini, xmax_ini, 
                                        ymin_ini, ymax_ini, iprint
                                    )    
            s = int(round(s))
            scalars_xy_interp_np[i][j] = s
            if int(s) == int(naflag):
                AOI_np[i][j]=0


@jit(nopython=True, cache=True)
def zmap_interp_integers(
                            xx, yy, naflag, scalars_xy_np, nx, ny, dx, dy, 
                            xmin, xmax, ymin, ymax, iprint
):
    TOL=1e-3
    if xx < xmin+TOL or xx > xmax-TOL or yy < ymin+TOL or yy > ymax-TOL:
        s = naflag
    else:
        xx = xx - xmin
        yy = yy - ymin
        
        iLL = int(xx/dx)
        jLL = int(yy/dy)
        xLL = xmin + float(iLL)*dx
        yLL = ymin + float(jLL)*dy
        sLL = scalars_xy_np[iLL][jLL]
        dddx = abs(xx - xLL)/dx
        dddy = abs(yy - yLL)/dy
        fLL = (1-dddx)*(1-dddy)
        
        iUL = iLL
        jUL = jLL+1
        xUL = xmin + float(iUL)*dx
        yUL = ymin + float(jUL)*dy
        sUL = scalars_xy_np[iUL][jUL]
        dddx = abs(xx - xUL)/dx
        dddy = abs(yy - yUL)/dy
        fUL = (1-dddx)*(1-dddy)
        
        iUR = iLL + 1
        jUR = jLL + 1
        xUR = xmin + float(iUR)*dx
        yUR = ymin + float(jUR)*dy
        sUR = scalars_xy_np[iUR][jUR]
        dddx = abs(xx - xUR)/dx
        dddy = abs(yy - yUR)/dy
        fUR = (1-dddx)*(1-dddy)
        
        iLR = iLL + 1
        jLR = jLL
        xLR = xmin + float(iLR)*dx
        yLR = ymin + float(jLR)*dy
        sLR = scalars_xy_np[iLR][jLR]
        dddx = abs(xx - xLR)/dx
        dddy = abs(yy - yLR)/dy
        fLR = (1-dddx)*(1-dddy)
        
        if fLL > fUL and fLL > fUR and fLL > fLR:
            s = sLL 
        if fUL > fLL and fUL > fUR and fUL > fLR:
            s = sUL
        if fUR > fLL and fUR > fUL and fUR > fLR:
            s = sUR
        if fLR > fLL and fLR > fUL and fLR > fUR:
            s = sLR
        if (
               int(sLL) == int(naflag) or int(sUL) == int(naflag) 
            or int(sUR) == int(naflag) or int(sLR) == int(naflag)
        ):
            s = int(naflag)    
    return s


def get_zmap_info(input_path, zmap_name):
        zmap_name = zmap_name + ".dat"
        print ("Getting information for ZMAP : ", zmap_name/0)
        AOI_in = np.ones((1,1))
        (
            scalars, nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_out
         ) = read_ZMAP(input_path, zmap_name, AOI_in)

        print ("----------------------------------")
        print ("Information for ZMAP ", zmap_name)
        print ("----------------------------------")
        print ("nx : ny : ", nx, " : ", ny)
        print ("dx : dy : ", dx, " : ", dy)
        print ("xmin : xmax : ", xmin, " : ", xmax)
        print ("ymin : ymax : ", ymin, " : ", ymax)
        print ("----------------------------------")


def clip_zmap(
                input_path, output_path, zmap_name, 
                i_xmin, i_xmax, j_ymin, j_ymax, sflag
):
        zmap_file_name = zmap_name + ".dat"
        print ("Clipping ZMAP : ", zmap_name)
        AOI_in = np.ones((1,1))
        (
            scalars_xy_ini, nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_out
         ) = read_ZMAP(input_path, zmap_file_name, AOI_in)
        print ("-----------------------------------------")
        print ("Information for initial ZMAP ", zmap_name)
        print ("-----------------------------------------")
        print ("nx : ny : ", nx, " : ", ny)
        print ("dx : dy : ", dx, " : ", dy)
        print ("xmin : xmax : ", xmin, " : ", xmax)
        print ("ymin : ymax : ", ymin, " : ", ymax)
        print ("-----------------------------------------")
        scalars_xy_clipped = []
        for i in range(nx):
            if i >= i_xmin and i <= i_xmax:
                col = []
                for j in range(ny):
                    s = scalars_xy_ini[i][j]         
                    if j >= j_ymin and j <= j_ymax:
                        col.append(s)           
                scalars_xy_clipped.append(col)
        nx_c = i_xmax-i_xmin+1
        ny_c = j_ymax-j_ymin+1
        dx_c = dx
        dy_c = dy
        xmin_c = xmin+float(i_xmin)*dx_c
        xmax_c = xmin+float(i_xmax)*dx_c
        ymin_c = ymin+float(j_ymin)*dy_c
        ymax_c = ymin+float(j_ymax)*dy_c
        print ("-----------------------------------------")
        print ("Information for clipped ZMAP ", zmap_name)
        print ("-----------------------------------------")
        print ("nx : ny : ", nx_c, " : ", ny_c)
        print ("dx : dy : ", dx_c, " : ", dy)
        print ("xmin : xmax : ", xmin_c, " : ", xmax_c)
        print ("ymin : ymax : ", ymin_c, " : ", ymax_c)
        print ("-----------------------------------------")
        if sflag != "none":
                file_name = zmap_name+sflag
        else:
                file_name = zmap_name
        AOI = np.ones((nx_c, ny_c))
        make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        scalars_xy_clipped, 
                                        nx_c, ny_c, dx_c, dy_c, 
                                        xmin_c, xmax_c, ymin_c, ymax_c,
                                        AOI
                                    )
    