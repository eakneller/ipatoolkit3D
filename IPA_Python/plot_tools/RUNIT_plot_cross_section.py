# -*- coding: utf-8 -*-
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
import cross_section_tools


zmin_xs = -1000.0
zmax_xs = 10000.0
fig_x_inches = 10
ve_fac = 6.0
#TorresArch
#x1 = 878275.63 
#y1 = 6687422.50
#x2 = 1075968.00
#y2 = 6915071.50
# ColoradoBasin profile 1: NW-SE
#x1 = 4796234.50
#y1 = 5672108.00
#x2 = 5194755.50
#y2 = 5398873.00
## ColoradoBasin profile 2: NE-SW
#x1 = 4883454.50
#y1 = 5399186.00
#x2 = 5084229.50
#y2 = 5656497.50
## Labrador profile 1
#x1 = 147710.69
#y1 = 6068618.5
#x2 = 272499.00
#y2 = 6211108.00
## Labrador profile 1
#x1 = 216651.75
#y1 = 6035353.5
#x2 = 356116.47
#y2 = 6211099.00
# Liberia profile 1
x1 = 992650.00
y1 = 488052.44
x2 = 1078789.63
y2 = 574099.50

main_output_path = "D:\\IPAtoolkit3D\\TestFlex2022_flex_r3\\"

cross_section_tools.plot_xs_main(
                                 zmin_xs, zmax_xs, 
                                 fig_x_inches, ve_fac, 
                                 x1, x2, y1, y2, 
                                 main_output_path
                                 )



                
                
        
        
    
            
            
