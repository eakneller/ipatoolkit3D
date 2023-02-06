# ipatoolkit3D
IPAtoolkit3D is a collection of Python scripts using Scipy, Numpy and Numba for 
parallel map-based thermal, integrated charge and flexure calculations.

The following versions were used to build the IPAtoolkit3D:
* conda 4.3.3
* Python 3.6.3
* numpy 1.13.3
* numba 0.35.0
* scipy 0.19.1

All Python code is located in a directory called IPA_Python.

Defining Inputs and Executing IPAtoolkit3D
-------------------------------------------------------------------------------
Input, output and Python execution can be managed via the macro-enabled workbook 
IPAtoolkit3D_1.0.xlsm that uses simple VBA scripts to export and import input 
files.

Input files include CSV files (.csv), Zmaps and ASCII polygon files (.poly) that 
are located in a user defined directory with a user defined path. Use a map 
editing software package to build and edit Zmaps. The user can manually edit 
CSV files but the format used in example files should be strictly followed. We 
recommended using the macro-enabled workbook called IPAtoolkit3D_1.0.xlsm to 
create, load and export input files. See the Workflow tab in the workbook for 
details.

IPAtoolkit3D can be executed using different methods:

* Use the "Execute Python Script" button in the "Model Input" tab of 
    IPAtoolkit3D_1.0.xlsm after setting paths and ensuring that a model is 
    loaded into the workbook (see Workflow tab).
* Use the Python or .bat scripts in BatchScripts. These scripts require 
    the user to define run type information, paths and other inputs before 
    executing.

Running Integrated Tests
-------------------------------------------------------------------------------
Use the script IPA_Python\integrated_tests\RUN_TESTS.py to run integrated
test cases. We recommend running these cases before starting a model exercise
to ensure that the code is working properly on your system. This script 
requires the user to choose which test to run and provides information on the 
parts of the code that are tested and success criteria. Integrated tests 
should also be used for testing modified versions of this code.

Turning Parallel On/Off
-------------------------------------------------------------------------------
Set the variable iforce_serial to 0 in the manage_parallle.py module to enable 
automatic parallelization for Numba via the parallel=True option. Note that
the parallel option in Numba can be very sensitive to environment setup. If you
are experiencing Kernal failures try turning the parallel option off. Also, the
parameter iuse_numerical_rift should not be set to zero (i.e. analytical) if 
using the parallel option (i.e. iforce_serial = 0). Only the numerical rifting
model (iuse_numerical_rift=1) works when using the parallel option.

Running the Colorado Basin Example
-------------------------------------------------------------------------------
Update the paths and then execute the following script using the Anaconda 
python executable:
  
    BatchScripts\RUN_BATCH_ColoradoBasin.py

Developer Notes
-------------------------------------------------------------------------------
Refactoring To-Do List 
*  Replace tops_list_bs and event_dict_bs with mutable (structref) and immutable (named tuples) data structures.
*  Convert classes used in calculations to Numba structref classes or immutable named tuples to eliminate the need to pass 
large numbers of arguments to jitted functions.
*  Decouple large functions and classes and increase separation of responsibilities. Coupling is a relict associated with optimizing a 
pure-Python/Numpy version prior to incorporating Numba.   
* Refactor large modules into packages with submodules that have fewer lines of code.

Output Maps and CSV files
-------------------------------------------------------------------------------
The IPAtoolkit produces a large number of compressed output maps in ZMAP format 
and output csv files depending on output options. Maps and CSV files are 
automatically sent to hard coded directories. ZMaps can be plotted using the 
ZMap_View tool in the prototype UI. The following section provides a brief 
description of directories, output maps and csv files:

Charge_History_ZMaps
* Charge_bGOR_g_g: bulk GOR maps for post-trap charge in g/g
* Charge_bGOR_scf_bbl: bulk GOR maps for post-trap charge in scf/bbl
* Charge_Gas_Tg: Post-trap expulsed gas in terragrams
* Charge_Oil_Tg: Post-trap expulsed oil in terragrams
* TRAP_fpress_psi: Hydrostatic pressure of trap layer through time in psi
* TRAP_tempC: Temperature (C) of trap layer through time
* TRAP_gasGOR_scf_bbl: GOR of gas at pressure conditions of trap layer in scf/bbl
* TRAP_oilGOR_scf_bbl: GOR of oil at pressure and temperature of trap layer in scf/bbl
* TRAP_iphase: maps with integer ID's specifying saturation state of post-trap 
expulsed fluids at reservoir conditions assuming vertical migration: 
    * 0 = single phase undersaturated oil
    * 1 = deep single phase
    * 2 = saturated dual phase oil and gas
    * 3 = single phase gas

Flexure_Output
* BMST_flex_iter: Flexurally restored basement
* Flex_Te_clean: Cleaned effective elastic thickness
* Flex_Te_m: effective elastic thickness (pre-cleaning)
* Flex_Crustal_Load_MPa: Crustal load in MPa
* Flex_delta_UC_Load_MPa: Change in crustal load in MPa
* Flex_Sed_Load_MPa: Sediment load in MPa
* Flex_delta_Sed_Load_MPa: Change in sediment load in MPa
* Flex_Thermal_Load_MPa: Thermal load in MPa
* Flex_delta_Thermal_Load_MPa: Change in thermal load in MPa
* Flex_Flex_w_inc_iter: Incremental flexural deflection on large grid with taper
* Flex_w_inc_iter: Incremental flexural deflection on small grid without taper

GOR_Bulk
* Maps with cumulative and incremental bulk gas-oil-ratios in g/g and scf/bbl, 
  total gas in terragrams, total oil in terragrams and secondary gas in 
  terragrams based on integrated post-trap expulsion from multiple source 
  rocks.
* Maps are also provided for integrated values within migration polygons, 
  average API, bulk API, primary oil mass fractions and average 
  transformation ratio. 
* The term "VERTICAL" refers to vertical migration, "t" refers to total, "p" 
  refers to primary and "sec" refers to secondary
  
GOR_Yield
* Maps with cumulative gas-oil ratios (GOR) in g/g and scf/bbl expulsed 
  primary and total hydrocarbons (primary + secondary + reactive residual)
  
SRC_Yield_Cumulative
* Maps with cumulative expulsed primary dissolved gas (PDS), primary free gas 
  (PFG), primary oil (PO) and secondary free gas (SFG) for each source layer in
  terragrams
  
SRC_Yield_Incremental
* Maps with incremental expulsed primary dissolved gas (PDS), primary free gas 
  (PFG), primary oil (PO) and secondary free gas (SFG) for each source layer in
  terragrams
  
Trap_Charge_History
* Maps and output files from charge-to-trap calculations including reservoir 
condition maps, GOR maps and csv files with integrated charge history at traps

Crustal_Structure
* best_fit_deltas.dat: Zmap of best fit crustal stretching factors (unitless)
* best_fit_xth.dat: ZMap with calculated isostatic crustal thickness (meters)
* Moho.dat: ZMap with Moho depth (meters)

Depth_ZMaps
* Initial and updated maps based on paleo-water depth inversion

EasyRo_Zmaps
* ZMaps with Easy Ro maturity calculations

LOM_ZMaps
* LOM (Level of Organic Maturity) maturity maps

LOM_AGE
* Maps showing the age at which a layer reached different LOM values

LOM_Depth
* Maps showing the present-day depth to different LOM values

Transformation_Ratio_ZMaps
* Transformation ratio ZMaps for primary generation

HC_Mass_Gen_Zmaps
* Total primary hydrocarbon mass generation in mgHC/gTOC

FreeGas_Pore_Expulsion_Zmaps
* Primary free gas pore expulsion in mg HC/gTOC

Liquid_Pore_Expulsion_ZMaps
* Primary liquid (Oil + Dissolved Gas) pore expulsion in mgHC/gTOC

Pore_Expulsion_Rate_ZMaps
* Rate of primary pore expulsion in mgHC/gTOC/Myr

Secondary_FreeGas_Pore_Epulsion_ZMaps
* Free gas pore expulsion associated with secondary cracking of residual oil 
  in mgHC/gTOC

Secondary_Pore_Expulsion_Rate_ZMaps
* Rate of free gas pore expulsion associated with secondary cracking of 
  residual oil in mgHC/gTOC/Myr

HeatFlow_ZMaps
* Calculated crustal heat flow maps in mW/m/m including total heat flow maps 
  that and heat flow anomaly maps (i.e. transient heat flow).

PWD_ZMaps
* Calculated paleo-water depth ZMaps.
* PWD_Flex_iter: Restored paleo-water depth before low-pass filter is applied
* PWD_Flex_LPF_iterID: Restored paleo-water depth with a low pass filter 
  applied from flexural_backstripping_method1()
* PWD_Flex_WL_iterID: Water loaded restored paleo-water depth from 
  flexural_backstripping_method1()
* PWD_Flex_wBL_iterID: Restored paleo-water depth with base-level shifts from 
  flexural_backstripping_method1()
* PWD_LocalIsostasy: Restored paleo-water depth with local isostasy and 
  residual subsidence using calculate_res_sub_and_corrected_PWD()
                     
Residual_Subsidence
* Residual subsidence ZMaps

Salt_Reconstruction
* Salt thickness maps
* Salt_Thick_fromFLEX_iter: Salt thickness restoration from flexure. 

Temperature_ZMaps
* Temperature ZMaps in degrees C

Thermotectonic_Subsidence
* Forward and backstripped thermotectonic subsidence assuming local isostasy

Total_Source_Expulsion
* CSV files containing cumulative and incremental source expulsion volumes in 
  Tcf or GOB at each model age

Well_Extractions
* CSV files containing model information extracted at well locations.


-- Erik A. Kneller & David J. Gombosi 2022
