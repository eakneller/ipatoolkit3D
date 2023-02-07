@ECHO OFF

ECHO "#########################################################################"
ECHO "Start of RUN_BATCH.bat script"
ECHO "#########################################################################"

ECHO ">> Setting variables"
::#******************************************************************************
::# Use this bat file to run IPA in the bakcground on Windows machines.
::#******************************************************************************
::# Input Descriptions
::#******************************************************************************
::# run_stype: Type of multi-scenario run
::#	Options
::#     -------------------------------------------------------------------------
::# 	single      : only one run
::#
::# 	highlow     : run high and lowside scenarios using delta_min and delta_max 
::#                 parameters for variable keys in HL_dict. Number of runs is 
::#                 determined from total number of combinations. Note that 
::#                 key enteries can be commented out in HL_dict below to 
::#                 exclude parameters.
::#
::# 	monte_carlo : run a Monte Carlo simulations using delta_min, delta_max
::#                 parameters, pdf type and distribution parameters for each
::#                 variable. Number of runs is defined by nruns.
::#
::# 	scenario    : Run a discrete number of scenarios based on the size of list
::#                 associated with the scenario_delta_dict dictionary.
::#
::# ioutput: standard output control
::#     Options
::#     -------------------------------------------------------------------------- 
::# 	0 = send output to standard terminal and redirect to file
::#	1 = send output directly to file
::# 
::# nruns: number of Monte Carlo runs
::#
::********************************************************************************

:: Python Anaconda executable path
set exe_path=C:\ProgramData\Anaconda3\python.exe

:: IPAtoolkit3D directory path
set ipa_main_path=C:\Users\eaknell\Desktop\ipatoolkit3D-main

:: Main IPA input directory located at main IPA path
set input_dir=Input_Maps\ColoradoBasin

:: Path where output will be sent
set output_path=C:\Users\eaknell\Desktop\ipatoolkit3D-main\Output_Maps

set run_stype=single
set nruns=1
set ioutput=1

set ipa_python_path=%ipa_main_path%\IPA_Python
set script_path=%ipa_python_path%\IPA3D_RUNIT.py
set input_path=%ipa_main_path%\%input_dir%

ECHO ">> Running Python Anaconda....."
%exe_path% %script_path% %ipa_python_path% %input_path% %output_path% %run_stype% %nruns% %ioutput%

ECHO "#########################################################################"
ECHO "End of RUN_BATCH.bat script"
ECHO "#########################################################################"

PAUSE
