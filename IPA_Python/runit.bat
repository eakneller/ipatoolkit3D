@ECHO OFF

ECHO "#########################################################################"
ECHO "Start of IPA runit.bat script"
ECHO "#########################################################################"

ECHO ">> Setting variables"
set exe_path=%1
set script_path=%2
set ipa_python_path=%3
set input_path=%4%
set output_path=%5%
set run_stype=%6%
set nruns=%7%
set ioutput=%8%
:: set log_file_path=%9%

For %%A in ("%exe_path%") do (
    set act_root=%%~dpA
)

set act_path=%act_root%Scripts\activate.bat

call %act_path%

ECHO ">> Running Python Anaconda....."

%exe_path% %script_path% %ipa_python_path% %input_path% %output_path% %run_stype% %nruns% %ioutput%

:: > %log_file_path%

::PAUSE
::cmd.exe /c %exe_path% %script_path% %ipa_python_path% %input_path% %output_path% > %log_file_path%

ECHO "#########################################################################"
ECHO "End of IPA runit.bat script"
ECHO "#########################################################################"