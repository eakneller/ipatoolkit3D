Attribute VB_Name = "Model_IO_Tools_IPA3D"
Sub SaveCSV(sheet_with_input_path As String, cell_input_path As String, _
            output_csv As String, sheet_tmp_csv As String, _
            range_tmp_csv As String, row_max As Integer, _
            col_max As Integer)

Dim ColNum As Integer
Dim RowNum As Integer
Dim Line As String
Dim LineValues() As Variant
Dim OutputFileNum As Integer
Dim SheetValues() As Variant

input_path = Sheets(sheet_with_input_path).Range(cell_input_path).Value
output_csv_path = input_path & "\" & output_csv
Debug.Print "output_csv_path : " & output_csv_path

OutputFileNum = FreeFile
Open output_csv_path For Output Lock Write As #OutputFileNum

SheetValues = Sheets(sheet_tmp_csv).Range(range_tmp_csv).Value
ReDim LineValues(1 To col_max)

For RowNum = 1 To row_max
  For ColNum = 1 To col_max
    If IsError(SheetValues(RowNum, ColNum)) Then
        LineValues(ColNum) = 0
    Else
        LineValues(ColNum) = SheetValues(RowNum, ColNum)
    End If
  Next
  Line = Join(LineValues, ",")
  Print #OutputFileNum, Line
Next

Close OutputFileNum

End Sub
Sub ExecuteScript1D()

Application.ScreenUpdating = False
Application.EnableEvents = False

'RetVal = Shell("C:\ProgramData\Anaconda3\python.exe " & "C:\Users\eaknell\Desktop\IPA3D_v1_Local\IPA3D_RUNIT.py.")

Dim wsh As Object
Set wsh = VBA.CreateObject("WScript.Shell")
Dim waitOnReturn As Boolean: waitOnReturn = True
Dim windowStyle As Integer: windowStyle = 0
Dim py_cmd, py_exe_path, ipa1D_input_file_path As String
Dim arg1, arg2, arg3, runit_path, log_file_path As String

Dim PW As String

' Error handling
On Error GoTo eh

Application.StatusBar = "Progress % :" & Format(0, "0%")

PW = "ibaw123A"
Sheets("IPA1Dpy").Unprotect Password:=PW

py_exe_path = Worksheets("Set Paths").Range("C11").Value

' Scripts path
arg1 = Worksheets("Set Paths").Range("C12").Value

runit_path = arg1 & "\IPA1D_RUNIT.py"

' Save IPA input sheet
file_path = arg1 & "\ipa_1D\IPA1Dinput.csv"
'Debug.Print "file_path : " & file_path

OutputFileNum = FreeFile

'Open PathName & fNAME For Output Lock Write As #OutputFileNum
Open file_path For Output Lock Write As #OutputFileNum

SheetValues = Sheets("IPA1Dinput").Range("A1:AZ1171").Value
ReDim LineValues(1 To 52)

For RowNum = 1 To 1171
  For ColNum = 1 To 52
    If IsError(SheetValues(RowNum, ColNum)) Then
        LineValues(ColNum) = 0
    Else
        LineValues(ColNum) = SheetValues(RowNum, ColNum)
    End If
  Next
  Line = Join(LineValues, ",")
  Print #OutputFileNum, Line
Next

Close OutputFileNum

Application.StatusBar = "Progress % :" & Format(0.25, "0%")

' Define command
cmd = "cmd.exe /c " & py_exe_path & " " & runit_path & " " & arg1
'Debug.Print cmd

wsh.Run cmd, windowStyle, waitOnReturn

Application.StatusBar = "Progress % :" & Format(0.95, "0%")

strpath5 = arg1 & "\ipa_1D\output\" & "nodei1j1__TVDssm_History.csv"

Dim wbkS5 As Workbook
Dim wshS5 As Worksheet
Set wbkS5 = Workbooks.Open(Filename:=strpath5)
Set wshS5 = wbkS5.Worksheets(1)

Application.ScreenUpdating = False
wshS5.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("IPA1Doutput").Range("A1")

wbkS5.Close SaveChanges:=False

Range("D4").Value = "Up-to-date"

Application.StatusBar = "Progress % :" & Format(1, "0%")

Sheets("IPA1Dpy").Protect Password:=PW
Application.ScreenUpdating = True
Application.EnableEvents = True

Done:
    Exit Sub
eh:
    ' Protect worksheet
    Sheets("IPA1Dpy").Protect Password:=PW
    Application.EnableEvents = True
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub ExecuteScript()

Application.ScreenUpdating = False

Dim wsh As Object
Set wsh = VBA.CreateObject("WScript.Shell")
Dim waitOnReturn As Boolean: waitOnReturn = True
Dim windowStyle As Integer: windowStyle = 1
Dim py_cmd, py_exe_path As String
Dim arg1, arg2, arg3, runit_path, log_file_path As String
Dim arg4, arg5, arg6 As String

SaveModel2

py_exe_path = Worksheets("Model Input").Range("G1").Value
' Scripts path
arg1 = Worksheets("Model Input").Range("G2").Value
' input path
arg2 = Worksheets("Model Input").Range("G3").Value
' output path
arg3 = Worksheets("Model Input").Range("G4").Value
arg4 = "single" ' multirun_stype "single" = do not do a multirun
arg5 = "1" ' number of Monte Carlo runs
arg6 = "1" ' "0" = send output to terminal "1" = send to log file via python

log_file_path = arg3 & "\log.txt"

runit_path = arg1 & "\IPA3D_RUNIT.py"

bat_runit_path = arg1 & "\runit.bat"
cmd = "cmd.exe /c " & bat_runit_path & " " & py_exe_path & " " & runit_path _
      & " " & arg1 & " " & arg2 & " " & arg3 & " " & arg4 & " " _
      & arg5 & " " & arg6 ' & " " & log_file_path

Debug.Print cmd
wsh.Run cmd, windowStyle, waitOnReturn

Application.ScreenUpdating = True
MsgBox "IPA is done. ZMaps are located in the following directory: " & arg3

End Sub

Sub ExecuteScript_Risk()

Application.ScreenUpdating = False

Dim wsh As Object
Set wsh = VBA.CreateObject("WScript.Shell")
Dim waitOnReturn As Boolean: waitOnReturn = True
Dim windowStyle As Integer: windowStyle = 1
Dim py_cmd, py_exe_path As String
Dim arg1, arg2, arg3, runit_path, log_file_path As String
Dim arg4, arg5, arg6 As String

SaveModel2

py_exe_path = Worksheets("Model Input").Range("G1").Value

' Scripts path
arg1 = Worksheets("Model Input").Range("G2").Value
' input path
arg2 = Worksheets("Model Input").Range("G3").Value
' output path
arg3 = Worksheets("Model Input").Range("G4").Value
arg4 = Worksheets("Risk&Sensitivity").Range("C2").Value
arg5 = Worksheets("Risk&Sensitivity").Range("C3").Value
arg6 = "1" ' "0" = send output to terminal "1" = send to log file via python

log_file_path = arg3 & "\log.txt"

runit_path = arg1 & "\IPA3D_RUNIT.py"

bat_runit_path = arg1 & "\runit.bat"
cmd = "cmd.exe /c " & bat_runit_path & " " & py_exe_path & " " & runit_path _
      & " " & arg1 & " " & arg2 & " " & arg3 & " " & arg4 & " " & arg5 _
      & " " & arg6 ' & " " & log_file_path

Debug.Print cmd

wsh.Run cmd, windowStyle, waitOnReturn

Application.ScreenUpdating = True
MsgBox "IPA is done. ZMaps are located in the following directory: " & arg3

End Sub

Sub ExecuteScript_Scenario()

Application.ScreenUpdating = False

Dim wsh As Object
Set wsh = VBA.CreateObject("WScript.Shell")
Dim waitOnReturn As Boolean: waitOnReturn = True
Dim windowStyle As Integer: windowStyle = 1
Dim py_cmd, py_exe_path As String
Dim arg1, arg2, arg3, runit_path, log_file_path As String
Dim arg4, arg5, arg6 As String

SaveModel2

py_exe_path = Worksheets("Model Input").Range("G1").Value

' Scripts path
arg1 = Worksheets("Model Input").Range("G2").Value
' input path
arg2 = Worksheets("Model Input").Range("G3").Value
' output path
arg3 = Worksheets("Model Input").Range("G4").Value
arg4 = "scenario"
arg5 = "2"
arg6 = "1" ' "0" = send output to terminal "1" = send to log file via python

log_file_path = arg3 & "\log.txt"

runit_path = arg1 & "\IPA3D_RUNIT.py"

bat_runit_path = arg1 & "\runit.bat"
cmd = "cmd.exe /c " & bat_runit_path & " " & py_exe_path & " " & runit_path _
      & " " & arg1 & " " & arg2 & " " & arg3 & " " & arg4 & " " & arg5 _
      & " " & arg6 ' & " " & log_file_path

Debug.Print cmd

wsh.Run cmd, windowStyle, waitOnReturn

Application.ScreenUpdating = True
MsgBox "IPA is done. ZMaps are located in the following directory: " & arg3

End Sub

Sub FindRiskMapFiles()

Dim fNAME As String
Dim fldr As FileDialog
Dim sItem, file_path As String
Dim strFolder As String
Dim strPattern As String
Dim strFile As String
Dim irow, icol, irow_ini, icount As Integer

' Error handling
On Error GoTo eh

'''''''''''''''''
' Get folder path
'''''''''''''''''
Application.ScreenUpdating = False

sItem = Sheets("Model Input").Range("G3").Value
' Only export file if folder path has been defined
If sItem <> "" Then
    Worksheets("Risk&Sensitivity").Range("J7:J206").Value = ""
    Debug.Print "Folder path : " & sItem
    strPattern = "*.dat"
    strFolder = sItem & "\risk_maps\"
    strFile = Dir(strFolder & strPattern, vbNormal)
    irow_ini = 7 ' row 7 is the initial row
    icol = 10 ' column N is the target column
    Do While Len(strFile) > 0
        irow = irow_ini + icount
        icount = icount + 1
        Worksheets("Risk&Sensitivity").Cells(irow, icol).Value = strFile
        strFile = Dir
    Loop
Else
    MsgBox "No folder selected"
End If

Application.ScreenUpdating = True
MsgBox "Risk ZMAPs have been loaded. You can now select them in drop down menus."

Done:
    Exit Sub
eh:
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True
    
End Sub

Sub FindInputMapFiles()

Dim fNAME As String
Dim fldr As FileDialog
Dim sItem, file_path As String
Dim strFolder As String
Dim strPattern As String
Dim strFile As String
Dim irow, icol, irow_ini, icount As Integer

' Error handling
On Error GoTo eh

'''''''''''''''''
' Get folder path
'''''''''''''''''

Application.ScreenUpdating = False
sItem = Sheets("Model Input").Range("G3").Value

' Only export file if folder path has been defined
If sItem <> "" Then
    Worksheets("Model Input").Range("R7:R206").Value = ""
    Debug.Print "Folder path : " & sItem
    strPattern = "*.dat"
    strFolder = sItem & "\"
    strFile = Dir(strFolder & strPattern, vbNormal)
    irow_ini = 7 ' row 7 is the initial row
    icol = 18 ' column N is the target column
    Do While Len(strFile) > 0
        irow = irow_ini + icount
        icount = icount + 1
        Worksheets("Model Input").Cells(irow, icol).Value = strFile
        strFile = Dir
    Loop
Else
    MsgBox "No folder selected"
End If

Application.ScreenUpdating = True
MsgBox "ZMAPs have been loaded. You can now select them in drop down menus."

Done:
    Exit Sub
eh:
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True
    
End Sub

Sub FindAsciiPolyFiles()

Dim fNAME As String
Dim fldr As FileDialog
Dim sItem, file_path As String
Dim strFolder As String
Dim strPattern As String
Dim strFile As String
Dim irow, icol, irow_ini, icount As Integer

' Error handling
On Error GoTo eh

'''''''''''''''''
' Get folder path
'''''''''''''''''

Application.ScreenUpdating = False
sItem = Sheets("Model Input").Range("G3").Value

' Only export file if folder path has been defined
If sItem <> "" Then
    Worksheets("Model Input").Range("T7:T206").Value = ""
    Debug.Print "Folder path : " & sItem
    strPattern = "*.poly"
    strFolder = sItem & "\"
    strFile = Dir(strFolder & strPattern, vbNormal)
    irow_ini = 7 ' row 7 is the initial row
    icol = 20 ' column N is the target column
    Do While Len(strFile) > 0
        irow = irow_ini + icount
        icount = icount + 1
        Worksheets("Model Input").Cells(irow, icol).Value = strFile
        strFile = Dir
    Loop
Else
    MsgBox "No folder selected"
End If

Application.ScreenUpdating = True
MsgBox "ASCII .poly files have been loaded. You can now select them in drop down menus."

Done:
    Exit Sub
eh:
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True
    
End Sub

Sub LoadModel()

' Error handling
On Error GoTo eh

Application.ScreenUpdating = False

'''''''''''''''''''''''
' Load loaded maps list
'''''''''''''''''''''''
strpath1 = Sheets("Model Input").Range("G3").Value & "\ipa_loaded_maps.csv"
Dim wbkS1 As Workbook
Dim wshS1 As Worksheet
Set wbkS1 = Workbooks.Open(Filename:=strpath1)
Set wshS1 = wbkS1.Worksheets(1)
wshS1.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ipa_loaded_maps_tmp").Range("A1")
wbkS1.Close SaveChanges:=False
Sheets("ipa_loaded_maps_tmp").Range("A1:A200").Copy
Sheets("Model Input").Range("R7:R206").PasteSpecial Paste:=xlPasteValues
'''''''''''''''''''''''
' Load loaded polys list
'''''''''''''''''''''''
Debug.Print "C1"
strpathA = Sheets("Model Input").Range("G3").Value & "\ipa_loaded_polys.csv"
Dim strFileExists As String
strFileExists = Dir(strpathA)
If strFileExists = "" Then
    Debug.Print "Poly file not found."
Else
    Dim wbkSA As Workbook
    Dim wshSA As Worksheet
    Set wbkSA = Workbooks.Open(Filename:=strpathA)
    Set wshSA = wbkSA.Worksheets(1)
    wshSA.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ipa_loaded_polys_tmp").Range("A1")
    wbkSA.Close SaveChanges:=False
    Sheets("ipa_loaded_polys_tmp").Range("A1:A200").Copy
    Sheets("Model Input").Range("T7:T206").PasteSpecial Paste:=xlPasteValues
End If
''''''''''''''''''''''
' Load main input file
''''''''''''''''''''''
strpath2 = Sheets("Model Input").Range("G3").Value & "\ipa_input.csv"
Dim wbkS2 As Workbook
Dim wshS2 As Worksheet
Set wbkS2 = Workbooks.Open(Filename:=strpath2)
Set wshS2 = wbkS2.Worksheets(1)
wshS2.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ipa_input_tmp").Range("A1")
wbkS2.Close SaveChanges:=False
' Stratigraphic inputs
Sheets("ipa_input_tmp").Range("B3:G32").Copy
Sheets("Model Input").Range("B7:G36").PasteSpecial Paste:=xlPasteValues
' Stratigraphic inputs part 2
Sheets("ipa_input_tmp").Range("H3:N32").Copy
Sheets("Model Input").Range("H7:N36").PasteSpecial Paste:=xlPasteValues
' Time-to-depth conversion
Sheets("ipa_input_tmp").Range("B47:B48").Copy
Sheets("Model Input").Range("B40:B41").PasteSpecial Paste:=xlPasteValues
' Compaction Options
Sheets("ipa_input_tmp").Range("B49").Copy
Sheets("Model Input").Range("B44").PasteSpecial Paste:=xlPasteValues
' Background heat flow
Sheets("ipa_input_tmp").Range("B46").Copy
Sheets("Model Input").Range("B47").PasteSpecial Paste:=xlPasteValues
' Stretching factor inversion
Sheets("ipa_input_tmp").Range("B53:B55").Copy
Sheets("Model Input").Range("B50:B52").PasteSpecial Paste:=xlPasteValues
' Lithospheric stretching model maps
Sheets("ipa_input_tmp").Range("B33:B45").Copy
Sheets("Model Input").Range("B55:B67").PasteSpecial Paste:=xlPasteValues
' Lithospheric stretching model parameters
Sheets("ipa_input_tmp").Range("B56:B57").Copy
Sheets("Model Input").Range("B70:B71").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_input_tmp").Range("B64:B70").Copy
Sheets("Model Input").Range("B72:B78").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_input_tmp").Range("B58:B61").Copy
Sheets("Model Input").Range("B79:B82").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_input_tmp").Range("B71:B75").Copy
Sheets("Model Input").Range("B83:B87").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_input_tmp").Range("B62:B63").Copy
Sheets("Model Input").Range("B88:B89").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_input_tmp").Range("B76:B77").Copy
Sheets("Model Input").Range("B90:B91").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_input_tmp").Range("B78:B79").Copy
Sheets("Model Input").Range("B92:B93").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_input_tmp").Range("B50:B52").Copy
Sheets("Model Input").Range("B96:B98").PasteSpecial Paste:=xlPasteValues
' Thermal calculation Parameters
Sheets("ipa_input_tmp").Range("B80:B94").Copy
Sheets("Model Input").Range("B101:B115").PasteSpecial Paste:=xlPasteValues
' Mass Generation
Sheets("ipa_input_tmp").Range("B95:B96").Copy
Sheets("Model Input").Range("B118:B119").PasteSpecial Paste:=xlPasteValues
' Map Output
Sheets("ipa_input_tmp").Range("B97:B101").Copy
Sheets("Model Input").Range("B122:B126").PasteSpecial Paste:=xlPasteValues
' Other calculation Options
Sheets("ipa_input_tmp").Range("B102:B104").Copy
Sheets("Model Input").Range("B129:B131").PasteSpecial Paste:=xlPasteValues
' Salt restoration and low-pass filter
Sheets("ipa_input_tmp").Range("B105:B107").Copy
Sheets("Model Input").Range("B134:B136").PasteSpecial Paste:=xlPasteValues
' Flexure
Sheets("ipa_input_tmp").Range("B122:B125").Copy
Sheets("Model Input").Range("B152:B155").PasteSpecial Paste:=xlPasteValues
' Other Map Output Options
Sheets("ipa_input_tmp").Range("B126:B135").Copy
Sheets("Model Input").Range("B158:B167").PasteSpecial Paste:=xlPasteValues
' Variable Gas Fraction
Sheets("ipa_input_tmp").Range("B136:B139").Copy
Sheets("Model Input").Range("B170:B173").PasteSpecial Paste:=xlPasteValues
' Other Parameters
Sheets("ipa_input_tmp").Range("B140:B141").Copy
Sheets("Model Input").Range("B176:B177").PasteSpecial Paste:=xlPasteValues
'''''''''''''''''''''
' Load lithology file
'''''''''''''''''''''
strpath3 = Sheets("Model Input").Range("G3").Value & "\ipa_lithology.csv"
Dim wbkS3 As Workbook
Dim wshS3 As Worksheet
Set wbkS3 = Workbooks.Open(Filename:=strpath3)
Set wshS3 = wbkS3.Worksheets(1)
wshS3.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ipa_lithology_tmp").Range("A1")
wbkS3.Close SaveChanges:=False
Sheets("ipa_lithology_tmp").Range("A2:W101").Copy
Sheets("RockProps").Range("B4:X103").PasteSpecial Paste:=xlPasteValues
''''''''''''''''
' Load well file
''''''''''''''''
strpath4 = Sheets("Model Input").Range("G3").Value & "\ipa_wells.csv"
Dim wbkS4 As Workbook
Dim wshS4 As Worksheet
Set wbkS4 = Workbooks.Open(Filename:=strpath4)
Set wshS4 = wbkS4.Worksheets(1)
wshS4.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ipa_wells_tmp").Range("A1")
wbkS4.Close SaveChanges:=False
Sheets("ipa_wells_tmp").Range("A1:F200").Copy
Sheets("Well_List").Range("C3:H202").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_wells_tmp").Range("H1:H200").Copy
Sheets("Well_List").Range("J3:J202").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_wells_tmp").Range("G1:G200").Copy
Sheets("Calibration_Viewer").Range("D30:D229").PasteSpecial Paste:=xlPasteValues
'''''''''''''''''''''''
' Load calibration file
'''''''''''''''''''''''
strpath5 = Sheets("Model Input").Range("G3").Value & "\ipa_calibration.csv"
Dim wbkS5 As Workbook
Dim wshS5 As Worksheet
Set wbkS5 = Workbooks.Open(Filename:=strpath5)
Set wshS5 = wbkS5.Worksheets(1)
wshS5.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ipa_calibration_tmp").Range("A1")
wbkS5.Close SaveChanges:=False
Sheets("ipa_calibration_tmp").Range("C8:C1007").Copy
Sheets("Calibration_Data").Range("C8:C1007").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_calibration_tmp").Range("F8:M1007").Copy
Sheets("Calibration_Data").Range("F8:M1007").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_calibration_tmp").Range("O8:O1007").Copy
Sheets("Calibration_Data").Range("O8:O1007").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_calibration_tmp").Range("R8:Y1007").Copy
Sheets("Calibration_Data").Range("R8:Y1007").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_calibration_tmp").Range("AI8:AM1007").Copy
Sheets("Calibration_Data").Range("AI8:AM1007").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_calibration_tmp").Range("AO8:AO1007").Copy
Sheets("Calibration_Data").Range("AO8:AO1007").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_calibration_tmp").Range("AQ8:AU1007").Copy
Sheets("Calibration_Data").Range("AQ8:AU1007").PasteSpecial Paste:=xlPasteValues
Sheets("ipa_calibration_tmp").Range("AW8:BF1007").Copy
Sheets("Calibration_Data").Range("AW8:BF1007").PasteSpecial Paste:=xlPasteValues
'''''''''''''''''''''''
' Load traps file
'''''''''''''''''''''''
strpath6 = Sheets("Model Input").Range("G3").Value & "\ipa_traps.csv"
strFileExists = Dir(strpath6)
If strFileExists = "" Then
    Debug.Print "Traps file not found."
Else

    Dim wbkS6 As Workbook
    Dim wshS6 As Worksheet
    Set wbkS6 = Workbooks.Open(Filename:=strpath6)
    Set wshS6 = wbkS6.Worksheets(1)
    wshS6.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ipa_traps_tmp").Range("B5")
    wbkS6.Close SaveChanges:=False
    Sheets("ipa_traps_tmp").Range("C5:C23").Copy
    Sheets("Traps").Range("D5:D23").PasteSpecial Paste:=xlPasteValues
    Sheets("ipa_traps_tmp").Range("C27:N56").Copy
    Sheets("Traps").Range("D27:O56").PasteSpecial Paste:=xlPasteValues
    Sheets("ipa_traps_tmp").Range("C60:I1059").Copy
    Sheets("Traps").Range("D60:J1059").PasteSpecial Paste:=xlPasteValues
End If

Sheets("Model Input").Activate
Application.ScreenUpdating = True
MsgBox "Model csv files located in input directory have been loaded."

Done:
    Exit Sub
eh:
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True

End Sub

Sub SaveModel2()

Dim sheet_with_input_path As String
Dim cell_input_path As String
Dim output_csv As String
Dim sheet_tmp_csv As String
Dim range_tmp_csv As String
Dim row_max As Integer
Dim col_max As Integer

' Error handling
On Error GoTo eh

active_sheet_name = ActiveSheet.Name
Debug.Print "Saving Model Files to Input Directory ...."
Application.ScreenUpdating = False

''''''''''''''''''''''''''''''''
' Transfer data to ipa_input_tmp
''''''''''''''''''''''''''''''''
Sheets("Model Input").Range("A5:N36").Copy
Sheets("ipa_input_tmp").Range("A1:N32").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B40:B41").Copy
Sheets("ipa_input_tmp").Range("B47:B48").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B44").Copy
Sheets("ipa_input_tmp").Range("B49").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B47").Copy
Sheets("ipa_input_tmp").Range("B46").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B50:B52").Copy
Sheets("ipa_input_tmp").Range("B53:B55").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B55:B67").Copy
Sheets("ipa_input_tmp").Range("B33:B45").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B70:B71").Copy
Sheets("ipa_input_tmp").Range("B56:B57").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B72:B78").Copy
Sheets("ipa_input_tmp").Range("B64:B70").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B79:B82").Copy
Sheets("ipa_input_tmp").Range("B58:B61").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B83:B87").Copy
Sheets("ipa_input_tmp").Range("B71:B75").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B88:B89").Copy
Sheets("ipa_input_tmp").Range("B62:B63").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B90:B91").Copy
Sheets("ipa_input_tmp").Range("B76:B77").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B92:B93").Copy
Sheets("ipa_input_tmp").Range("B78:B79").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B96:B98").Copy
Sheets("ipa_input_tmp").Range("B50:B52").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B101:B115").Copy
Sheets("ipa_input_tmp").Range("B80:B94").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B118:B119").Copy
Sheets("ipa_input_tmp").Range("B95:B96").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B122:B126").Copy
Sheets("ipa_input_tmp").Range("B97:B101").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B129:B131").Copy
Sheets("ipa_input_tmp").Range("B102:B104").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B134:B150").Copy
Sheets("ipa_input_tmp").Range("B105:B121").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B152:B155").Copy
Sheets("ipa_input_tmp").Range("B122:B125").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B158:B167").Copy
Sheets("ipa_input_tmp").Range("B126:B135").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B170:B173").Copy
Sheets("ipa_input_tmp").Range("B136:B139").PasteSpecial Paste:=xlPasteValues
Sheets("Model Input").Range("B176:B177").Copy
Sheets("ipa_input_tmp").Range("B140:B141").PasteSpecial Paste:=xlPasteValues
'''''''''''''''''''''''''''
' Save to ipa_input.csv
'''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "ipa_input.csv"
sheet_tmp_csv = "ipa_input_tmp"
range_tmp_csv = "A1:N141"
row_max = 141
col_max = 14
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''
' Transfer data to ipa_lithology_tmp
''''''''''''''''''''''''''''''''''''
Sheets("RockProps").Range("B3:X103").Copy
Sheets("ipa_lithology_tmp").Range("A1:W101").PasteSpecial Paste:=xlPasteValues
''''''''''''''''''''''''''''''''''''
' Save data to ipa_lithology.csv
''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "ipa_lithology.csv"
sheet_tmp_csv = "ipa_lithology_tmp"
range_tmp_csv = "A1:W101"
row_max = 101
col_max = 23
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''''''
' Save poly list to ipa_loaded_polys.csv
''''''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "ipa_loaded_polys.csv"
sheet_tmp_csv = "Model Input"
range_tmp_csv = "T7:T206"
row_max = 200
col_max = 1
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''''
' Save map list to ipa_loaded_maps.csv
''''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "ipa_loaded_maps.csv"
sheet_tmp_csv = "Model Input"
range_tmp_csv = "R7:R206"
row_max = 200
col_max = 1
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''''
' Save well list to ipa_wells.csv
''''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "ipa_wells.csv"
sheet_tmp_csv = "Well_List"
range_tmp_csv = "C3:J202"
row_max = 200
col_max = 8
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''''''''''''
' Save calibration data to ipa_calibration.csv
''''''''''''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "ipa_calibration.csv"
sheet_tmp_csv = "Calibration_Data"
range_tmp_csv = "A1:BF1007"
row_max = 1007
col_max = 58
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''''''''''''''''
' Save Monte Carlo Inputs to Risk_Sensitivity.csv
''''''''''''''''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "Risk_Sensitivity.csv"
sheet_tmp_csv = "Risk&Sensitivity"
range_tmp_csv = "A1:H45"
row_max = 45
col_max = 8
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''''''''''''''''
' Save Batch Run Inputs to BatchRun.csv
''''''''''''''''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "BatchRun.csv"
sheet_tmp_csv = "BatchRunInput"
range_tmp_csv = "A1:M33"
row_max = 33
col_max = 13
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
''''''''''''''''''''''''''''''''''''
' Transfer data to ipa_traps_tmp
''''''''''''''''''''''''''''''''''''
Sheets("Traps").Range("A1:O1059").Copy
Sheets("ipa_traps_tmp").Range("A1:O1059").PasteSpecial Paste:=xlPasteValues
''''''''''''''''''''''''''''''''''''
' Save data to ipa_traps.csv
''''''''''''''''''''''''''''''''''''
sheet_with_input_path = "Model Input"
cell_input_path = "G3"
output_csv = "ipa_traps.csv"
sheet_tmp_csv = "ipa_traps_tmp"
range_tmp_csv = "A1:O1059"
row_max = 1059
col_max = 15
Call SaveCSV(sheet_with_input_path, cell_input_path, _
             output_csv, sheet_tmp_csv, _
             range_tmp_csv, row_max, _
             col_max)
             
Sheets(active_sheet_name).Activate
Application.ScreenUpdating = True
Done:
    Exit Sub
eh:
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True
End Sub
