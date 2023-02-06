Attribute VB_Name = "Calibration_Tools_IPA3D"
Sub ClearPlot1()

Application.ScreenUpdating = False

Sheets("ModelOutput_Plot1").Range("A1:L100").Clear

Sheets("ModelOutput_Plot1_B").Range("A1:L100").Clear

Sheets("ModelOutput_Plot1_C").Range("A1:L100").Clear

End Sub

Sub Create_BGHF_map()

Application.ScreenUpdating = False

Dim wsh As Object
Set wsh = VBA.CreateObject("WScript.Shell")
Dim waitOnReturn As Boolean: waitOnReturn = True
Dim windowStyle As Integer: windowStyle = 1
Dim py_cmd, py_exe_path As String
Dim arg1, arg2, arg3, arg4, arg5, arg6, runit_path, log_file_path As String

Dim ColNum As Integer
Dim Line As String
Dim LineValues() As Variant
Dim LineValues2() As Variant
Dim OutputFileNum As Integer
Dim PathName As String
Dim RowNum As Integer
Dim SheetValues() As Variant

''''''''''''''''''''''''''''''''''''''
' Save well list to ipa_wells.csv
''''''''''''''''''''''''''''''''''''''

' Define file name and path
input_path = Sheets("Model Input").Range("G3").Value
fNAME = "\" & "ipa_wells.csv"
file_path = input_path & fNAME

Debug.Print "file_path : " & file_path

OutputFileNum = FreeFile

'Open PathName & fNAME For Output Lock Write As #OutputFileNum
Open file_path For Output Lock Write As #OutputFileNum

SheetValues = Sheets("Well_List").Range("C3:J202").Value
ReDim LineValues(1 To 8)

For RowNum = 1 To 200
  For ColNum = 1 To 8
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

py_exe_path = Worksheets("Model Input").Range("G1").Value

' Scripts path
arg1 = Worksheets("Model Input").Range("G2").Value

' input path
arg2 = Worksheets("Model Input").Range("G3").Value

' output path
arg3 = Worksheets("Model Input").Range("G4").Value

Debug.Print "Test 2"

' file_name_new
arg4 = Worksheets("Calibration_Viewer").Range("F30").Value

Debug.Print "Test 3"

' file_name_ref
arg5 = Worksheets("Model Input").Range("B47").Value

Debug.Print "Test 4"

' Search radius
arg6 = Worksheets("Calibration_Viewer").Range("F33").Value

Debug.Print "Test 5"

runit_path = arg1 & "\make_bghf_map.py"

cmd = "cmd.exe /c " & py_exe_path & " " & runit_path & " " & arg1 & " " & arg2 & " " & arg3 & " " & arg4 & " " & arg5 & " " & arg6

Debug.Print cmd

wsh.Run cmd, windowStyle, waitOnReturn

Application.ScreenUpdating = True

MsgBox " The calibrated background heat flow map can be found in " & arg2

End Sub

Sub Load_ModelA_Path()

Application.ScreenUpdating = False

Dim fldr As FileDialog
Dim sItem, file_path As String
    
' Error handling
On Error GoTo eh

'''''''''''''''''
' Get folder path
'''''''''''''''''

Set fldr = Application.FileDialog(msoFileDialogFolderPicker)

Debug.Print "fldr : " & fldr

With fldr
    .Title = "Select a Folder"
    .AllowMultiSelect = False
    '.InitialFileName = strPath
    If .Show <> -1 Then GoTo NextCode
    sItem = .SelectedItems(1)
End With
NextCode:
GetFolder = sItem
Set fldr = Nothing

Debug.Print "sItem : " & sItem

Sheets("Calibration_Viewer").Range("N4").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub Load_ModelB_Path()

Application.ScreenUpdating = False

Dim fldr As FileDialog
Dim sItem, file_path As String
    
' Error handling
On Error GoTo eh

'''''''''''''''''
' Get folder path
'''''''''''''''''

Set fldr = Application.FileDialog(msoFileDialogFolderPicker)

Debug.Print "fldr : " & fldr

With fldr
    .Title = "Select a Folder"
    .AllowMultiSelect = False
    '.InitialFileName = strPath
    If .Show <> -1 Then GoTo NextCode
    sItem = .SelectedItems(1)
End With
NextCode:
GetFolder = sItem
Set fldr = Nothing

Debug.Print "sItem : " & sItem

Sheets("Calibration_Viewer").Range("N5").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub Load_ModelC_Path()

Application.ScreenUpdating = False

Dim fldr As FileDialog
Dim sItem, file_path As String
    
' Error handling
On Error GoTo eh

'''''''''''''''''
' Get folder path
'''''''''''''''''

Set fldr = Application.FileDialog(msoFileDialogFolderPicker)

Debug.Print "fldr : " & fldr

With fldr
    .Title = "Select a Folder"
    .AllowMultiSelect = False
    '.InitialFileName = strPath
    If .Show <> -1 Then GoTo NextCode
    sItem = .SelectedItems(1)
End With
NextCode:
GetFolder = sItem
Set fldr = Nothing

Debug.Print "sItem : " & sItem

Sheets("Calibration_Viewer").Range("N6").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub Load_Well_Extractions()

' Error handling
On Error GoTo eh

Application.ScreenUpdating = False

Dim PW As String
PW = "ibaw123A"
Sheets("Calibration_Viewer").Unprotect Password:=PW

ClearPlot1

Dim strpath4, well_name As String

well_name = Sheets("Calibration_Viewer").Range("M2").Value

'''''''''''
' Model A
'''''''''''

strpath4 = Sheets("Calibration_Viewer").Range("N4").Value & "\" & well_name & "_ModelOutput.csv"

Dim wbkS4 As Workbook
Dim wshS4 As Worksheet
Set wbkS4 = Workbooks.Open(Filename:=strpath4)
Set wshS4 = wbkS4.Worksheets(1)

'Sheets("ModelOutput_Plot1").Range("A1:L100").Clear

wshS4.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ModelOutput_Plot1").Range("A1")

wbkS4.Close SaveChanges:=False

Sheets("Calibration_Viewer").Range("N34").Value = strpath4

''''''''''
' Model B
''''''''''

strpath5 = Sheets("Calibration_Viewer").Range("N5").Value & "\" & well_name & "_ModelOutput.csv"

Dim wbkS5 As Workbook
Dim wshS5 As Worksheet
Set wbkS5 = Workbooks.Open(Filename:=strpath5)
Set wshS5 = wbkS5.Worksheets(1)

'Sheets("ModelOutput_Plot1").Range("A1:L100").Clear

wshS5.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ModelOutput_Plot1_B").Range("A1")

wbkS5.Close SaveChanges:=False

Sheets("Calibration_Viewer").Range("N35").Value = strpath5

''''''''''
' Model C
''''''''''

strpath6 = Sheets("Calibration_Viewer").Range("N6").Value & "\" & well_name & "_ModelOutput.csv"

Dim wbkS6 As Workbook
Dim wshS6 As Worksheet
Set wbkS6 = Workbooks.Open(Filename:=strpath6)
Set wshS6 = wbkS6.Worksheets(1)

'Sheets("ModelOutput_Plot1").Range("A1:L100").Clear

wshS6.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("ModelOutput_Plot1_C").Range("A1")

wbkS6.Close SaveChanges:=False

Sheets("Calibration_Viewer").Range("N36").Value = strpath6

Sheets("Calibration_Viewer").Protect Password:=PW

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    Sheets("Calibration_Viewer").Protect Password:=PW
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True

End Sub

Sub UpdateLabels()

Dim FilmDataSeries As Series
Dim SingleCell As Range
Dim FilmList As Range
Dim FilmCounter As Integer

On Error GoTo eh

Dim PW As String
PW = "ibaw123A"
Sheets("Calibration_Viewer").Unprotect Password:=PW

FilmCounter = 1

Set FilmList = Sheets("Well_List").Range("M3:M202")

Set FilmDataSeries = ActiveSheet.ChartObjects("BASEMAP").Chart.SeriesCollection(1)

FilmDataSeries.HasDataLabels = True

For Each SingleCell In FilmList

    FilmDataSeries.Points(FilmCounter).DataLabel.Text = SingleCell.Value

    FilmCounter = FilmCounter + 1

Next SingleCell

Sheets("Calibration_Viewer").Protect Password:=PW

Done:
    Exit Sub
eh:
    Sheets("Calibration_Viewer").Protect Password:=PW
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True

End Sub
