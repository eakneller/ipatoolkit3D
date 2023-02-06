Attribute VB_Name = "History_Tools_IPA3D"
Sub ClearPlots_HP()

Application.ScreenUpdating = False

Debug.Print "Deleting pre-existing data"

Sheets("burial_history").Range("A1:AF100").Clear

Sheets("temp_history").Range("A1:AF100").Clear

Sheets("mass_gen_history").Range("A1:AF100").Clear

Sheets("pore_exp_rate_history").Range("A1:AF100").Clear

Sheets("vODGEXSR_history").Range("A1:AF100").Clear

Sheets("vFGEXSR_history").Range("A1:AF100").Clear

Sheets("Ro_history").Range("A1:AF100").Clear

Sheets("LOM_history").Range("A1:AF100").Clear

Sheets("SEC_EXP_history").Range("A1:AF100").Clear

Sheets("SEC_EXP_RATE_history").Range("A1:AF100").Clear

Sheets("SEC_vFGEXSR_history").Range("A1:AF100").Clear

Sheets("TR_history").Range("A1:AF100").Clear

Application.ScreenUpdating = True

End Sub

Sub CopyPlot_to_Work_HP()

Dim PW As String

' Error handling
On Error GoTo eh

PW = "ibaw123A"

Dim chart_name As String

Sheets("History_Plots").Unprotect Password:=PW

chart_name = Range("W4").Value

new_name = chart_name & "_C"

Sheets("History_Plots").ChartObjects(chart_name).Select
Sheets("History_Plots").ChartObjects(chart_name).Copy
Sheets("User_Work_Area").Paste

Sheets("User_Work_Area").ChartObjects(chart_name).Name = new_name

' Protect Worksheet
Sheets("History_Plots").Protect Password:=PW

Done:
    Exit Sub
eh:

    ' Protect worksheet
    Sheets("History_Plots").Protect Password:=PW

    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub Load_Well_Extraction_Path_HP()

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

Sheets("History_Plots").Range("N4").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub


Sub Load_Well_Extractions_HP()

' Error handling
On Error GoTo eh

Application.ScreenUpdating = False

Dim PW As String
PW = "ibaw123A"
Sheets("History_Plots").Unprotect Password:=PW

ClearPlots_HP

Dim strpath4, well_name As String

well_name = Sheets("History_Plots").Range("M2").Value

'''''''''''''''''''''''''''''
' Pore expulsion rate history
'''''''''''''''''''''''''''''

Debug.Print "Test 1"

strpath4 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_mHC_mg_gOC_Myr__History.csv"

Dim wbkS4 As Workbook
Dim wshS4 As Worksheet
Set wbkS4 = Workbooks.Open(Filename:=strpath4)
Set wshS4 = wbkS4.Worksheets(1)

Debug.Print "Test 2"

Application.ScreenUpdating = False
wshS4.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("pore_exp_rate_history").Range("A1")

Debug.Print "Test 3"

wbkS4.Close SaveChanges:=False

Sheets("History_Plots").Range("N100").Value = strpath4

Debug.Print "Test 4"

''''''''''
' HC Mass generation history
''''''''''

strpath5 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_mHC_mg_gOC__History.csv"

Dim wbkS5 As Workbook
Dim wshS5 As Worksheet
Set wbkS5 = Workbooks.Open(Filename:=strpath5)
Set wshS5 = wbkS5.Worksheets(1)

Application.ScreenUpdating = False
wshS5.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("mass_gen_history").Range("A1")

wbkS5.Close SaveChanges:=False

Sheets("History_Plots").Range("N101").Value = strpath5

Debug.Print "Test 5"

''''''''''
' Temperature history
''''''''''

strpath6 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_TEMP_C__History.csv"

Dim wbkS6 As Workbook
Dim wshS6 As Worksheet
Set wbkS6 = Workbooks.Open(Filename:=strpath6)
Set wshS6 = wbkS6.Worksheets(1)

Application.ScreenUpdating = False
wshS6.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("temp_history").Range("A1")

wbkS6.Close SaveChanges:=False

Sheets("History_Plots").Range("N102").Value = strpath6

''''''''''
' Burial history
''''''''''

strpath7 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_TVDssm_History.csv"

Dim wbkS7 As Workbook
Dim wshS7 As Worksheet
Set wbkS7 = Workbooks.Open(Filename:=strpath7)
Set wshS7 = wbkS7.Worksheets(1)

Application.ScreenUpdating = False
wshS7.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("burial_history").Range("A1")

wbkS7.Close SaveChanges:=False

Sheets("History_Plots").Range("N103").Value = strpath7

''''''''''
' LOM history
''''''''''

strpath8 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_LOM__History.csv"

Dim wbkS8 As Workbook
Dim wshS8 As Worksheet
Set wbkS8 = Workbooks.Open(Filename:=strpath8)
Set wshS8 = wbkS8.Worksheets(1)

Application.ScreenUpdating = False
wshS8.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("LOM_history").Range("A1")

wbkS8.Close SaveChanges:=False

Sheets("History_Plots").Range("N104").Value = strpath8

Debug.Print "Test 6"

''''''''''
' Secondary expulsion rate history
''''''''''

strpath9 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_SEC_mHC_mg_gOC_Myr__History.csv"

Dim wbkS9 As Workbook
Dim wshS9 As Worksheet
Set wbkS9 = Workbooks.Open(Filename:=strpath9)
Set wshS9 = wbkS9.Worksheets(1)

Application.ScreenUpdating = False
wshS9.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("SEC_EXP_RATE_history").Range("A1")

wbkS9.Close SaveChanges:=False

Sheets("History_Plots").Range("N105").Value = strpath9

''''''''''
' Secondary expulsion history
''''''''''

strpath10 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_SEC_mHC_mg_gOC__History.csv"

Dim wbkS10 As Workbook
Dim wshS10 As Worksheet
Set wbkS10 = Workbooks.Open(Filename:=strpath10)
Set wshS10 = wbkS10.Worksheets(1)

Application.ScreenUpdating = False
wshS10.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("SEC_EXP_history").Range("A1")

wbkS10.Close SaveChanges:=False

Sheets("History_Plots").Range("N106").Value = strpath10

''''''''''
' Ro history
''''''''''

strpath11 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_EasyRo__History.csv"

Dim wbkS11 As Workbook
Dim wshS11 As Worksheet
Set wbkS11 = Workbooks.Open(Filename:=strpath11)
Set wshS11 = wbkS11.Worksheets(1)

Application.ScreenUpdating = False
wshS11.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("Ro_history").Range("A1")

wbkS11.Close SaveChanges:=False

Sheets("History_Plots").Range("N107").Value = strpath11

''''''''''
' Liquid source expulsion history
''''''''''

strpath12 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_vODGEXSR_GOB__History.csv"

Dim wbkS12 As Workbook
Dim wshS12 As Worksheet
Set wbkS12 = Workbooks.Open(Filename:=strpath12)
Set wshS12 = wbkS12.Worksheets(1)

Application.ScreenUpdating = False
wshS12.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("vODGEXSR_history").Range("A1")

wbkS12.Close SaveChanges:=False

Sheets("History_Plots").Range("N108").Value = strpath12

''''''''''
' FG source expulsion history
''''''''''

strpath13 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_vFGEXSR_Tcf__History.csv"

Dim wbkS13 As Workbook
Dim wshS13 As Worksheet
Set wbkS13 = Workbooks.Open(Filename:=strpath13)
Set wshS13 = wbkS13.Worksheets(1)

Application.ScreenUpdating = False
wshS13.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("vFGEXSR_history").Range("A1")

wbkS13.Close SaveChanges:=False

Sheets("History_Plots").Range("N108").Value = strpath13

''''''''''
' secondary FG source expulsion history
''''''''''

strpath14 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_SEC_vFGEXSR_Tcf__History.csv"

Dim wbkS14 As Workbook
Dim wshS14 As Worksheet
Set wbkS14 = Workbooks.Open(Filename:=strpath14)
Set wshS14 = wbkS14.Worksheets(1)

Application.ScreenUpdating = False
wshS14.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("SEC_vFGEXSR_history").Range("A1")

wbkS14.Close SaveChanges:=False

Sheets("History_Plots").Range("N109").Value = strpath14

''''''''''
' TR history
''''''''''

strpath15 = Sheets("History_Plots").Range("N4").Value & "\" & well_name & "_TR__History.csv"

Dim wbkS15 As Workbook
Dim wshS15 As Worksheet
Set wbkS15 = Workbooks.Open(Filename:=strpath15)
Set wshS15 = wbkS15.Worksheets(1)

Application.ScreenUpdating = False
wshS15.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("TR_history").Range("A1")

wbkS15.Close SaveChanges:=False

Sheets("History_Plots").Range("N110").Value = strpath15

''''

Sheets("History_Plots").Protect Password:=PW

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    Sheets("History_Plots").Protect Password:=PW
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True

End Sub

Sub UpdateLabels_HP()

Dim FilmDataSeries As Series
Dim SingleCell As Range
Dim FilmList As Range
Dim FilmCounter As Integer

On Error GoTo eh

Dim PW As String
PW = "ibaw123A"
Sheets("History_Plots").Unprotect Password:=PW

FilmCounter = 1

Set FilmList = Sheets("Well_List").Range("M3:M202")

Set FilmDataSeries = ActiveSheet.ChartObjects("BASEMAP").Chart.SeriesCollection(1)

FilmDataSeries.HasDataLabels = True

For Each SingleCell In FilmList

    FilmDataSeries.Points(FilmCounter).DataLabel.Text = SingleCell.Value

    FilmCounter = FilmCounter + 1

Next SingleCell

Sheets("History_Plots").Protect Password:=PW

Done:
    Exit Sub
eh:
    Sheets("History_Plots").Protect Password:=PW
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True

End Sub
