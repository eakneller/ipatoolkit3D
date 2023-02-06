Attribute VB_Name = "Source_Expulsion_Tools_IPA3D"
Sub ClearPlots_SR()

Application.ScreenUpdating = False

Debug.Print "Deleting pre-existing data"

Sheets("SRC_EXP_L_CUM").Range("A1:AF100").Clear

Sheets("SRC_EXP_L_INC").Range("A1:AF100").Clear

Sheets("SRC_EXP_G_CUM").Range("A1:AF100").Clear

Sheets("SRC_EXP_G_INC").Range("A1:AF100").Clear

Application.ScreenUpdating = True

End Sub

Sub Copy_Plot_to_Work_SR()

Dim PW As String

' Error handling
On Error GoTo eh

PW = "ibaw123A"

Dim chart_name As String

Sheets("Source_Expulsion").Unprotect Password:=PW

chart_name = Range("H4").Value

new_name = chart_name & "_C"

Sheets("Source_Expulsion").ChartObjects(chart_name).Select
Sheets("Source_Expulsion").ChartObjects(chart_name).Copy
Sheets("User_Work_Area").Paste

Sheets("User_Work_Area").ChartObjects(chart_name).Name = new_name

' Protect Worksheet
Sheets("Source_Expulsion").Protect Password:=PW

Done:
    Exit Sub
eh:

    ' Protect worksheet
    Sheets("Source_Expulsion").Protect Password:=PW

    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub Load_SR_EXP_Path_SR()

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

Sheets("Source_Expulsion").Range("D4").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub


Sub Load_SR_EXP_Data()

' Error handling
On Error GoTo eh

Application.ScreenUpdating = False

Dim PW As String
PW = "ibaw123A"
Sheets("Source_Expulsion").Unprotect Password:=PW

ClearPlots_SR

Dim strpath4, well_name As String

'''''''''''''''''''''''''''''''''''''''''''''''
' Source liquid expulsion history (Cummulative)
'''''''''''''''''''''''''''''''''''''''''''''''

strpath4 = Sheets("Source_Expulsion").Range("D4").Value & "\" & "Source_Expulsion_History_Liquid_GOB_CUM.csv"

Dim wbkS4 As Workbook
Dim wshS4 As Worksheet
Set wbkS4 = Workbooks.Open(Filename:=strpath4)
Set wshS4 = wbkS4.Worksheets(1)

Application.ScreenUpdating = False
wshS4.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("SRC_EXP_L_CUM").Range("A1")

wbkS4.Close SaveChanges:=False

Sheets("Source_Expulsion").Range("D55").Value = strpath4

'''''''''''''''''''''''''''''''''''''''''''''''
' Source liquid expulsion history (Incremental)
'''''''''''''''''''''''''''''''''''''''''''''''

strpath5 = Sheets("Source_Expulsion").Range("D4").Value & "\" & "Source_Expulsion_History_Liquid_GOB_INCR.csv"

Dim wbkS5 As Workbook
Dim wshS5 As Worksheet
Set wbkS5 = Workbooks.Open(Filename:=strpath5)
Set wshS5 = wbkS5.Worksheets(1)

Application.ScreenUpdating = False
wshS5.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("SRC_EXP_L_INC").Range("A1")

wbkS5.Close SaveChanges:=False

Sheets("Source_Expulsion").Range("D56").Value = strpath5

Debug.Print "Test 5"

'''''''''''''''''''''''''''''''''''''''''''''''
' Source gas expulsion history (Cummulative)
'''''''''''''''''''''''''''''''''''''''''''''''

strpath6 = Sheets("Source_Expulsion").Range("D4").Value & "\" & "Source_Expulsion_History_Gas_Tcf_CUM.csv"

Dim wbkS6 As Workbook
Dim wshS6 As Worksheet
Set wbkS6 = Workbooks.Open(Filename:=strpath6)
Set wshS6 = wbkS6.Worksheets(1)

Application.ScreenUpdating = False
wshS6.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("SRC_EXP_G_CUM").Range("A1")

wbkS6.Close SaveChanges:=False

Sheets("Source_Expulsion").Range("D57").Value = strpath6

'''''''''''''''''''''''''''''''''''''''''''''''
' Source gas expulsion history (Incremental)
'''''''''''''''''''''''''''''''''''''''''''''''

strpath7 = Sheets("Source_Expulsion").Range("D4").Value & "\" & "Source_Expulsion_History_Gas_Tcf_INCR.csv"

Dim wbkS7 As Workbook
Dim wshS7 As Worksheet
Set wbkS7 = Workbooks.Open(Filename:=strpath7)
Set wshS7 = wbkS7.Worksheets(1)

Application.ScreenUpdating = False
wshS7.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("SRC_EXP_G_INC").Range("A1")

wbkS7.Close SaveChanges:=False

Sheets("Source_Expulsion").Range("D58").Value = strpath7

''''

Sheets("Source_Expulsion").Protect Password:=PW

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    Sheets("Source_Expulsion").Protect Password:=PW
    MsgBox "The following error occurred: " & Err.Description
    Application.ScreenUpdating = True

End Sub
