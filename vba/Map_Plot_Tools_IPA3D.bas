Attribute VB_Name = "Map_Plot_Tools_IPA3D"
Sub Plot_Map(strpath As String, itype_option As Integer)

Dim wsh As Object
Set wsh = VBA.CreateObject("WScript.Shell")
Dim waitOnReturn As Boolean: waitOnReturn = True
Dim windowStyle As Integer: windowStyle = 1
Dim py_cmd, py_exe_path, scripts_path As String
Dim arg1, arg2, arg3, log_file_path As String
Dim file_out_name, file_out_name_base, new_file_path As String
Dim strpath1 As String
Dim wbkS1 As Workbook
Dim wshS1 As Worksheet
Dim sheet_with_input_path As String
Dim cell_input_path As String
Dim output_csv As String
Dim sheet_tmp_csv As String
Dim range_tmp_csv As String
Dim row_max As Integer
Dim col_max As Integer

' Error handling
On Error GoTo eh

Application.ScreenUpdating = False

Dim PW As String
PW = "ibaw123A"
Sheets("ZMAP_Viewer").Unprotect Password:=PW

py_exe_path = Worksheets("Model Input").Range("G1").Value
scripts_path = Worksheets("Model Input").Range("G2").Value

If strpath <> "" Then
    strpath = Replace(strpath, ".zip", "")
    With CreateObject("Scripting.FileSystemObject")
        strExt = .GetExtensionName(strpath)
    End With
    Debug.Print "strExt : ", strExt
    ' Get the filename with extension
    file_out_name = Right(strpath, Len(strpath) - InStrRev(strpath, "\"))
    ' Get the file name without extension
    file_out_name_base = Replace(file_out_name, ".zip", "")
    Debug.Print "file_out_name_base: ", file_out_name_base
    file_out_name_base = Replace(file_out_name_base, ".dat", "")
    Debug.Print "file_out_name_base: ", file_out_name_base
    ' Path to file
    arg1 = strpath
    arg1 = Replace(arg1, ".zip", "")
    ' Output directory path
    arg2 = Worksheets("Model Input").Range("G4").Value
    arg2 = arg2 + "\"
    ' Base name of map file
    arg3 = file_out_name_base
    
    ''''''''''''''''''''''''''''''''''''''
    ' Save plot info to ipa_plot_zmaps.csv
    ''''''''''''''''''''''''''''''''''''''
    sheet_with_input_path = "Model Input"
    cell_input_path = "G3"
    output_csv = "ipa_plot_zmaps.csv"
    sheet_tmp_csv = "ZMAP_Viewer"
    range_tmp_csv = "B3:E35"
    row_max = 33
    col_max = 4
    Call Model_IO_Tools_IPA3D.SaveCSV(sheet_with_input_path, cell_input_path, _
                 output_csv, sheet_tmp_csv, _
                 range_tmp_csv, row_max, _
                 col_max)
    
    ' input directory path
    inp = Worksheets("Model Input").Range("G3").Value
    inp = inp + "\"
    
    Debug.Print "File path: ", arg1
    Debug.Print "Output path: ", arg2
    Debug.Print "Basename: ", arg3
    
    cmd = "cmd.exe /c " & py_exe_path & " " & scripts_path & "\plotit_maps.py" _
    & " " & arg1 & " " & arg2 & " " & arg3 & " " & inp & " " & itype_option
    Debug.Print cmd
    
    If strExt = "dat" Then
        Debug.Print "We found a .dat file"
        wsh.Run cmd, windowStyle, waitOnReturn
        'new_file_path = arg2 & file_out_name_base & ".png"
        new_file_path = Replace(arg1, ".dat", ".png")
        Debug.Print "new_file_path : " & new_file_path
        If Dir(new_file_path) <> "" Then
            With Sheets("ZMAP_Viewer").Shapes("Chart 3").Chart.PlotArea
                .Fill.UserPicture PictureFile:=new_file_path
                .Fill.Visible = True
            End With
            Sheets("ZMAP_Viewer").Range("J3").Value = file_out_name_base
            Sheets("ZMAP_Viewer").Range("F50").Value = strpath
            'strpath1 = arg2 & arg3 & ".csv"
            strpath1 = Replace(arg1, ".dat", ".csv")
            If Dir(strpath1) <> "" Then
                Debug.Print "Path to info file : " & strpath1
                Set wbkS1 = Workbooks.Open(Filename:=strpath1)
                Set wshS1 = wbkS1.Worksheets(1)
                wshS1.UsedRange.Copy Destination:=ThisWorkbook.Worksheets("tmp_map_info").Range("A1")
                wbkS1.Close SaveChanges:=False
                Sheets("tmp_map_info").Range("A2:J2").Copy
                Sheets("ZMAP_Viewer").Range("F48:O48").PasteSpecial Paste:=xlPasteValues
                If itype_option = 0 Then ' Update if loading for first time
                    Debug.Print "Update if loading for first time"
                    Sheets("ZMAP_Viewer").Range("C6").Value = Sheets("ZMAP_Viewer").Range("N48").Value ' gmin =
                    Sheets("ZMAP_Viewer").Range("C7").Value = Sheets("ZMAP_Viewer").Range("O48").Value ' gmax =
                    Sheets("ZMAP_Viewer").Range("C8").Value = Sheets("ZMAP_Viewer").Range("J48").Value ' xmin =
                    Sheets("ZMAP_Viewer").Range("C9").Value = Sheets("ZMAP_Viewer").Range("K48").Value ' xmax =
                    Sheets("ZMAP_Viewer").Range("C10").Value = Sheets("ZMAP_Viewer").Range("L48").Value ' ymin =
                    Sheets("ZMAP_Viewer").Range("C11").Value = Sheets("ZMAP_Viewer").Range("M48").Value ' ymax =
                End If
            Else:
                MsgBox "Problem map info path : " & strpath1
            End If
        Else
            MsgBox "Problem new image file path : " & new_file_path
        End If
    End If
End If

Sheets("ZMAP_Viewer").Protect Password:=PW
Application.ScreenUpdating = True
Done:
    Exit Sub
eh:
    Sheets("ZMAP_Viewer").Protect Password:=PW
    Application.ScreenUpdating = True
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub SelectMapFile_ZMAP()

Dim intChoice As Integer
Dim itype_option As Integer
Dim strpath As String
        
' Error handling
On Error GoTo eh
Application.ScreenUpdating = False
' Turn on manual mode to present re-calculation
'Application.Calculation = xlManual
'only allow the user to select one file
Application.FileDialog(msoFileDialogOpen).AllowMultiSelect = False
'make the file dialog visible to the user
intChoice = Application.FileDialog(msoFileDialogOpen).Show
'determine what choice the user made
If intChoice <> 0 Then
    'get the file path selected by the user
    strpath = Application.FileDialog( _
    msoFileDialogOpen).SelectedItems(1)

End If
Debug.Print "int_choice : " & int_choice & "  : strPath : " & strpath
itype_option = 0
Plot_Map strpath, itype_option
Application.ScreenUpdating = True
Done:
    Exit Sub
eh:
    Application.ScreenUpdating = True
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub UpdateMap_ZMAP()

Dim strpath As String
Dim itype_option As Integer

Application.ScreenUpdating = False
strpath = Sheets("ZMAP_Viewer").Range("F50").Value
itype_option = 1

Plot_Map strpath, itype_option

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    Application.ScreenUpdating = True
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub PlotAll_ZMAP()

Dim strpath As String
Dim itype_option As Integer

Application.ScreenUpdating = False
strpath = Sheets("ZMAP_Viewer").Range("F50").Value
itype_option = 2

Plot_Map strpath, itype_option

Application.ScreenUpdating = True

Done:
    MsgBox "All plots are located in the user defined output directory."
    Exit Sub
eh:
    Application.ScreenUpdating = True
    MsgBox "The following error occurred: " & Err.Description

End Sub

