Attribute VB_Name = "Set_Paths_Tools_IPA3D"
Sub SetInputPath_SP()

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

Sheets("Set Paths").Range("C13").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub SetOutputPath_SP()

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

Sheets("Set Paths").Range("C14").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub

Sub SetScriptsPath_SP()

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

Sheets("Set Paths").Range("C12").Value = sItem

Application.ScreenUpdating = True

Done:
    Exit Sub
eh:
    
    Application.ScreenUpdating = True
    
    MsgBox "The following error occurred: " & Err.Description

End Sub
