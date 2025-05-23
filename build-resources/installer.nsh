!macro customHeader
  !system "echo Installing VG Control with Python runtime..."
!macroend

!macro preInit
  ; Check for Python runtime
  SetRegView 64
  ReadRegStr $0 HKLM "SOFTWARE\Python\PythonCore\3.8\InstallPath" ""
  ${If} $0 == ""
    ReadRegStr $0 HKLM "SOFTWARE\Python\PythonCore\3.9\InstallPath" ""
  ${EndIf}
  ${If} $0 == ""
    ReadRegStr $0 HKLM "SOFTWARE\Python\PythonCore\3.10\InstallPath" ""
  ${EndIf}
  ${If} $0 == ""
    ReadRegStr $0 HKLM "SOFTWARE\Python\PythonCore\3.11\InstallPath" ""
  ${EndIf}
  SetRegView 32
!macroend

!macro customInit
  ; Custom initialization code
!macroend

!macro customInstall
  ; Install Visual C++ Redistributable if needed
  ${ifNot} ${FileExists} "$SYSDIR\msvcp140.dll"
    DetailPrint "Installing Visual C++ Redistributable..."
    File /oname=$PLUGINSDIR\vc_redist.x64.exe "${BUILD_RESOURCES_DIR}\vc_redist.x64.exe"
    ExecWait '"$PLUGINSDIR\vc_redist.x64.exe" /quiet /norestart'
  ${endIf}
  
  ; Create shortcuts with proper working directory
  CreateShortCut "$DESKTOP\OWL Control.lnk" "$INSTDIR\OWL Control.exe" "" "$INSTDIR\OWL Control.exe" 0 SW_SHOWNORMAL "" "Video game control input tracking"
  CreateShortCut "$SMPROGRAMS\OWL Control\OWL Control.lnk" "$INSTDIR\OWL Control.exe" "" "$INSTDIR\OWL Control.exe" 0 SW_SHOWNORMAL "" "Video game control input tracking"
  
  ; Register application
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "UninstallString" "$INSTDIR\Uninstall ${PRODUCT_NAME}.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "DisplayIcon" "$INSTDIR\OWL Control.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "Publisher" "Open World Labs"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "NoRepair" 1
!macroend

!macro customUnInstall
  ; Remove shortcuts
  Delete "$DESKTOP\OWL Control.lnk"
  Delete "$SMPROGRAMS\OWL Control\OWL Control.lnk"
  RMDir "$SMPROGRAMS\OWL Control"
  
  ; Remove registry entries
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}"
!macroend