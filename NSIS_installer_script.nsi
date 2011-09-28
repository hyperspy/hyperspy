;NSIS Modern User Interface
;Basic Example Script
;Written by Joost Verburg

;--------------------------------
;Include Modern UI

  !include "MUI2.nsh"

;--------------------------------
;Include Registry tools
  !include "Registry.nsh"

;--------------------------------
;General

  ;Name and file
  Name "Hyperspy"
  OutFile "Hyperspy_X86_0.3_dev.exe"

  ;Default installation folder
  InstallDir "$DOCUMENTS\Hyperspy"
  
  ;Get installation folder from registry if available
  InstallDirRegKey HKCU "Software\Hyperspy" ""

  RequestExecutionLevel user

;--------------------------------
;Interface Settings

  !define MUI_ABORTWARNING

;--------------------------------
;Pages

  !insertmacro MUI_PAGE_LICENSE "COPYING.txt"
  !insertmacro MUI_PAGE_COMPONENTS
  !insertmacro MUI_PAGE_DIRECTORY
  !insertmacro MUI_PAGE_INSTFILES
  
  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_INSTFILES
  
;--------------------------------
;Languages
 
  !insertmacro MUI_LANGUAGE "English"

;--------------------------------
;Installer Sections

Section "Hyperspy" SecDummy

  SetOutPath "$INSTDIR"
  
  ;ADD YOUR OWN FILES HERE...
  File /r "PortableInstall\*"
  ;File /r "PortableInstall\python.exe"

  ;Write start menu shortcut
  CreateDirectory "$SMPROGRAMS\Hyperspy"
  createShortCut "$SMPROGRAMS\Hyperspy\Hyperspy Interpreter.lnk" "$INSTDIR\Scripts\hyperspy.bat"
  ;Write uninstall start menu shortcut
  createShortCut "$SMPROGRAMS\Hyperspy\Uninstall Hyperspy.lnk" "$INSTDIR\uninstall.exe"

  ;Write registry shortcuts (open hyperspy here)
  ${registry::CreateKey} "HKEY_CLASSES_ROOT\Directory\shell\Hyperspy" $R0
  ${registry::Write} "HKEY_CLASSES_ROOT\Directory\shell\Hyperspy" "" "Hyperspy Here" "REG_EXPAND_SZ" $R0
  ${registry::CreateKey} "HKEY_CLASSES_ROOT\Directory\shell\Hyperspy\command" $R0
  ${registry::Write} "HKEY_CLASSES_ROOT\Directory\shell\Hyperspy\command" "" 'cmd.exe /k "cd %1 & $INSTDIR\Scripts\hyperspy.bat"' "REG_EXPAND_SZ" $R0
  
  ;Store installation folder
  WriteRegStr HKCR "Software\Hyperspy" "" $INSTDIR
  
  ;Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

SectionEnd

;--------------------------------
;Descriptions

  ;Language strings
  LangString DESC_SecDummy ${LANG_ENGLISH} "A test section."

  ;Assign language strings to sections
  !insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SecDummy} $(DESC_SecDummy)
  !insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
;Uninstaller Section

Section "Uninstall"

  ;ADD YOUR OWN FILES HERE...

  Delete $INSTDIR\uninstaller.exe

  RMDir /r /REBOOTOK "$INSTDIR"
  Delete "$SMPROGRAMS\Hyperspy\Hyperspy Interpreter.lnk"
  Delete "$SMPROGRAMS\Hyperspy\Uninstall Hyperspy.lnk"	
  RMDir "$SMPROGRAMS\Hyperspy"

  DeleteRegKey /ifempty HKCU "Software\Hyperspy"
  DeleteRegKey HKCR "Directory\shell\Hyperspy\command"
  DeleteRegKey HKCR "Directory\shell\Hyperspy"

SectionEnd
