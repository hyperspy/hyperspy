; These are the programs that are needed by ACME Suite.
!define PRODUCT_NAME "EELSLab bundle installer"
!define PRODUCT_VERSION "1.0a"

!include "UserManagement.nsh"
 
 
; MUI 1.67 compatible ------
!include "MUI.nsh"

; Welcome page
!insertmacro MUI_PAGE_WELCOME
; Components page
!insertmacro MUI_PAGE_COMPONENTS
; Instfiles page
!insertmacro MUI_PAGE_INSTFILES
; Finish page
!insertmacro MUI_PAGE_FINISH
 
; Language files
!insertmacro MUI_LANGUAGE "English"
 
; Reserve files
!insertmacro MUI_RESERVEFILE_INSTALLOPTIONS
 
; MUI end ------

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "EELSLab_bundle.exe"
ShowInstDetails show
SetOutPath $INSTDIR\prerequisites

Section -Prerequisites
  SetOutPath $INSTDIR\Prerequisites
  MessageBox MB_YESNO "Install Python 2.5?" /SD IDYES IDNO endPython
    File "\prerequisites\python-2.5.2.msi"
    ExecWait '"msiexec" /i "$INSTDIR\Prerequisites\python-2.5.2.msi"'
    Goto endPython
  endPython:
  MessageBox MB_YESNO "Install Numpy" /SD IDYES IDNO endNumpy
    File "\prerequisites\numpy-1.1.1-win32-superpack-python2.5.exe"
    ExecWait "$INSTDIR\prerequisites\numpy-1.1.1-win32-superpack-python2.5.exe"
    Goto endNumpy
  endNumpy:
    MessageBox MB_YESNO "Install Scipy" /SD IDYES IDNO endScipy
        File "\prerequisites\scipy-0.6.0.win32-py2.5.exe"
        ExecWait "$INSTDIR\prerequisites\scipy-0.6.0.win32-py2.5.exe"
        Goto endScipy
  endScipy:
    MessageBox MB_YESNO "Install Matplotlib" /SD IDYES IDNO endMatplotlib
        File "\prerequisites\matplotlib-0.98.3.win32-py2.5.exe"
        ExecWait "$INSTDIR\prerequisites\matplotlib-0.98.3.win32-py2.5.exe"
        Goto endMatplotlib
  endMatplotlib:
    MessageBox MB_YESNO "Install Ipython" /SD IDYES IDNO endIpython
        File "\prerequisites\ipython-0.8.4.win32-setup.exe"
        ExecWait "$INSTDIR\prerequisites\ipython-0.8.4.win32-setup.exe"
        Goto endIpython
  endIpython:
    MessageBox MB_YESNO "Install Pyreadline" /SD IDYES IDNO endPyreadline
        File "\prerequisites\pyreadline-1.5-win32-setup.exe"
        ExecWait "$INSTDIR\prerequisites\pyreadline-1.5-win32-setup.exe"
        Goto endPyreadline
  endPyreadline:
SectionEnd
Section EELSLab
    MessageBox MB_YESNO "Install EELSLab" /SD IDYES IDNO endEELSLab
        File "\prerequisites\eelslab-0.1a.dev-r106.win32.exe"
        ExecWait "$INSTDIR\prerequisites\eelslab-0.1a.dev-r106.win32.exe"
        Goto endEELSLab
  endEELSLab:
SectionEnd

