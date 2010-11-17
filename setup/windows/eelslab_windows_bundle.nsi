; These are the programs that are needed by ACME Suite.
!define PRODUCT_NAME "EELSLab 0.2.5-dev bundle installer"
!define PRODUCT_VERSION "0.1"

; !include "UserManagement.nsh"
 
 
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
 
Section -Prerequisites
  SetOutPath $INSTDIR\requires
  MessageBox MB_YESNO "Install Python 2.7?" /SD IDYES IDNO endPython
    File "requires\python-2.7.amd64.msi"
    ExecWait '"msiexec" /i "$INSTDIR\requires\python-2.7.amd64.msi"'
    Goto endPython
  endPython:
  MessageBox MB_YESNO "Install Numpy" /SD IDYES IDNO endNumpy
    File "requires\numpy-1.5.0.win-amd64-py2.7-mkl.exe"
    ExecWait "$INSTDIR\requires\numpy-1.5.0.win-amd64-py2.7-mkl.exe"
    Goto endNumpy
  endNumpy:
    MessageBox MB_YESNO "Install Scipy" /SD IDYES IDNO endScipy
        File "requires\scipy-0.8.0.win-amd64-py2.7.exe"
        ExecWait "$INSTDIR\requires\scipy-0.8.0.win-amd64-py2.7.exe"
        Goto endScipy
  endScipy:
    MessageBox MB_YESNO "Install Matplotlib" /SD IDYES IDNO endMatplotlib
        File "requires\matplotlib-1.0.0.win-amd64-py2.7.exe"
        ExecWait "$INSTDIR\requires\matplotlib-1.0.0.win-amd64-py2.7.exe"
        Goto endMatplotlib
  endMatplotlib:
    MessageBox MB_YESNO "Install Ipython" /SD IDYES IDNO endIpython
        File "requires\ipython-0.10.1.win-amd64-py2.7.exe"
        ExecWait "$INSTDIR\requires\ipython-0.10.1.win-amd64-py2.7.exe"
        Goto endIpython
  endIpython:
    MessageBox MB_YESNO "Install Pyreadline" /SD IDYES IDNO endPyreadline
        File "requires\pyreadline-1.6.1.win-amd64.exe"
        ExecWait "$INSTDIR\requires\pyreadline-1.6.1.win-amd64.exe"
        Goto endPyreadline
  endPyreadline:
    MessageBox MB_YESNO "Install NetCDF4" /SD IDYES IDNO endPyreadline
        File "requires\netCDF4-0.9.3.win-amd64-py2.7.exe"
        ExecWait "$INSTDIR\requires\netCDF4-0.9.3.win-amd64-py2.7.exe"
        Goto endNetCDF
  endNetCDF:
    MessageBox MB_YESNO "Install dateutil" /SD IDYES IDNO endPyreadline
        File "requires\python-dateutil-1.5.win-amd64.exe"
        ExecWait "$INSTDIR\requires\python-dateutil-1.5.win-amd64.exe"
        Goto endDateutil
  endDateutil:
    MessageBox MB_YESNO "Install WXPython" /SD IDYES IDNO endPyreadline
        File "requires\wxPython-2.8.11.0.win-amd64-py2.7.exe"
        ExecWait "$INSTDIR\requires\wxPython-2.8.11.0.win-amd64-py2.7.exe"
        Goto endWXPython
  endWXPython:
     MessageBox MB_YESNO "Install WXPython common" /SD IDYES IDNO endPyreadline
        File "requires\wxPython-common-2.8.11.0.win-amd64-py2.7.exe"
        ExecWait "$INSTDIR\requires\wxPython-common-2.8.11.0.win-amd64-py2.7.exe"
        Goto endWXPythonCommon
	endWXPythonCommon:
SectionEnd
Section EELSLab
	SetOutPath $INSTDIR\requires
    MessageBox MB_YESNO "Install EELSLab" /SD IDYES IDNO endEELSLab
        File "eelslab-0.2.5-dev.win-amd64.exe"
        ExecWait "$INSTDIR\eelslab-0.2.5-dev.win-amd64.exe"
        Goto endEELSLab
  endEELSLab:
SectionEnd

