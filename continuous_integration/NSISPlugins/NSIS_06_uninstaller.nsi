; Hyperspy installer script for Nullsoft installer system.
; Tested using version 2.46.
; requires installation of 1 extra plugins:
;     * UAC - http://nsis.sourceforge.net/UAC_plug-in

; This file based heavily on UAC_DualMode from the UAC plugin.

!include "MUI2.nsh"
!include "UAC.nsh"
!include "nsDialogs.nsh"
!include "StrFunc.nsh"
!include "LogicLib.nsh"
!include "X64.nsh"
!include "hspy_delete06_32.nsh"
!include "hspy_delete06_64.nsh"
!define APPNAME "HyperSpy"
;!define CL64 1 # Uncomment this line for 64bit
!define S_DEFINSTDIR_USER "$LocalAppData"
!define S_DEFINSTDIR_PORTABLE "$DOCUMENTS"
!ifdef CL64
	!define S_DEFINSTDIR_ADMIN "$ProgramFiles64"
!else
	!define S_DEFINSTDIR_ADMIN "$ProgramFiles"
!endif

!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\orange-uninstall.ico"

SetCompressor ZLIB

Name "Uninstall HyperSpy 0.6"


!ifndef BCM_SETSHIELD
!define BCM_SETSHIELD 0x0000160C
!endif

!define MUI_FINISHPAGE_NOAUTOCLOSE
!define MUI_UNFINISHPAGE_NOAUTOCLOSE

; !macro BUILD_LANGUAGES
; !macroend

/***************************************************
*                   Installer                      *
***************************************************/

SilentInstall silent
OutFile "uninstaller_06.exe"
!if ${BUILDUNINST} > 0
	RequestExecutionLevel admin
!else
	RequestExecutionLevel user
!endif
!insertmacro MUI_PAGE_INSTFILES

Section
	# Remove StartMenu entries
	Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
	Delete "$SMPROGRAMS\${APPNAME}\${APPNAME} QtConsole.lnk"
	Delete "$SMPROGRAMS\${APPNAME}\${APPNAME} Notebook.lnk"
	Delete "$SMPROGRAMS\${APPNAME}\Uninstall ${APPNAME}.lnk"
	RMDir "$SMPROGRAMS\${APPNAME}"
	ReadRegStr $R0 SHCTX \
				"Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
				"UninstallString"
	Delete /REBOOTOK $R0 ; delete the evil uninstaller
	DeleteRegKey SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"

	${If} ${FileExists} `$InstDir\python-2.7.4.amd64\*.*`
		!insertmacro hspy_delete06_64 $InstDir
	${EndIf}
	${If} ${FileExists} `$InstDir\python-2.7.4\*.*`
		!insertmacro hspy_delete06_32 $InstDir
	${EndIf}
SectionEnd
!insertmacro MUI_LANGUAGE "English"
