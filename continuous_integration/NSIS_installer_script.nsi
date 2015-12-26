; Hyperspy installer script for Nullsoft installer system.
; Tested using version 2.46.
; requires installation of 1 extra plugins:
;     * UAC - http://nsis.sourceforge.net/UAC_plug-in

; This file based heavily on UAC_DualMode from the UAC plugin.

!addPluginDir "__NSIS_PLUGINS__"
!addIncludeDir "__NSIS_PLUGINS__"
!include "MUI2.nsh"
!include "UAC.nsh"
!include "nsDialogs.nsh"
!include "StrFunc.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "X64.nsh"
!include "__DELETE_MACRO_NAME__.nsh"
!define APPNAME "HyperSpy"
!define APPVERSION "__VERSION__"
!define ARCHITECTURE "__ARCHITECTURE__"
;!define CL64 1 # Uncomment this line for 64bit
!define WINPYTHON_PATH "__WINPYTHON_PATH__"
!define PYTHON_FOLDER "__PYTHON_FOLDER__"
!define S_NAME "HyperSpy-${APPVERSION}-Bundle-Windows-${ARCHITECTURE}"
!define S_DEFINSTDIR_USER "$LocalAppData"
!define S_DEFINSTDIR_PORTABLE "$DOCUMENTS"
!ifdef CL64
	!define S_DEFINSTDIR_ADMIN "$ProgramFiles64"
!else
	!define S_DEFINSTDIR_ADMIN "$ProgramFiles"
!endif
!define APP_REL_INSTDIR "${APPNAME} ${APPVERSION}"
!define APP_INSTDIR "$INSTDIR\${APP_REL_INSTDIR}"
!define UNINSTALLER_FULLPATH "${APP_INSTDIR}\Uninstall_Hyperspy_Bundle.exe"

!define MUI_ICON "__HSPY_ICON__"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\orange-uninstall.ico"

SetCompressor /SOLID ZLIB

Name "${S_NAME}"
OutFile "${S_NAME}.exe"


!ifndef BCM_SETSHIELD
!define BCM_SETSHIELD 0x0000160C
!endif

!define MUI_FINISHPAGE_NOAUTOCLOSE
!define MUI_UNFINISHPAGE_NOAUTOCLOSE

; !macro BUILD_LANGUAGES
; !macroend

!macro Go_admin
	uac_tryagain:
	!insertmacro UAC_RunElevated
	${Switch} $0
		${Case} 0
			${IfThen} $1 = 1 ${|} Quit ${|} ;we are the outer process, the inner process has done its work, we are done
			${IfThen} $3 <> 0 ${|} ${Break} ${|} ;we are admin, let the show go on
			${If} $1 = 3 ;RunAs completed successfully, but with a non-admin user
				MessageBox mb_YesNo|mb_IconExclamation|mb_TopMost|mb_SetForeground "This uninstaller requires admin privileges, try again" /SD IDNO IDYES uac_tryagain IDNO 0
			${EndIf}
			;fall-through and die
		${Case} 1223
			MessageBox MB_ICONSTOP|MB_TOPMOST|MB_SETFOREGROUND "This uninstaller requires admin privileges, aborting!"
			Quit
		${Case} 1062
			MessageBox MB_ICONSTOP|MB_TOPMOST|MB_SETFOREGROUND "Logon service not running, aborting!"
			Quit
		${Default}
			MessageBox MB_ICONSTOP|MB_TOPMOST|MB_SETFOREGROUND "Unable to elevate, error $0"
			Quit
	${EndSwitch}

	SetShellVarContext all
!macroend

/***************************************************
*                   Installer                      *
***************************************************/

RequestExecutionLevel user

!macro SetInstMode m
	StrCpy $InstMode ${m}
	; messageBox MB_OK "$InstMode"
	Call InstModeChanged
!macroend


; ------- Installer structure -------------------------------------------------
!define MUI_COMPONENTSPAGE_NODESC
!define MUI_CUSTOMFUNCTION_GUIINIT GuiInit

!insertmacro MUI_PAGE_LICENSE COPYING.txt
Page custom InstModeSelectionPage_Create InstModeSelectionPage_Leave
!define MUI_PAGE_CUSTOMFUNCTION_PRE disableBack
!define MUI_DIRECTORYPAGE_TEXT_TOP "${APPNAME} ${APPVERSION} will be installed to a subfolder named $\"${APP_REL_INSTDIR}$\" in the following folder. To install to a different base folder, click Browse and select another folder."
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; ------- UAC functions -------------------------------------------------------
Var InstMode ;0=current, 1=portable, 2=all users,

Function InstModeChanged
	${If} $InstMode = 0 ; Current user
		SetShellVarContext CURRENT
		${IfNotThen} ${Silent} ${|} StrCpy $InstDir "${S_DEFINSTDIR_USER}" ${|}
	${ElseIf} $InstMode = 1 ; Portable
		SetShellVarContext CURRENT
		${IfNotThen} ${Silent} ${|} StrCpy $InstDir "${S_DEFINSTDIR_PORTABLE}" ${|}
	${ElseIf} $InstMode = 2 ; Admin
		SetShellVarContext ALL
		${IfNotThen} ${Silent} ${|} StrCpy $InstDir "${S_DEFINSTDIR_ADMIN}" ${|}
	${EndIf}
FunctionEnd

Function .onInit
	!insertmacro UAC_PageElevation_OnInit
	${If} ${UAC_IsInnerInstance}
	${AndIfNot} ${UAC_IsAdmin}
		;special return value for outer instance so it knows we did not
		; have admin rights
		SetErrorLevel 0x666666
		Quit
	${EndIf}

	StrCpy $InstMode 0
	${IfThen} ${UAC_IsAdmin} ${|} StrCpy $InstMode 2 ${|}
	Call InstModeChanged

	${If} ${Silent}
	${AndIf} $InstDir == "" ;defaults (for silent installs)
		SetSilent normal
		Call InstModeChanged
		SetSilent silent
	${EndIf}


FunctionEnd

Function GuiInit
	!insertmacro UAC_PageElevation_OnGuiInit
FunctionEnd

Function disableBack
	${If} ${UAC_IsInnerInstance}
		GetDlgItem $0 $HWNDParent 3
		EnableWindow $0 0
	${EndIf}
FunctionEnd

Function RemoveNextBtnShield
	GetDlgItem $0 $hwndParent 1
	SendMessage $0 ${BCM_SETSHIELD} 0 0
FunctionEnd

Function InstModeSelectionPage_Create
	!insertmacro MUI_HEADER_TEXT_PAGE "Select install type" ""
	GetFunctionAddress $8 InstModeSelectionPage_OnClick
	nsDialogs::Create /NOUNLOAD 1018
	Pop $9
	${NSD_OnBack} RemoveNextBtnShield
	${NSD_CreateLabel} 0 20u 75% 20u "Please select install type..."
	Pop $0
	System::Call "advapi32::GetUserName(t.r0,*i${NSIS_MAX_STRLEN})i"
	${NSD_CreateRadioButton} 0 40u 75% 15u "Single User ($0) (no right-click shortcut)"
	Pop $R0
	nsDialogs::OnClick $R0 $8
	nsDialogs::SetUserData $R0 0

	${NSD_CreateRadioButton} 0 60u 75% 15u "Portable (no right-click shortcut and startup menu entry)"
	Pop $R1
	nsDialogs::OnClick $R1 $8
	nsDialogs::SetUserData $R1 0

	${NSD_CreateRadioButton} 0 80u 75% 15u "All users (Right-click shortcut too)"
	Pop $R2
	nsDialogs::OnClick $R2 $8
	nsDialogs::SetUserData $R2 1

	${If} $InstMode = 0
		SendMessage $R0 ${BM_CLICK} 0 0
	${ElseIf} $InstMode = 1
		SendMessage $R1 ${BM_CLICK} 0 0
	${ElseIf} $InstMode = 2
		SendMessage $R2 ${BM_CLICK} 0 0
	${EndIf}

	Push $R1 ;store portable radio hwnd on stack
	Push $R2 ;store allusers radio hwnd on stack
	nsDialogs::Show
FunctionEnd

Function InstModeSelectionPage_OnClick
	Pop $1
	nsDialogs::GetUserData $1
	Pop $1
	GetDlgItem $0 $hwndParent 1
	SendMessage $0 ${BCM_SETSHIELD} 0 $1
FunctionEnd

Function InstModeSelectionPage_Leave
	Pop $0  ;get all users hwnd
	Pop $1 ;get portable hwnd
	; push $1 ;put back portable hwnd
	; push $0 ;put back all users hwnd
	${NSD_GetState} $0 $9
	${NSD_GetState} $1 $8
	${If} $8 = 1
		!insertmacro SetInstMode 1
	${ElseIf} $9 = 1
		!insertmacro SetInstMode 2
		${IfNot} ${UAC_IsAdmin}
			GetDlgItem $9 $HWNDParent 1
			System::Call user32::GetFocus()i.s
			EnableWindow $9 0 ;disable next button
			!insertmacro UAC_PageElevation_RunElevated
			EnableWindow $9 1
			System::Call user32::SetFocus(is) ;Do we need WM_NEXTDLGCTL or can we get away with this hack?
			${If} $2 = 0x666666 ;our special return, the new process was not admin after all
				MessageBox mb_iconExclamation "You need to login with an account that is a member of the admin group to continue..."
				Abort
			${ElseIf} $0 = 1223 ;cancel
				Abort
			${Else}
				${If} $0 <> 0
					${If} $0 = 1062
						MessageBox MB_ICONSTOP "Unable to elevate, Secondary Logon service not running!"
					${Else}
						MessageBox MB_ICONSTOP "Unable to elevate, error $0"
					${EndIf}
				Abort
				${EndIf}
			${EndIf}
			Quit ;We now have a new process, the install will continue there, we have nothing left to do here
	        ${EndIf}
	${Else}
		!insertmacro SetInstMode 0
	${EndIf}

	${If} $InstMode <> 1
		ReadRegStr $R0 SHCTX \
		"Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
		"UninstallString"
		ReadRegStr $R1 SHCTX \
		"Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
		"DisplayVersion"
		ReadRegStr $R2 SHCTX \
		"Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
		"InstallLocation"

		StrCmp $R0 "" done
		MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION \
		'${APPNAME} is already installed and must be uninstalled to \
		install another version.\
		$\n$\nClick "OK" to \
		uninstall it or "Cancel" to cancel the installation.' \
		IDOK uninst
		Abort
		uninst:
		${If} $R1 == ""
			; DisplayVersion was not defined in HSpy 0.6,
			; so it is 0.6
			;Do not use the 0.6 uninstaller to uninstall
			;HyperSpy because is buggy.
			;The following code uninstalls correctly HSpy 0.6
			${If} $InstMode = 2
				Exec 'cmd.exe /C ""$R2\WinPython Command Prompt.exe" uninstall_hyperspy_here & exit"'
				Sleep 3000
			${EndIf}

			; Execute fixed, embedded 06 uninstaller
			ClearErrors
			File "continuous_integration\NSISPlugins\uninstaller_06.exe"
			ExecWait '"$INSTDIR\uninstaller_06.exe" /S _?=$INSTDIR'
			IfErrors error_uninstalling_06
			Goto uninstall_06_complete
			error_uninstalling_06:
				Abort
			uninstall_06_complete:
			Delete $INSTDIR\uninstaller_06.exe
		${Else}
			Exec $R0
		${EndIf}
	done:
	${EndIf}
FunctionEnd

; -------- Sections -----------------------------------------------------------
Section "Required Files"
SectionIn RO
	SetOutPath "${APP_INSTDIR}"
	File /r "${WINPYTHON_PATH}\*"
	${If} $InstMode = 2
	; Create right-click context menu entries for Hyperspy Here
		Exec 'cmd.exe /C ""${APP_INSTDIR}\WinPython Command Prompt.exe" install_hyperspy_here & exit"'
	Sleep 3000
	${EndIf}

	${If} $InstMode <> 1
	; Create StartMenu shortcuts
		CreateDirectory "$SMPROGRAMS\${APPNAME}"
		CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "${APP_INSTDIR}\${PYTHON_FOLDER}\Scripts\hyperspy.bat"
		CreateShortCut "$SMPROGRAMS\${APPNAME}\Uninstall ${APPNAME}.lnk" '"${UNINSTALLER_FULLPATH}"'
		CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME} QtConsole.lnk" "${APP_INSTDIR}\${PYTHON_FOLDER}\Scripts\hyperspy_qtconsole.bat" "" "${APP_INSTDIR}\${PYTHON_FOLDER}\Lib\site-packages\hyperspy\data\hyperspy_qtconsole_logo.ico" 0
		CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME} Notebook.lnk" "${APP_INSTDIR}\${PYTHON_FOLDER}\Scripts\hyperspy_notebook.bat" "" "${APP_INSTDIR}\${PYTHON_FOLDER}\Lib\site-packages\hyperspy\data\hyperspy_notebook_logo.ico" 0

		WriteUninstaller "${UNINSTALLER_FULLPATH}"
		WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" DisplayName "${APPNAME}"
		WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" DisplayVersion "__VERSION__"
		WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" UninstallString "$\"${UNINSTALLER_FULLPATH}$\" /MODE=$InstMode"
		WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" QuietUninstallString "$\"${UNINSTALLER_FULLPATH}$\" /MODE=$InstMode /S"
		WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" InstallLocation "${APP_INSTDIR}"
		WriteRegDWORD SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" NoModify 1
		WriteRegDWORD SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" NoRepair 1
	${EndIf}
SectionEnd


/***************************************************
** Uninstaller
***************************************************/

Function UN.onInit
	Push R0
	${GetParameters} $R0
	${GetOptions} $R0 "/MODE=" $InstMode
	${If} $InstMode = 2
		!insertmacro Go_admin
	${EndIf}

	#Verify the uninstaller - last chance to back out
	MessageBox MB_OKCANCEL "Permanantly remove ${APPNAME}?" IDOK next
		Abort
	next:
FunctionEnd

Section "Uninstall"
	; Currently do not create uninstaller for mode 1, so ignore
	SetOutPath "$TEMP"
	${If} $InstMode = 2
		Exec 'cmd.exe /C ""$INSTDIR\WinPython Command Prompt.exe" uninstall_hyperspy_here & exit"'
		Sleep 3000
	${EndIf}
	!insertmacro __DELETE_MACRO_NAME__ $INSTDIR
	DeleteRegKey SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
	# Remove StartMenu entries
	Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
	Delete "$SMPROGRAMS\${APPNAME}\${APPNAME} QtConsole.lnk"
	Delete "$SMPROGRAMS\${APPNAME}\${APPNAME} Notebook.lnk"
	Delete "$SMPROGRAMS\${APPNAME}\Uninstall ${APPNAME}.lnk"
	RMDir "$SMPROGRAMS\${APPNAME}"

SectionEnd


!insertmacro MUI_LANGUAGE "English"
