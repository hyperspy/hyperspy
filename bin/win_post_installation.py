#commons_sm = get_special_folder_path("CSIDL_COMMON_STARTMENU")
#local_sm = get_special_folder_path("CSIDL_STARTMENU")
#to_print = [commons_sm, local_sm, ]

##for item in to_print:
##    print(item)

# create_shortcut(target, description, filename[, arguments[, workdir[, iconpath[, iconindex]]]])

  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  createShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\Scripts\hyperspy.bat"
  createShortCut "$SMPROGRAMS\${APPNAME}\Update ${APPNAME}.lnk" "$INSTDIR\Scripts\hyperspy_update.bat"
