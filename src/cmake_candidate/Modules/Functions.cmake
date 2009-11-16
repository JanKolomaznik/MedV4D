
FUNCTION(FILTER_HEADERS_FOR_MOC inputlist outputlist)
	SET(${outputlist} "" PARENT_SCOPE)
	SET(tmp_list "" )
	FOREACH(header ${inputlist})
		FILE(STRINGS ${header} file_strings REGEX "Q_OBJECT")
		IF( "${file_strings}" MATCHES "Q_OBJECT" )
			SET( tmp_list "${tmp_list}" "${header}" )
		ENDIF( "${file_strings}" MATCHES "Q_OBJECT" )
	ENDFOREACH(header ${inputlist})
	SET(${outputlist} ${tmp_list} PARENT_SCOPE)
ENDFUNCTION(FILTER_HEADERS_FOR_MOC)



FUNCTION(APPEND_STRING_TO_LIST_MEMBERS app lst outputlist)
	SET(tmp_list "" )
	FOREACH(member ${${lst}} )
		SET(tmp_list ${tmp_list} "${member}${app}" )
	ENDFOREACH(member ${lst})
	SET(${outputlist} ${tmp_list} PARENT_SCOPE)
ENDFUNCTION(APPEND_STRING_TO_LIST_MEMBERS)

FUNCTION(PREPEND_STRING_TO_LIST_MEMBERS prep lst outputlist)
	SET(tmp_list "" )
	FOREACH(member ${${lst}} )
		SET(tmp_list ${tmp_list} "${prep}${member}")
	ENDFOREACH(member ${lst})
	SET(${outputlist} ${tmp_list} PARENT_SCOPE)
ENDFUNCTION(PREPEND_STRING_TO_LIST_MEMBERS)

FUNCTION(INSERT_KEYWORD_BEFORE_EACH_MEMBER kword lst outputlist)
	SET(tmp_list "" )
	FOREACH(member ${${lst}} )
		SET(tmp_list ${tmp_list} "${kword}" "${member}" )
	ENDFOREACH(member ${lst})
	SET(${outputlist} ${tmp_list} PARENT_SCOPE)
ENDFUNCTION(INSERT_KEYWORD_BEFORE_EACH_MEMBER)



FUNCTION(TARGET_MEDV4D_PROGRAM prog_name source_dir )
	SET(SRC_DIR ${source_dir})
	SET(OUTPUT ${prog_name})
	SET( mocinput )
	SET( mocoutput )
	SET( rccinput )
	SET( rccoutput )
	SET( uiinput )
	SET( uioutput )
	
	AUX_SOURCE_DIRECTORY( ${SRC_DIR} sources )
	#FILE( GLOB rccinput ${SRC_DIR}/*.qrc )
	#FILE( GLOB uiinput ${SRC_DIR}/*.ui )
	#FILE( GLOB header_files ${SRC_DIR}/*.h )
	FILE( GLOB_RECURSE rccinput ${SRC_DIR}/*.qrc )
	FILE( GLOB_RECURSE uiinput ${SRC_DIR}/*.ui )
	FILE( GLOB_RECURSE header_files ${SRC_DIR}/*.h )
	#message( "------------------${header_files}" )
	
	SET_SOURCE_FILES_PROPERTIES(${header_files} PROPERTIES HEADER_FILE_ONLY TRUE)
	FILTER_HEADERS_FOR_MOC( "${header_files}" mocinput )
	
	QT4_WRAP_CPP(mocoutput ${mocinput})
	QT4_ADD_RESOURCES(rccoutput ${rccinput} )
	QT4_WRAP_UI(uioutput ${uiinput} )

	SOURCE_GROUP( ${prog_name}_Sources FILES ${sources} )
	SOURCE_GROUP( ${prog_name}_Resources FILES  ${rccinput} )
	SOURCE_GROUP( ${prog_name}_UI FILES  ${uiinput} )
	SOURCE_GROUP( ${prog_name}_Header FILES  ${header_files} )

	#Message( "---------- ${OUTPUT} ${sources} ${mocoutput} ${rccoutput}" )
	ADD_EXECUTABLE(${OUTPUT} ${sources} ${mocoutput} ${rccoutput} ${uioutput})
	TARGET_LINK_LIBRARIES(${OUTPUT} ${MEDV4D_ALL_LIBRARIES})

	ADD_DEPENDENCIES(${OUTPUT} ${MEDV4D_LIBRARIES})
	IF( DCMTK_OPTIONS )
		SET_TARGET_PROPERTIES( ${OUTPUT} PROPERTIES COMPILE_DEFINITIONS ${DCMTK_OPTIONS} )	
	ENDIF( DCMTK_OPTIONS )
ENDFUNCTION(TARGET_MEDV4D_PROGRAM prog_name source_dir)

MACRO(MEDV4D_LIBRARY_TARGET_PREPARATION libName libSrcDir libHeaderDir)

	Message( "${libName}" )
	Message( "${libSrcDir}" )
	Message( "${libHeaderDir}" )
	Message( "-------------------------------------" )
	#AUX_SOURCE_DIRECTORY(${libSrcDir} SrcFiles )
	FILE( GLOB SrcFiles "${libSrcDir}/*.cpp" )
	FILE( GLOB Header_files "${libHeaderDir}/*.h" )

	#message( "${Header_files}" )
	#message( "${SrcFiles}" )
	SOURCE_GROUP( ${libName}_Headers FILES ${Header_files} )
	#SOURCE_GROUP( ${libName}_Sources FILES ${SrcFiles}  )
	SOURCE_GROUP( ${libName}_Sources FILES ${SrcFiles}  )
	
	ADD_LIBRARY(${libName} ${SrcFiles} ${Header_files})

ENDMACRO(MEDV4D_LIBRARY_TARGET_PREPARATION)

