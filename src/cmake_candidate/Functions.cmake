
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



MACRO(TARGET_MEDV4D_APPLICATION app_name)

	SET(SRC_DIR ${APPLICATION_DIR}/${app_name})
	SET(OUTPUT ${app_name})
	SET( mocinput )
	SET( mocoutput )
	SET( rccinput )
	SET( rccoutput )
	
	AUX_SOURCE_DIRECTORY( ${SRC_DIR} sources )

	FILE( GLOB rccinput ${SRC_DIR}/*.qrc )
	FILE( GLOB header_files ${SRC_DIR}/*.h )
	SET_SOURCE_FILES_PROPERTIES(${header_files} PROPERTIES HEADER_FILE_ONLY TRUE)

	
	FILTER_HEADERS_FOR_MOC( "${header_files}" mocinput )

	
	QT4_WRAP_CPP(mocoutput ${mocinput})
	QT4_ADD_RESOURCES(rccoutput ${rccinput} )


	SOURCE_GROUP( ${app_name}_Sources FILES ${sources} )
	SOURCE_GROUP( ${app_name}_Resources FILES  ${rccinput} )
	SOURCE_GROUP( ${app_name}_Header FILES  ${header_files} )

	ADD_EXECUTABLE(${OUTPUT} ${sources} ${mocoutput} ${rccoutput})
	TARGET_LINK_LIBRARIES(${OUTPUT} ${MEDV4D_ALL_LIBRARIES})

	ADD_DEPENDENCIES(${OUTPUT} ${MEDV4D_LIBRARIES})
	IF( DCMTK_OPTIONS )
		SET_TARGET_PROPERTIES( ${OUTPUT} PROPERTIES COMPILE_DEFINITIONS ${DCMTK_OPTIONS} )	
	ENDIF( DCMTK_OPTIONS )

ENDMACRO(TARGET_MEDV4D_APPLICATION app_name)
