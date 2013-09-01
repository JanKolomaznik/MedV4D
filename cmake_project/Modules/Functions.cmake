INCLUDE(${CMAKE_ROOT}/Modules/CMakeParseArguments.cmake)

function(M4D_QT5_WRAP_UI outfiles )
    set(options)
    set(oneValueArgs)
    set(multiValueArgs OPTIONS)

    cmake_parse_arguments(_WRAP_UI "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(ui_files ${_WRAP_UI_UNPARSED_ARGUMENTS})
    set(ui_options ${_WRAP_UI_OPTIONS})

    foreach(it ${ui_files})
        get_filename_component(outfile ${it} NAME_WE)
        get_filename_component(infile ${it} ABSOLUTE)
        set(outfile ${MEDV4D_GENERATED_HEADERS_DIR}/ui_${outfile}.h)
        add_custom_command(OUTPUT ${outfile}
          COMMAND ${Qt5Widgets_UIC_EXECUTABLE}
          ARGS ${ui_options} -o ${outfile} ${infile}
          MAIN_DEPENDENCY ${infile} VERBATIM)
        list(APPEND ${outfiles} ${outfile})
    endforeach()
    set(${outfiles} ${${outfiles}} PARENT_SCOPE)
endfunction()


FUNCTION(FILTER_HEADERS_FOR_MOC inputlist outputlist)
	#SET(${outputlist} "" PARENT_SCOPE)
	SET(tmp_list "" )
	FOREACH(header ${inputlist})
		FILE(STRINGS ${header} file_strings REGEX "Q_OBJECT")
		IF( "${file_strings}" MATCHES "Q_OBJECT" )
			SET( tmp_list ${tmp_list} "${header}" )
		ENDIF( "${file_strings}" MATCHES "Q_OBJECT" )
	ENDFOREACH(header ${inputlist})
	#message( "+++++++++++++++++++++++${tmp_list}" )
	SET(${outputlist} ${tmp_list} PARENT_SCOPE)
ENDFUNCTION(FILTER_HEADERS_FOR_MOC)

FUNCTION(FILTER_FILES_FOR_STRING str inputlist outputlistMatching outputlistNotMatching)
	SET(tmp_listMatching "" )
	SET(tmp_listNotMatching "" )
	FOREACH(fileName ${inputlist})
		IF( "${fileName}" MATCHES ${str} )
			SET( tmp_listMatching ${tmp_listMatching} "${fileName}" )
			#message( "match ${fileName}" )
		ELSE( "${fileName}" MATCHES ${str} )
			SET( tmp_listNotMatching ${tmp_listNotMatching} "${fileName}" )
			#message( "noatch ${fileName}" )
		ENDIF( "${fileName}" MATCHES ${str} )
	ENDFOREACH(fileName ${inputlist})
	SET(${outputlistMatching} ${tmp_listMatching} PARENT_SCOPE)
	SET(${outputlistNotMatching} ${tmp_listNotMatching} PARENT_SCOPE)
ENDFUNCTION(FILTER_FILES_FOR_STRING)


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

MACRO (CREATE_LIB_NAMES_FROM_TARGET_NAMES arg_input arg_output)
	SET(${arg_output} )
	FOREACH(currentName ${${arg_input}} )
		SET( ${arg_output} ${${arg_output}} "optimized" "${MEDV4D_LIBRARY_PREFIX}${currentName}${MEDV4D_RELEASE_POSTFIX}" "debug" "${MEDV4D_LIBRARY_PREFIX}${currentName}${MEDV4D_DEBUG_POSTFIX}" )		
	ENDFOREACH(currentName) 
ENDMACRO (CREATE_LIB_NAMES_FROM_TARGET_NAMES)

#MACRO(TARGET_MEDV4D_PROGRAM_DIRS prog_name source_dirs )
#message( "prog_name ${prog_name}" )
#message( "source_dir ${${source_dirs}}" )
#FOREACH(currentName ${${source_dirs}} )
#message( "${currentName}" )
#
#ENDFOREACH(currentName)
#ENDMACRO(TARGET_MEDV4D_PROGRAM_DIRS)

FUNCTION(TARGET_MEDV4D_PROGRAM prog_name source_dir )
	SET(SRC_DIR ${source_dir})
	SET(OUTPUT_NAME ${prog_name})
	SET( mocinput )
	SET( mocoutput )
	SET( rccinput )
	SET( rccoutput )
	SET( uiinput )
	SET( uioutput )
	
	MESSAGE( STATUS "Preparing build system for ${prog_name} (source directory : ${SRC_DIR})" )
	
	#AUX_SOURCE_DIRECTORY( ${SRC_DIR} sources )
	FILE( GLOB_RECURSE sources "${SRC_DIR}/*.cpp" )
	FILE( GLOB_RECURSE rccinput "${SRC_DIR}/*.qrc" )
	FILE( GLOB_RECURSE uiinput "${SRC_DIR}/*.ui" )
	FILE( GLOB_RECURSE header_files "${SRC_DIR}/*.h" "${SRC_DIR}/*.hpp" )
	FILE( GLOB_RECURSE tcc_files "${SRC_DIR}/*.tcc" )
	
	
	
	SET_SOURCE_FILES_PROPERTIES(${tcc_files} PROPERTIES HEADER_FILE_ONLY TRUE)
	FILTER_HEADERS_FOR_MOC( "${header_files}" mocinput )
	
	
	#message( "++++++++++++++ ${moc_options}" )
	QT4_WRAP_CPP(mocoutput ${mocinput})
	QT4_ADD_RESOURCES(rccoutput ${rccinput} )
	QT4_WRAP_UI(uioutput ${uiinput} )
	
	#message( "++++Sources: ${sources}" )
	#message( "++++Rccinput: ${rccinput}" )
	#message( "++++Mocinput: ${mocinput}" )
	#message( "++++UIinput: ${uiinput}" )
	#message( "++++HeaderFiles: ${header_files}" )
	#message( "++++TCCFiles: ${tcc_files}" )
	#message( "++++Rccoutput: ${rccoutput}" )
	#message( "++++UIoutput: ${uioutput}" )
	
	ADD_DEFINITIONS( ${MEDV4D_COMPILE_DEFINITIONS} )
	
	SOURCE_GROUP( ${prog_name}_Sources FILES "" ${sources} )
	SOURCE_GROUP( ${prog_name}_Header FILES  "" ${header_files} ${tcc_files} )
	SOURCE_GROUP( ${prog_name}_UI FILES  "" ${uiinput})
	SOURCE_GROUP( ${prog_name}_Resources FILES  "" ${rccinput} )
	SOURCE_GROUP( ${prog_name}_Generated FILES "" ${mocoutput} ${rccoutput} ${uioutput} )
	
	INCLUDE_DIRECTORIES( ${SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${MEDV4D_TMP_DIR} )
	ADD_EXECUTABLE(${OUTPUT_NAME} ${sources} ${uioutput}  ${header_files} ${tcc_files} ${mocoutput} ${rccoutput} ) #${uiinput} ${rccinput} )
	TARGET_LINK_LIBRARIES(${OUTPUT_NAME} ${MEDV4D_ALL_LIBRARIES} ${MEDV4D_ALL_LIBRARIES})

	ADD_DEPENDENCIES(${OUTPUT_NAME} ${MEDV4D_LIB_TARGETS})
	IF( DCMTK_OPTIONS )
		SET_TARGET_PROPERTIES( ${OUTPUT_NAME} PROPERTIES COMPILE_DEFINITIONS ${DCMTK_OPTIONS} )	
	ENDIF( DCMTK_OPTIONS )
ENDFUNCTION(TARGET_MEDV4D_PROGRAM prog_name source_dir)

FUNCTION(TARGET_MEDV4D_PROGRAM_NONRECURSIVE prog_name source_dir )
	SET(SRC_DIR ${source_dir})
	SET(OUTPUT_NAME ${prog_name})
	SET( mocinput )
	SET( mocoutput )
	SET( rccinput )
	SET( rccoutput )
	SET( uiinput )
	SET( uioutput )
	
	MESSAGE( STATUS "Preparing build system for ${prog_name} (source directory : ${SRC_DIR})" )
	
	#AUX_SOURCE_DIRECTORY( ${SRC_DIR} sources )
	FILE( GLOB sources "${SRC_DIR}/*.cpp" )
	FILE( GLOB rccinput "${SRC_DIR}/*.qrc" )
	FILE( GLOB uiinput "${SRC_DIR}/*.ui" )
	FILE( GLOB header_files "${SRC_DIR}/*.h" "${SRC_DIR}/*.hpp" )
	FILE( GLOB tcc_files "${SRC_DIR}/*.tcc" )
	
	
	
	SET_SOURCE_FILES_PROPERTIES(${tcc_files} PROPERTIES HEADER_FILE_ONLY TRUE)
	FILTER_HEADERS_FOR_MOC( "${header_files}" mocinput )
	
	
	#message( "++++++++++++++ ${moc_options}" )
	QT4_WRAP_CPP(mocoutput ${mocinput})
	QT4_ADD_RESOURCES(rccoutput ${rccinput} )
	QT4_WRAP_UI(uioutput ${uiinput} )
	
	#message( "++++Sources: ${sources}" )
	#message( "++++Rccinput: ${rccinput}" )
	#message( "++++Mocinput: ${mocinput}" )
	#message( "++++UIinput: ${uiinput}" )
	#message( "++++HeaderFiles: ${header_files}" )
	#message( "++++TCCFiles: ${tcc_files}" )
	#message( "++++Rccoutput: ${rccoutput}" )
	#message( "++++UIoutput: ${uioutput}" )
	
	ADD_DEFINITIONS( ${MEDV4D_COMPILE_DEFINITIONS} )
	
	SOURCE_GROUP( ${prog_name}_Sources FILES "" ${sources} )
	SOURCE_GROUP( ${prog_name}_Header FILES  "" ${header_files} ${tcc_files} )
	SOURCE_GROUP( ${prog_name}_UI FILES  "" ${uiinput})
	SOURCE_GROUP( ${prog_name}_Resources FILES  "" ${rccinput} )
	SOURCE_GROUP( ${prog_name}_Generated FILES "" ${mocoutput} ${rccoutput} ${uioutput} )
	
	INCLUDE_DIRECTORIES( ${SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}  ${MEDV4D_TMP_DIR} )
	ADD_EXECUTABLE(${OUTPUT_NAME} ${sources} ${uioutput}  ${header_files} ${tcc_files} ${mocoutput} ${rccoutput} ) #${uiinput} ${rccinput} )
	TARGET_LINK_LIBRARIES(${OUTPUT_NAME} ${MEDV4D_ALL_LIBRARIES} ${MEDV4D_ALL_LIBRARIES})

	ADD_DEPENDENCIES(${OUTPUT_NAME} ${MEDV4D_LIB_TARGETS})
	IF( DCMTK_OPTIONS )
		SET_TARGET_PROPERTIES( ${OUTPUT_NAME} PROPERTIES COMPILE_DEFINITIONS ${DCMTK_OPTIONS} )	
	ENDIF( DCMTK_OPTIONS )
ENDFUNCTION(TARGET_MEDV4D_PROGRAM_NONRECURSIVE prog_name source_dir)

MACRO(MEDV4D_LIBRARY_TARGET_PREPARATION libName libSrcDir libHeaderDir)

	FILE( GLOB SrcFiles "${libSrcDir}/*.cpp" )
	FILE( GLOB Header_files "${libHeaderDir}/*.h" "${libHeaderDir}/*.hpp" )
	FILE( GLOB Tcc_files "${libHeaderDir}/*.tcc" )

	SOURCE_GROUP( ${libName}_Headers FILES ${Header_files} )
	SOURCE_GROUP( ${libName}_Tcc FILES ${Tcc_files} )
	SOURCE_GROUP( ${libName}_Sources FILES ${SrcFiles}  )
	
	ADD_LIBRARY(${libName} ${SrcFiles} ${Header_files} ${Tcc_files})
	SET_TARGET_PROPERTIES( ${libName} PROPERTIES DEBUG_POSTFIX ${MEDV4D_DEBUG_POSTFIX} ) 
	SET_TARGET_PROPERTIES( ${libName} PROPERTIES RELEASE_POSTFIX ${MEDV4D_RELEASE_POSTFIX} ) 

ENDMACRO(MEDV4D_LIBRARY_TARGET_PREPARATION)

#ADD_MEDV4D_EXECUTABLE( Viewer SOURCES ${VIEWER_SOURCES} HEADERS ${VIEWER_HEADERS} UI ${VIEWER_UI} )
FUNCTION(ADD_MEDV4D_EXECUTABLE prog_name)
	set(groups SOURCES HEADERS UIS RESOURCES)
	cmake_parse_arguments(MEDV4D_EXECUTABLE "" "" "${groups}" ${ARGN} )
	
	#MESSAGE( STATUS "${prog_name} sources : ${MEDV4D_EXECUTABLE_SOURCES}" )
	#MESSAGE( STATUS "${prog_name} headers : ${MEDV4D_EXECUTABLE_HEADERS}" )

	#FILTER_HEADERS_FOR_MOC( "${MEDV4D_EXECUTABLE_HEADERS}" mocinput )
	#QT4_WRAP_CPP(mocoutput ${mocinput}  OPTIONS -DBOOST_TT_HAS_OPERATOR_HPP_INCLUDED ) #Define is workaround for moc bug
	#QT4_ADD_RESOURCES(rccoutput ${MEDV4D_EXECUTABLE_RESOURCES} )
	M4D_QT5_WRAP_UI(uioutput ${MEDV4D_EXECUTABLE_UIS} )
	ADD_EXECUTABLE( ${prog_name} ${MEDV4D_EXECUTABLE_SOURCES} ${uioutput} ${MEDV4D_EXECUTABLE_HEADERS} )
	qt5_use_modules(${prog_name} Widgets OpenGL)
ENDFUNCTION(ADD_MEDV4D_EXECUTABLE prog_name)
