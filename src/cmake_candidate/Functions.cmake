
FUNCTION(FILTER_HEADERS_FOR_MOC inputlist outputlist)
	SET(${outputlist} "" PARENT_SCOPE)
	SET(tmp_list "" )
	
	FOREACH(header ${inputlist})
		FILE(STRINGS ${header} file_strings REGEX "Q_OBJECT")
		IF( ${file_strings} MATCHES "Q_OBJECT" )
			SET( tmp_list "${tmp_list}" "${header}" )
		ENDIF( ${file_strings} MATCHES "Q_OBJECT" )
	ENDFOREACH(header ${inputlist})
	SET(${outputlist} ${tmp_list} PARENT_SCOPE)

ENDFUNCTION(FILTER_HEADERS_FOR_MOC)

