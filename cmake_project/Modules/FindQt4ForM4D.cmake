IF( ${WIN32} )
	SET( WIN32_USE_PREPARED_PACKAGES_QT TRUE )
ENDIF( ${WIN32} )

SET( QtComponentList "QtCore" "QtGui" "qtmain" "QtOpenGL" )

SET(QT_USE_QTMAIN 1)
SET(QT_USE_QTOPENGL 1)

IF( WIN32_USE_PREPARED_PACKAGES_QT )
	get_filename_component(QT_MOC_EXECUTABLE ${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/moc.exe ABSOLUTE)
	get_filename_component(QT_UIC_EXECUTABLE ${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/uic.exe ABSOLUTE)
	get_filename_component(QT_RCC_EXECUTABLE ${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/rcc.exe ABSOLUTE)
	get_filename_component(QT_QMAKE_EXECUTABLE ${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/qmake.exe ABSOLUTE)	
	
	SET(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} imm32 winmm)
	SET(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES} ws2_32)

	INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/QtMacros.cmake" )
	
	SET(QT_INCLUDE_DIR "${MEDV4D_CMAKE_SOURCE_DIR}/../include/qt/headers")
	SET(QT_QT_INCLUDE_DIR "${MEDV4D_CMAKE_SOURCE_DIR}/../include/qt/headers")
	SET(QT_LIBRARY_DIR "${MEDV4D_CMAKE_SOURCE_DIR}/../lib/qt" )	
	
	FOREACH( component ${QtComponentList} )
		#MESSAGE( "Adding info for Qt component ${component}" )
		STRING( TOUPPER ${component} _COMPONENT )
		SET( QT_USE_${_COMPONENT} 1 )
		SET( QT_${_COMPONENT}_FOUND 1 )
		SET(QT_${_COMPONENT}_LIBRARY "optimized" "${QT_LIBRARY_DIR}/${component}.lib" "debug" "${QT_LIBRARY_DIR}/${component}d.lib" )
		SET(QT_${_COMPONENT}_INCLUDE_DIR "${MEDV4D_CMAKE_SOURCE_DIR}/../include/qt/headers/${component}")
	ENDFOREACH( component )
	
	
	
	SET(QT_USE_FILE ${CMAKE_ROOT}/Modules/UseQt4.cmake)
	
	SET(QT_QTCORE_LIBRARY "optimized" "${QT_LIBRARY_DIR}/QtCore4.lib" "debug" "${QT_LIBRARY_DIR}/QtCored4.lib" )
	SET(QT_QTGUI_LIBRARY "optimized" "${QT_LIBRARY_DIR}/QtGui4.lib" "debug" "${QT_LIBRARY_DIR}/QtGuid4.lib" )
	SET(QT_QTMAIN_LIBRARY "optimized" "${QT_LIBRARY_DIR}/qtmain.lib" "debug" "${QT_LIBRARY_DIR}/qtmaind.lib" )
	SET(QT_QTOPENGL_LIBRARY "optimized" "${QT_LIBRARY_DIR}/QtOpenGL4.lib" "debug" "${QT_LIBRARY_DIR}/QtOpenGLd4.lib" )

	SET( QT_DEFINITIONS "")
	
ELSE( WIN32_USE_PREPARED_PACKAGES_QT )
	FIND_PACKAGE(Qt4 REQUIRED COMPONENTS ${ComponentList})
ENDIF( WIN32_USE_PREPARED_PACKAGES_QT )

#MESSAGE( "Including QT_USE_FILE" )
INCLUDE(${QT_USE_FILE})
