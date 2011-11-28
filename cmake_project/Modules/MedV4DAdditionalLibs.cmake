


#**************************************************************
IF( ${PYTHON_ENABLED} MATCHES "ON" )
	#Do some checking
	SET(BOOST_PYTHON python)

	FIND_PACKAGE( PythonLibs REQUIRED ) #TODO  handle release/debug
ENDIF( ${PYTHON_ENABLED} MATCHES "ON" )

#**************************************************************
SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system thread ${BOOST_PYTHON})

#**************************************************************
FIND_PACKAGE(OpenGL REQUIRED)

#**************************************************************
FIND_PACKAGE(DCMTK REQUIRED)

#**************************************************************
FIND_PACKAGE( Cg REQUIRED )
SET( CG_SHADER_LIBRARIES ${CG_LIBRARY} ${CG_GL_LIBRARY} )
SET( CG_SHADER_LIBRARY_DIRS "" )
SET( CG_SHADER_INCLUDE_DIRS ${CG_INCLUDE_PATH} )

#**************************************************************
SET( QtComponentList "QtCore" "QtGui" "qtmain" "QtOpenGL" )
SET(QT_USE_QTMAIN 1)
SET(QT_USE_QTOPENGL 1)
FIND_PACKAGE(Qt4 REQUIRED COMPONENTS ${ComponentList})
INCLUDE(${QT_USE_FILE})
#**************************************************************
IF( ${DEVIL_ENABLED} MATCHES "ON" )
	#Do some checking
	FIND_PACKAGE( DevIL REQUIRED )
	SET( DEVIL_LIBRARIES ${DEVIL_LIB_IL} ${DEVIL_LIB_ILU} )
	SET( DEVIL_LIBRARY_DIRS "" )
	SET( DEVIL_INCLUDE_DIRS ${IL_INCLUDE_DIR} )
	#MESSAGE( STATUS "DevIL libs: ${DEVIL_LIBRARIES}"
ENDIF( ${DEVIL_ENABLED} MATCHES "ON" )
#**************************************************************

IF(UNIX)
	SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} wrap )
ENDIF(UNIX)

#**************************************************************
IF(USE_TBB)
	SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} tbb tbbmalloc)
ENDIF(USE_TBB)

#**************************************************************
#**************************************************************
SET( MEDV4D_LIB_DIRS_3RD_PARTY ${DCMTK_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS} ${QT_LIBRARY_DIR} ${CG_SHADER_LIBRARY_DIRS} ${DEVIL_LIBRARY_DIRS} ${ADDITIONAL_LIBRARY_DIR}  )
SET( MEDV4D_LIBRARIES_3RD_PARTY ${DCMTK_LIBRARIES} ${QT_LIBRARIES} ${ADDITIONAL_LIBRARIES} ${OPENGL_LIBRARY} ${Boost_LIBRARIES} ${CG_SHADER_LIBRARIES}	${DEVIL_LIBRARIES} ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES} )
SET( MEDV4D_INCLUDE_DIRS_3RD_PARTY ${DCMTK_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${QT_INCLUDES} ${CG_SHADER_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} ${DEVIL_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} )


