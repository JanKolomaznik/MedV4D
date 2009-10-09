#--------------------------------------------------------
#       COMMON
AUX_SOURCE_DIRECTORY(../common/src commonSrc )

FILE( GLOB commonHeader_files RELATIVE ${CMAKE_SOURCE_DIR} ../common/*.h ../common/*tcc )

SOURCE_GROUP( Common_Headers FILES ${commonHeader_files} )
SOURCE_GROUP( Common_Sources FILES ${commonSrc}  )

ADD_LIBRARY(Common ${commonSrc} ${commonHeader_files})

#--------------------------------------------------------
#       Imaging
AUX_SOURCE_DIRECTORY(../Imaging/src ImagingSrc )

FILE( GLOB ImagingHeader_files ../Imaging/*.h ../Imaging/*tcc )
SOURCE_GROUP( Imaging_Headers FILES ${ImagingHeader_files} )
SOURCE_GROUP( Imaging_Sources FILES ${ImagingSrc} )

ADD_LIBRARY(Imaging ${ImagingSrc} ${ImagingHeader_files} )
ADD_DEPENDENCIES(Imaging Common)

#--------------------------------------------------------
#       backendForDICOM
AUX_SOURCE_DIRECTORY(../backendForDICOM/src backendForDICOMSrc )

FILE( GLOB backendForDICOMHeader_files ../backendForDICOM/*.h ../backendForDICOM/*tcc )
SOURCE_GROUP( backendForDICOM_Headers FILES ${backendForDICOMHeader_files} )
SOURCE_GROUP( backendForDICOM_Sources FILES ${backendForDICOMSrc} )

ADD_LIBRARY(backendForDICOM ${backendForDICOMSrc} ${backendForDICOMHeader_files})
ADD_DEPENDENCIES(backendForDICOM Common Imaging )

IF( DCMTK_OPTIONS )
	SET_TARGET_PROPERTIES( backendForDICOM PROPERTIES COMPILE_DEFINITIONS ${DCMTK_OPTIONS} )
	#SET_TARGET_PROPERTIES( backendForDICOM PROPERTIES COMPILE_DEFINITIONS "HAVE_SSTREAM" )
ENDIF( DCMTK_OPTIONS )
#--------------------------------------------------------
#       vtkIntegration
AUX_SOURCE_DIRECTORY(../vtkIntegration/src vtkIntegrationSrc )


FILE( GLOB vtkIntegrationHeader_files ../vtkIntegration/*.h ../vtkIntegration/*tcc )
SOURCE_GROUP( vtkIntegration_Headers FILES ${vtkIntegrationHeader_files} )
SOURCE_GROUP( vtkIntegration_Sources FILES ${vtkIntegrationSrc} )

ADD_LIBRARY(vtkIntegration ${vtkIntegrationSrc} ${vtkIntegrationHeader_files})
ADD_DEPENDENCIES(vtkIntegration Common Imaging)

#--------------------------------------------------------
#       GUI
AUX_SOURCE_DIRECTORY(../GUI/widgets/src WidgetsSrc )
AUX_SOURCE_DIRECTORY(../GUI/widgets/ogl/src WidgetsOglSrc )
AUX_SOURCE_DIRECTORY(../GUI/widgets/utils/src WidgetsUtilsSrc )
AUX_SOURCE_DIRECTORY(../GUI/widgets/components/src WidgetsComponentsSrc )

FILE( GLOB rccinput ../GUI/widgets/src/*.qrc )
FILE( GLOB GUI_header_files ../GUI/widgets/*.h ../GUI/widgets/components/*.h ../GUI/widgets/utils/*.h )
FILTER_HEADERS_FOR_MOC( "${GUI_header_files}" mocinput )

QT4_WRAP_CPP(mocoutput ${mocinput})
QT4_ADD_RESOURCES(rccoutput ${rccinput} ) #OPTIONS --root ../tmp

SOURCE_GROUP( GUI_Sources FILES ${WidgetsSrc} ${WidgetsOglSrc} ${WidgetsUtilsSrc} ${WidgetsComponentsSrc} )
SOURCE_GROUP( GUI_Resources FILES  ${rccinput} )
SOURCE_GROUP( GUI_Generated FILES  ${mocoutput} ${rccoutput} )
SOURCE_GROUP( GUI_Headers FILES ${GUI_header_files} )

ADD_LIBRARY(m4dWidgets ${WidgetsSrc} ${WidgetsOglSrc} ${WidgetsUtilsSrc} ${WidgetsComponentsSrc} ${mocoutput} ${rccoutput} ${GUI_header_files} ${rccinput} )
ADD_DEPENDENCIES(m4dWidgets Common Imaging backendForDICOM vtkIntegration)

