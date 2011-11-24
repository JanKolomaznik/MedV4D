/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file ViewerFactory.h
 * @{ 
 **/

#ifndef MASKVIEWER_FACTORY_H
#define MASKVIEWER_FACTORY_H

#include "MedV4D/GUI/widgets/m4dGUIAbstractViewerWidget.h"
#include "MedV4D/GUI/widgets/m4dGUISliceViewerWidget.h"
#include "MedV4D/GUI/widgets/m4dGUIVtkViewerWidget.h"

#include <string>

#include "MedV4D/GUI/utils/ViewerFactory.h"


namespace M4D {
namespace GUI {

template< typename ViewerType >
class GenericViewerFactory: public ViewerFactory
{
  public:

	  GenericViewerFactory (): ViewerFactory( "" )     // TODO - set description right
		  {}

	  M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( unsigned index )
		  { return new ViewerType( index ); }

};


} // namespace GUI
} // namespace M4D

#endif // MASKVIEWER_FACTORY_H


/** @} */

