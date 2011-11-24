/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file ViewerFactory.h
 * @{ 
 **/

#ifndef VIEWER_FACTORY_H
#define VIEWER_FACTORY_H

#include "MedV4D/GUI/widgets/m4dGUIAbstractViewerWidget.h"
#include "MedV4D/GUI/widgets/m4dGUISliceViewerWidget.h"
#include "MedV4D/GUI/widgets/m4dGUIVtkViewerWidget.h"

#include <string>


namespace M4D {
namespace GUI {

class ViewerFactory
{ 
  public:

	  virtual ~ViewerFactory ()
		  {}

	  virtual M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( M4D::Imaging::ConnectionInterface* conn, 
                                                                 unsigned index ) = 0;

	  virtual M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( unsigned index ) = 0;

	  std::string getDescription () const
		  { return _description; }

  protected:

	  ViewerFactory ( const std::string &descr ): _description( descr )
		  {}

  private:

	  std::string	_description;
};


template< typename ViewerType >
class GenericViewerFactory: public ViewerFactory
{
  public:

	  GenericViewerFactory (): ViewerFactory( "" )     // TODO - set description right
		  {}

	  M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( M4D::Imaging::ConnectionInterface* conn, 
                                                         unsigned index )
		  { return new ViewerType( conn, index ); }

	  M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( unsigned index )
		  { return new ViewerType( index ); }

};

typedef GenericViewerFactory< M4D::Viewer::m4dGUISliceViewerWidget >	SliceViewerFactory;
typedef GenericViewerFactory< M4D::Viewer::m4dGUIVtkViewerWidget >	VtkViewerFactory;

} // namespace GUI
} // namespace M4D

#endif // VIEWER_FACTORY_H


/** @} */

