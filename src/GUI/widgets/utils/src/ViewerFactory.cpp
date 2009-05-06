/**
 *  @ingroup gui
 *  @file ViewerFactory.cpp
 */
#include "GUI/widgets/utils/ViewerFactory.h"

#include "GUI/widgets/m4dGUISliceViewerWidget.h"
#include "GUI/widgets/m4dGUIVtkViewerWidget.h"

using namespace M4D::Viewer;
using namespace M4D::Imaging;


namespace M4D {
namespace GUI {

/*const unsigned SliceViewerFactory::ID = SLICE_VIEWER_ID;
const char *SliceViewerFactory::description = "";

m4dGUIAbstractViewerWidget *SliceViewerFactory::newViewer ( ConnectionInterface* conn, unsigned index )
{
  return new m4dGUISliceViewerWidget( conn, index );
}


m4dGUIAbstractViewerWidget *SliceViewerFactory::newViewer ( unsigned index )
{
  return new m4dGUISliceViewerWidget( index );
}


const unsigned VtkViewerFactory::ID = VTK_VIEWER_ID;
const char *VtkViewerFactory::description = ""; 


m4dGUIAbstractViewerWidget *VtkViewerFactory::newViewer ( ConnectionInterface* conn, unsigned index )
{
  return new m4dGUIVtkViewerWidget( conn, index );
}


m4dGUIAbstractViewerWidget *VtkViewerFactory::newViewer ( unsigned index )
{
  return new m4dGUIVtkViewerWidget( index );
}*/

} // namespace GUI
} // namespace M4D

