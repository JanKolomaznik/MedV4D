#ifndef BASIC_SLICE_VIEWER_H
#define BASIC_SLICE_VIEWER_H

#include "GUI/widgets/AGUIViewer.h"
#include <QtGui>
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{

class BasicSliceViewer : 
	public ViewerConstructionKit<   QGLWidget, 
					PortInterfaceHelper< mpl::vector< AImage >
					>
{
	Q_OBJECT;
public:


protected:

private:
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*BASIC_SLICE_VIEWER_H*/



