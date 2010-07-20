#ifndef A_2D_VIEWER_H
#define A_2D_VIEWER_H

#include "GUI/widgets/AGUIViewer.h"
#include <QtGui>
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{

class A2DViewer: public AGUIViewer, public I3DViewer
{

};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*A_2D_VIEWER_H*/


