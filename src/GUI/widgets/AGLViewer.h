#ifndef A_GL_VIEWER_H
#define A_GL_VIEWER_H

#include "GUI/widgets/AViewer.h"
#include <QtGui>
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{


class AGLViewer: public M4D::GUI::GLWidget, public M4D::GUI::Viewer::AGUIViewer
{
public:
	QWidget* 
	CastToQWidget()
	{ return static_cast< QWidget *>( this ); }

};


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

#endif /*A_GL_VIEWER_H*/

