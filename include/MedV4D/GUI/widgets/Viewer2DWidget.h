#ifndef _VIEWER_2D_WIDGET_H
#define _VIEWER_2D_WIDGET_H

#include <QtOpenGL/QGLWidget>

struct Drawing2DJobInfo
{
	bool enabled;
};

class Viewer2DWidget: public QGLWidget
{
	Q_OBJECT;
public:
	void
	ProcessDrawingJobInfo( const Drawing2DJobInfo & info );
public slots:

protected:

};

#endif /*_VIEWER_2D_WIDGET_H*/
