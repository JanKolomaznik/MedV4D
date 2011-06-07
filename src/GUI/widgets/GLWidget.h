#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "GUI/utils/OGLTools.h"
#include <QtOpenGL>
#include "GUI/utils/OpenGLManager.h"



class GLWidget: public QGLWidget
{
	Q_OBJECT;
public:
	GLWidget( QWidget * parent = 0 ):
		QGLWidget( parent, OpenGLManager::getInstance()->getSharedGLWidget() )
	{}

};


#endif /*GLWIDGET_H*/
