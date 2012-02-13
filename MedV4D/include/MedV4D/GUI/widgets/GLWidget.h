#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "MedV4D/GUI/utils/OGLTools.h"
#include <QtOpenGL>
#include "MedV4D/GUI/managers/OpenGLManager.h"



class GLWidget: public QGLWidget
{
	Q_OBJECT;
public:
	GLWidget( QWidget * parent = 0 ):
		QGLWidget( parent, OpenGLManager::getInstance()->getSharedGLWidget() )
	{}

};


#endif /*GLWIDGET_H*/
