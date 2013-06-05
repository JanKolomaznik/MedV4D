#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <soglu/OGLTools.hpp>
//#include "MedV4D/GUI/utils/OGLTools.h"
#include <QGLWidget>
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
