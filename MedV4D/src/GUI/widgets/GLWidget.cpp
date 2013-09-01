#include <soglu/OGLTools.hpp>
#include "MedV4D/GUI/managers/OpenGLManager.h"
#include "MedV4D/GUI/widgets/GLWidget.h"


GLWidget::GLWidget( QWidget * parent):
	QGLWidget( parent, OpenGLManager::getInstance()->getSharedGLWidget() )
{}
