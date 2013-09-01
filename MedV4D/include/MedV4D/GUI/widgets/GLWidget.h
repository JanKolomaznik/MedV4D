#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QtOpenGL/QGLWidget>



class GLWidget: public QGLWidget
{
	Q_OBJECT;
public:
	GLWidget( QWidget * parent = 0 );

};


#endif /*GLWIDGET_H*/
