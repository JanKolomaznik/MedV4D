#ifndef _GL_THREADED_WIDGET_H
#define _GL_THREADED_WIDGET_H

#include "GUI/utils/OGLTools.h"
#include <QtOpenGL/QGLWidget>
#include <QtCore/QWaitCondition>
#include <QtCore/QMutex>
#include <QtCore/QThread>

class GLThreadedWidget : public QGLWidget
{
	Q_OBJECT;
public:
	GLThreadedWidget( QWidget * parent = NULL );

	~GLThreadedWidget();

	/** Call this method before doing any OpenGL rendering from a thread.
	* This method will aquire the GL rendering context for the calling thread.
	* Rendering will only be possible from this thread until unlockGLContext()
	* is called from the same thread.
	*/
	void 
	lockGLContext();
	
	/** Call this method to release the rendering context again after calling lockGLContext().
	*/
	void 
	unlockGLContext();

	/** Returns a reference to the render wait condition.
	* This is only for internal purpose (render thread communication)
	*/
	QWaitCondition& 
	GetRenderCondition();
	
	/** Returns a reference to the render context mutex.
	* This is only for internal purpose (render thread communication)
	*/
	QMutex& 
	GetRenderMutex();

public slots:
	
	/** Cause the rendering thread to render one frame of the OpenGL scene.
	* This method is thread save.
	* \warning If the rendering thread is currently rendering (not idle) when this method is called
	* NO additional new frame will be rendered afterwards!
	*/
	void 
	RenderRequest();
protected:
	void
	glDraw();

	void
	glInit(){}

	/**
	 * Will be called from rendering thread.
	 **/
	void 
	initializeGL();
	
	/**
	 * Will be called from rendering thread.
	 **/
	void
	resizeGL( int width, int height );

	/**
	 * Will be called from rendering thread.
	 **/
	void	
	paintGL();

	
	class RenderingThread: public QThread
	{
	public:
		RenderingThread( GLThreadedWidget &motherWidget );

		void 
		run();

		void 
		stop();

		void 
		resizeViewport(const QSize& _size);
	protected:
		GLThreadedWidget 	&_motherWidget;
		QSize			_viewportSize;
		/** Keep the thread running as long this flag is true. */
		volatile bool 		_renderFlag;
		/** Perform a resize when this flag is true. */
		volatile bool 		_resizeFlag;
	};

	void 
	InitRenderingThread();

	/** 
	 * Stops the rendering thread of the widget. 
	 **/
	void 
	StopRenderingThread();

	/** 
	 * Calls render() if the widget recieves a paint event. 
	 **/
	/*void 
	paintEvent(QPaintEvent*);*/
	
	/** 
	 * Requests a GL viewport resize from the rendering thread. 
	 **/
	void 
	resizeEvent(QResizeEvent* event);

	void
	closeEvent( QCloseEvent * event );
	

	RenderingThread	*_renderingThread;

	/** 
	 * Mutex for protecting the GL rendering context for multithreading. 
	 **/
	QMutex 		_renderMutex;
	
	/** 
	 * The rendering thread uses this wait condition to save CPU ressources. 
	 **/
	QWaitCondition 	_renderCondition;
};

#endif /*_GL_THREADED_WIDGET_H*/
