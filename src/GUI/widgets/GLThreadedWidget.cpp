
#include "GUI/widgets/GLThreadedWidget.h"

#include <QtGui/QResizeEvent>

GLThreadedWidget::GLThreadedWidget(QWidget *parent)
    : QGLWidget(parent)
{
	_renderingThread = new RenderingThread( *this );
	setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));

	// Buffer swap is handled in the rendering thread
	setAutoBufferSwap(false);

	// start the rendering thread
	InitRenderingThread();
}

GLThreadedWidget::~GLThreadedWidget()
{
	StopRenderingThread();
}

void 
GLThreadedWidget::InitRenderingThread( )
{
    // start the rendering thread
    _renderingThread->start();
    // wake the waiting render thread
    _renderCondition.wakeAll();
}

void 
GLThreadedWidget::StopRenderingThread()
{
    // request stopping
    _renderingThread->stop();
    // wake up render thread to actually perform stopping
    _renderCondition.wakeAll();
    // wait till the thread has exited
    _renderingThread->wait();
}

void 
GLThreadedWidget::initializeGL()
{
    // typical OpenGL init
    // see OpenGL documentation for an explanation
    glClearColor(0,0,0,1);
    glShadeModel(GL_FLAT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void 
GLThreadedWidget::resizeGL(int width, int height)
{
    // nothing special
    // see OpenGL documentation for an explanation
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    GLfloat x = (GLfloat)width / height;
    glFrustum(-x, x, -1.0, 1.0, 4.0, 15.0);
    glMatrixMode(GL_MODELVIEW);
}

void 
GLThreadedWidget::paintGL()
{
    // clear all and draw the scene
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    static const GLfloat coords[6][4][3] =
        {
            {
                {
                    +1.0, -1.0, +1.0
                }
                , { +1.0, -1.0, -1.0 },
                { +1.0, +1.0, -1.0 }, { +1.0, +1.0, +1.0 }
            },
            { { -1.0, -1.0, -1.0 }, { -1.0, -1.0, +1.0 },
              { -1.0, +1.0, +1.0 }, { -1.0, +1.0, -1.0 } },
            { { +1.0, -1.0, -1.0 }, { -1.0, -1.0, -1.0 },
              { -1.0, +1.0, -1.0 }, { +1.0, +1.0, -1.0 } },
            { { -1.0, -1.0, +1.0 }, { +1.0, -1.0, +1.0 },
              { +1.0, +1.0, +1.0 }, { -1.0, +1.0, +1.0 } },
            { { -1.0, -1.0, -1.0 }, { +1.0, -1.0, -1.0 },
              { +1.0, -1.0, +1.0 }, { -1.0, -1.0, +1.0 } },
            { { -1.0, +1.0, +1.0 }, { +1.0, +1.0, +1.0 },
              { +1.0, +1.0, -1.0 }, { -1.0, +1.0, -1.0 } }
        };

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -10.0);
    glRotatef(35, 1.0, 0.0, 0.0);
    glRotatef(12, 0.0, 1.0, 0.0);
    glRotatef(60, 0.0, 0.0, 1.0);

    for (int i = 0; i < 6; ++i)
    {
        // assign names for each surface
        // this make picking work
        glLoadName(i);
        glBegin(GL_QUADS);
        glColor3f(((float)i/6),
		((float)(i+3)/6),
		((float)(i+1)/6));
        for (int j = 0; j < 4; ++j)
        {
            glVertex3f(coords[i][j][0], coords[i][j][1],
                       coords[i][j][2]);
        }
        glEnd();
    }
}

void 
GLThreadedWidget::glDraw()
{
    RenderRequest();
}

void 
GLThreadedWidget::closeEvent( QCloseEvent * event )
{
    // request stopping
    StopRenderingThread();
    // close the widget (base class)
    QGLWidget::closeEvent(event);
}

void GLThreadedWidget::resizeEvent( QResizeEvent * event )
{
    // signal the rendering thread that a resize is needed
    _renderingThread->resizeViewport(event->size());

    RenderRequest();
}

void GLThreadedWidget::lockGLContext( )
{
    // lock the render mutex for the calling thread
    _renderMutex.lock();
    // make the render context current for the calling thread
    makeCurrent();
}

void GLThreadedWidget::unlockGLContext( )
{
    // release the render context for the calling thread
    // to make it available for other threads
    doneCurrent();
    // unlock the render mutex for the calling thread
    _renderMutex.unlock();
}

void GLThreadedWidget::RenderRequest()
{
    // let the wait condition wake up the waiting thread
    _renderCondition.wakeAll();
}

GLThreadedWidget::RenderingThread::RenderingThread( GLThreadedWidget &motherWidget )
	: _motherWidget( motherWidget ), _viewportSize( motherWidget.size() ), _renderFlag( true ), _resizeFlag( true )
{

}

void 
GLThreadedWidget::RenderingThread::resizeViewport( const QSize& size )
{
    // set size and flag to request resizing
    _viewportSize = size;
    _resizeFlag = true;
}

void 
GLThreadedWidget::RenderingThread::stop()
{
    // set flag to request thread to exit
    // REMEMBER: The thread needs to be woken up once
    // after calling this method to actually exit!
    _renderFlag = false;
}

void 
GLThreadedWidget::RenderingThread::run()
{
    // lock the render mutex of the Gl widget
    // and makes the rendering context of the glwidget current in this thread
    _motherWidget.lockGLContext();

    // general GL init
    _motherWidget.initializeGL();

    // do as long as the flag is true
    while( _renderFlag )
    {
        // resize the GL viewport if requested
        if (_resizeFlag)
        {
            _motherWidget.resizeGL(_viewportSize.width(), _viewportSize.height());
            _resizeFlag = false;
        }

        // render code goes here
        _motherWidget.paintGL();

        // swap the buffers of the GL widget
        _motherWidget.swapBuffers();

        _motherWidget.doneCurrent(); // release the GL render context to make picking work!

        // wait until the gl widget says that there is something to render
        // glwidget.lockGlContext() had to be called before (see top of the function)!
        // this will release the render mutex until the wait condition is met
        // and will lock the render mutex again before exiting
        // waiting this way instead of insane looping will not waste any CPU ressources
        _motherWidget._renderCondition.wait(&_motherWidget._renderMutex);

        _motherWidget.makeCurrent(); // get the GL render context back

        // DEACTIVATED -- alternatively render a frame after a certain amount of time
        // prevent to much continous rendering activity
        // msleep(16); //sleep for 16 ms
    }
    // unlock the render mutex before exit
    _motherWidget.unlockGLContext();
}
