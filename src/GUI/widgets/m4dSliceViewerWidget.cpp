#include "m4dSliceViewerWidget.h"

#include <QtGui>
#include <QtOpenGL>

namespace M4D
{
namespace Viewer
{

m4dSliceViewerWidget::m4dSliceViewerWidget(Imaging::Image< unsigned, 3 >::Ptr img, QWidget *parent)
    : QGLWidget(parent)
{
    _image = img;
    _sliceNum = _image->GetDimensionExtents(2).minimum;
    _offset.setX(0);
    _offset.setY(0);
    _lastPos.setX(-1);
    _lastPos.setY(-1);
    _zoomRate = 1.0;
}

m4dSliceViewerWidget::~m4dSliceViewerWidget()
{
}

void
m4dSliceViewerWidget::initializeGL()
{
}

void
m4dSliceViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);
    glLoadIdentity();
    size_t i, j;
    unsigned pixel;
    for ( i = _image->GetDimensionExtents(0).minimum; i <= _image->GetDimensionExtents(0).maximum; ++i )
        for ( j = _image->GetDimensionExtents(1).minimum; j <= _image->GetDimensionExtents(1).maximum; ++j )
	{
            glRasterPos2i( i - _image->GetDimensionExtents(0).minimum, j - _image->GetDimensionExtents(1).minimum );
            pixel = _image->GetElement( i, j, _sliceNum );
	    glDrawPixels( 1, 1, GL_RGBA, GL_UNSIGNED_INT, &pixel );
	}
}

void
m4dSliceViewerWidget::resizeGL(int winW, int winH)
{
    glViewport(0, 0, width(), height());
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, 0, 5);
}

void
m4dSliceViewerWidget::mousePressEvent(QMouseEvent *event)
{
    _lastPos = event->pos();
}

void
m4dSliceViewerWidget::mouseMoveEvent(QMouseEvent *event)
{
    if ( _lastPos.x() == -1 || _lastPos.y() == -1 ) return;

    int dx = event->x() - _lastPos.x();
    int dy = event->y() - _lastPos.y();

    if ( ( event->buttons() & Qt::LeftButton ) &&
         ( event->buttons() & Qt::RightButton ) )
    {
        zoomImage( dy );
    }
    else if (event->buttons() & Qt::LeftButton)
    {
        moveImage( QPoint( dx, dy ) );
    }
    else if (event->buttons() & Qt::RightButton)
    {
        adjustContrast(  dx );
	adjustBrightness( dy );
    }

    _lastPos = event->pos();
}

void
m4dSliceViewerWidget::zoomImage( int amount )
{
    _zoomRate += 0.1 * amount;
    updateGL();
}

void
m4dSliceViewerWidget::wheelEvent(QWheelEvent *event)
{
    int numDegrees = event->delta() / 8;
    int numSteps = numDegrees / 15;
    try
    {
        setSliceNum( _sliceNum + numSteps );
    }
    catch (...)
    {
        //TODO exception handling
    }
}

void
m4dSliceViewerWidget::setSliceNum( size_t num )
{
    if ( num < _image->GetDimensionExtents(2).minimum ||
         num > _image->GetDimensionExtents(2).maximum )
    {
        //TODO exception
    }
    _sliceNum = num;
}

void
m4dSliceViewerWidget::moveImage( QPoint diff )
{
    _offset += diff;
}

void
m4dSliceViewerWidget::adjustBrightness( int amount )
{
    //TODO
}

void
m4dSliceViewerWidget::adjustContrast( int amount )
{
    //TODO
}

} /*namespace Viewer*/
} /*namespace M4D*/
