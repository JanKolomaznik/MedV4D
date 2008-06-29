#include "GUI/m4dSliceViewerWidget.h"

#include <QtGui>
#include <QtOpenGL>

namespace M4D
{
namespace Viewer
{

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< int16, 3 > >& conn, QWidget *parent)
    : QGLWidget(parent)
{
    conn.ConnectConsumer( _inPort );
    _sliceNum = Imaging::Image< int16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(2).minimum;
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
m4dSliceViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);
    glLoadIdentity();
    size_t i, j;
    glRasterPos2i( 0, 0 );
    size_t height, width, depth;
    int stride;
    const int16* pixel = Imaging::Image< int16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
    pixel += ( _sliceNum - Imaging::Image< int16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(2).minimum ) * height * width;
    glDrawPixels( width, height, GL_LUMINANCE, GL_UNSIGNED_SHORT, pixel );
    glPixelZoom( _zoomRate, _zoomRate );
    glTranslatef( _offset.x(), _offset.y(), 0 );
    glTranslatef( -( _zoomRate - 1.0 ) * width, -( _zoomRate - 1.0 ) * height, 0 );
    glFlush();
}

void
m4dSliceViewerWidget::resizeGL(int winW, int winH)
{
    glViewport(0, 0, width(), height());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (double)winW, 0.0, (double)winH, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW); 
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
        zoomImage( -dy );
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

    updateGL();

    _lastPos = event->pos();
}

void
m4dSliceViewerWidget::zoomImage( int amount )
{
    _zoomRate += 0.01 * amount;
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
        //TODO handle
    }
    updateGL();

}

void
m4dSliceViewerWidget::setSliceNum( size_t num )
{
    if ( num < Imaging::Image< int16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(2).minimum ||
         num >= Imaging::Image< int16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(2).maximum )
    {
    	throw ErrorHandling::ExceptionBase( "Index out of bounds." );
    }
    _sliceNum = num;
}

void
m4dSliceViewerWidget::moveImage( QPoint diff )
{
    _offset += diff;
    updateGL();

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