#include "GUI/m4dSliceViewerWidget.h"

#include <QtGui>
#include <QtOpenGL>
#include <string.h>

namespace M4D
{
namespace Viewer
{

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn, QWidget *parent)
    : QGLWidget(parent)
{
    SetInputPort( conn );
}

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn, QWidget *parent)
    : QGLWidget(parent)
{
    SetInputPort( conn );
}

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn, QWidget *parent)
    : QGLWidget(parent)
{
    SetInputPort( conn );
}

m4dSliceViewerWidget::~m4dSliceViewerWidget()
{

}

void
m4dSliceViewerWidget::SetColorMode( ColorMode cm )
{
    _colorMode = cm;
}

void
m4dSliceViewerWidget::SetSelectionMode( bool mode )
{
    _selectionMode = mode;
}

bool
m4dSliceViewerWidget::GetSelectionMode()
{
    return _selectionMode;
}

void
m4dSliceViewerWidget::SetInputPort( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn )
{
    conn.ConnectConsumer( _inPort );
    SetColorMode( RGBA_UNSIGNED_BYTE );
    SetParameters();
}

void
m4dSliceViewerWidget::SetInputPort( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn )
{
    conn.ConnectConsumer( _inPort );
    SetColorMode( GRAYSCALE_UNSIGNED_SHORT );
    SetParameters();
}

void
m4dSliceViewerWidget::SetInputPort( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn )
{
    conn.ConnectConsumer( _inPort );
    SetColorMode( GRAYSCALE_UNSIGNED_BYTE );
    SetParameters();
}

void
m4dSliceViewerWidget::SetParameters()
{
    _sliceNum = _inPort.GetAbstractImage().GetDimensionExtents(2).minimum;
    _offset.setX(0);
    _offset.setY(0);
    _lastPos.setX(-1);
    _lastPos.setY(-1);
    _zoomRate = 1.0;
    _brightnessRate = 0.0;
    _contrastRate = 0.0;
    ButtonHandlers bh[] = { NONE, ZOOM, MOVE_H, MOVE_V, ADJUST_C, ADJUST_B };
    SetButtonHandlers( bh );
}

void
m4dSliceViewerWidget::SetButtonHandlers( ButtonHandlers* hnd )
{
    unsigned i;
    for ( i = 0; i < 6; i++ )
    {
        switch (hnd[i])
        {
            case NONE:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::none;
	    break;

	    case ZOOM:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::zoomImage;
	    break;

	    case MOVE_H:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::moveImageH;
	    break;

	    case MOVE_V:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::moveImageV;
	    break;

	    case ADJUST_C:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::adjustContrast;
	    break;

	    case ADJUST_B:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::adjustBrightness;
	    break;
	}
    }
}



void
m4dSliceViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);
    glLoadIdentity();
    unsigned i;
    //glTranslatef( _offset.x(), _offset.y(), 0 );
    glRasterPos2i( 0, 0 );
    size_t height, width, depth;
    int stride;
    //switch ( _colorMode )
    const uint16* pixel = Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
    pixel += ( _sliceNum - Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(2).minimum ) * height * width;
    uint16* black = new uint16[ height * width ];
    memset( black, 0, height * width * sizeof(uint16) );
    uint16* avgLum = new uint16[ height * width ];
    GLfloat avg = 0.0;
    for ( i = 0; i < height * width; ++i ) avg += pixel[i]/65535.;
    avg = avg * 65535. / (float)(width * height);
    for ( i = 0; i < height * width; ++i ) avgLum[i] = (uint16)avg;
    glDrawPixels( width, height, GL_LUMINANCE, GL_UNSIGNED_SHORT, avgLum );
    glAccum( GL_ACCUM, _contrastRate );
    glDrawPixels( width, height, GL_LUMINANCE, GL_UNSIGNED_SHORT, black );
    glAccum( GL_ACCUM, _brightnessRate );
    glDrawPixels( width, height, GL_LUMINANCE, GL_UNSIGNED_SHORT, pixel );
    glAccum( GL_ACCUM, 1.0 - ( _brightnessRate + _contrastRate ) );
    glAccum( GL_RETURN, 1.0 );
    delete[] black;
    delete[] avgLum;
    glPixelZoom( _zoomRate, _zoomRate );
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
	(this->*_buttonMethods[0][0])(dx);
	(this->*_buttonMethods[0][1])(dy);
    }
    else if (event->buttons() & Qt::LeftButton)
    {
	(this->*_buttonMethods[1][0])(dx);
	(this->*_buttonMethods[1][1])(dy);
    }
    else if (event->buttons() & Qt::RightButton)
    {
	(this->*_buttonMethods[2][0])(dx);
	(this->*_buttonMethods[2][1])(dy);
    }

    updateGL();

    _lastPos = event->pos();
}

void
m4dSliceViewerWidget::zoomImage( int amount )
{
    _zoomRate -= 0.01 * amount;
    //_offset.setX( _offset.x() - 
    //		  (int)( amount * ( Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(0).maximum -
    //		  		    Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(0).minimum ) * 0.005 ) );
    //_offset.setY( _offset.y() - 
    //		  (int)( amount * ( Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(1).maximum -
    //		  		    Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(1).minimum ) * 0.005 ) );
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
    if ( num < Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(2).minimum ||
         num >= Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetDimensionExtents(2).maximum )
    {
    	throw ErrorHandling::ExceptionBase( "Index out of bounds." );
    }
    _sliceNum = num;
}

void
m4dSliceViewerWidget::moveImageH( int amount )
{
    _offset.setX( _offset.x() + amount );
}

void
m4dSliceViewerWidget::moveImageV( int amount )
{
    _offset.setY( _offset.y() - amount );
}

void
m4dSliceViewerWidget::adjustBrightness( int amount )
{
    _brightnessRate += ((GLfloat)amount)/((GLfloat)height()/2.0);
}

void
m4dSliceViewerWidget::adjustContrast( int amount )
{
    _contrastRate -= ((GLfloat)amount)/((GLfloat)width()/2.0);
}

void
m4dSliceViewerWidget::none( int amount )
{

}

} /*namespace Viewer*/
} /*namespace M4D*/
