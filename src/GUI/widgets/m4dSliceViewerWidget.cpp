#include "GUI/m4dSliceViewerWidget.h"

#include <QtGui>
#include <QtOpenGL>
#include <GL/glut.h>
#include <string.h>

#define MINIMUM_SELECT_DISTANCE 5

namespace M4D
{
namespace Viewer
{

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn, QWidget *parent)
    : QGLWidget(parent)
{
    setInputPort( conn );
}

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn, QWidget *parent)
    : QGLWidget(parent)
{
    setInputPort( conn );
}

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn, QWidget *parent)
    : QGLWidget(parent)
{
    setInputPort( conn );
}

m4dSliceViewerWidget::~m4dSliceViewerWidget()
{

}

void
m4dSliceViewerWidget::setColorMode( ColorMode cm )
{
    _colorMode = cm;
}

void
m4dSliceViewerWidget::setSelectionMode( bool mode )
{
    _selectionMode = mode;
}

bool
m4dSliceViewerWidget::getSelectionMode()
{
    return _selectionMode;
}

void
m4dSliceViewerWidget::setInputPort( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn )
{
    conn.ConnectConsumer( _inPort );
    setColorMode( rgba_unsigned_byte );
    setParameters();
}

void
m4dSliceViewerWidget::setInputPort( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn )
{
    conn.ConnectConsumer( _inPort );
    setColorMode( grayscale_unsigned_short );
    setParameters();
}

void
m4dSliceViewerWidget::setInputPort( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn )
{
    conn.ConnectConsumer( _inPort );
    setColorMode( grayscale_unsigned_byte );
    setParameters();
}

void
m4dSliceViewerWidget::setParameters()
{
    _sliceNum = _inPort.GetAbstractImage().GetDimensionExtents(2).minimum;
    _offset = QPoint( 0, 0 );
    _lastPos = QPoint( -1, -1 );
    _zoomRate = 1.0;
    _brightnessRate = 0.0;
    _contrastRate = 0.0;
    _selectionMode = false;
    _printShapeData = false;
    _availableSlots = SETBUTTONHANDLERS | SETSELECTHANDLERS | SETSELECTIONMODE | SETCOLORMODE | SETSLICENUM | ZOOM | MOVEH | MOVEV | ADJUSTBRIGHTNESS | ADJUSTCONTRAST | NEWPOINT | NEWSHAPE | DELETEPOINT | DELETESHAPE;
    ButtonHandlers bh[] = { none_button, zoom, move_h, move_v, adjust_c, adjust_b };
    setButtonHandlers( bh );
    SelectHandlers ch[] = { new_point, delete_point, new_shape, delete_shape };
    setSelectHandlers( ch );
}

m4dSliceViewerWidget::AvailableSlots
m4dSliceViewerWidget::getAvailableSlots()
{
    return _availableSlots;
}

void
m4dSliceViewerWidget::setButtonHandlers( ButtonHandlers* hnd )
{
    unsigned i;
    for ( i = 0; i < 6; i++ )
    {
        switch (hnd[i])
        {
            case none_button:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::none;
	    break;

	    case zoom:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::zoomImage;
	    break;

	    case move_h:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::moveImageH;
	    break;

	    case move_v:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::moveImageV;
	    break;

	    case adjust_c:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::adjustContrast;
	    break;

	    case adjust_b:
	    _buttonMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::adjustBrightness;
	    break;
	}
    }
}

void
m4dSliceViewerWidget::setSelectHandlers( SelectHandlers* hnd )
{
    unsigned i;
    for ( i = 0; i < 4; i++ )
    {
        switch (hnd[i])
        {
            case none_select:
	    _selectMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::nonePos;
	    break;

	    case new_point:
	    _selectMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::newPoint;
	    break;

	    case new_shape:
	    _selectMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::newShape;
	    break;

	    case delete_point:
	    _selectMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::deletePoint;
	    break;

	    case delete_shape:
	    _selectMethods[i/2][i%2] = &M4D::Viewer::m4dSliceViewerWidget::deleteShape;
	    break;
	}
    }
}



void
m4dSliceViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);
    glLoadIdentity();
    unsigned i, offsetx = _offset.x()<0?-_offset.x():0, offsety = _offset.y()<0?-_offset.y():0;
    glRasterPos2i( _offset.x()>0?_offset.x():0, _offset.y()>0?_offset.y():0 );
    glPixelStorei( GL_UNPACK_SKIP_PIXELS, (GLint)( offsetx / _zoomRate ) );
    glPixelStorei( GL_UNPACK_SKIP_ROWS, (GLint)( offsety / _zoomRate ) );
    size_t height, width, depth;
    double maxvalue;
    int stride;
    GLfloat avg = 0.0;
    switch ( _colorMode )
    {
        case rgba_unsigned_byte:
	{
	    maxvalue = 255.;
	    const uint8* pixel = (const uint8*)Imaging::Image< uint32, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	    pixel += ( _sliceNum - _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ) * height * width * 4;
    	    glPixelStorei( GL_UNPACK_ROW_LENGTH, width );
	    uint8* black = new uint8[ 4 * height * width ];
	    memset( black, 0, height * width * sizeof(uint32) );
	    uint8* avgLum = new uint8[ height * width * 4 ];
	    memset( avgLum, 0, height * width * sizeof(uint32) );
	    for ( i = 0; i < height * width; ++i )
	        avg += ( (float)pixel[4*i]*RW + (float)pixel[4*i + 1]*GW + (float)pixel[4*i + 2]*BW ) / maxvalue;
            avg = avg * maxvalue / (float)( width * height );
	    for ( i = 0; i < height * width; ++i )
	        avgLum[4*i] = avgLum[4*i + 1] = avgLum[4*i + 2] = (uint8)avg;
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_RGBA, GL_UNSIGNED_BYTE, avgLum );
	    glAccum( GL_ACCUM, _contrastRate );
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_RGBA, GL_UNSIGNED_BYTE, black );
	    glAccum( GL_ACCUM, _brightnessRate );
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_RGBA, GL_UNSIGNED_BYTE, pixel );
	    glAccum( GL_ACCUM, 1.0 - ( _brightnessRate + _contrastRate ) );
	    glAccum( GL_RETURN, 1.0 );
	    delete[] black;
	    delete[] avgLum;
	}
	break;

        case grayscale_unsigned_byte:
	{    
	    maxvalue = 255.;
	    const uint8* pixel = Imaging::Image< uint8, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	    pixel += ( _sliceNum - _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ) * height * width;
    	    glPixelStorei( GL_UNPACK_ROW_LENGTH, width );
	    uint8* black = new uint8[ height * width ];
	    memset( black, 0, height * width * sizeof(uint8) );
	    uint8* avgLum = new uint8[ height * width ];
	    memset( avgLum, 0, height * width * sizeof(uint8) );
	    for ( i = 0; i < height * width; ++i )
	        avg += (float)pixel[i] / maxvalue;
            avg = avg * maxvalue / (float)( width * height );
	    for ( i = 0; i < height * width; ++i )
	        avgLum[i] = (uint8)avg;
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_LUMINANCE, GL_UNSIGNED_BYTE, avgLum );
	    glAccum( GL_ACCUM, _contrastRate );
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_LUMINANCE, GL_UNSIGNED_BYTE, black );
	    glAccum( GL_ACCUM, _brightnessRate );
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel );
	    glAccum( GL_ACCUM, 1.0 - ( _brightnessRate + _contrastRate ) );
	    glAccum( GL_RETURN, 1.0 );
	    delete[] black;
	    delete[] avgLum;
	}
	break;

        case grayscale_unsigned_short:
	{
	    maxvalue = 65535.;
	    const uint16* pixel = Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort.GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	    pixel += ( _sliceNum - _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ) * height * width;
    	    glPixelStorei( GL_UNPACK_ROW_LENGTH, width );
	    uint16* black = new uint16[ height * width ];
	    memset( black, 0, height * width * sizeof(uint16) );
	    uint16* avgLum = new uint16[ height * width ];
	    memset( avgLum, 0, height * width * sizeof(uint16) );
	    for ( i = 0; i < height * width; ++i )
	        avg += (float)pixel[i] / maxvalue;
            avg = avg * maxvalue / (float)( width * height );
	    for ( i = 0; i < height * width; ++i )
	        avgLum[i] = (uint16)avg;
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_LUMINANCE, GL_UNSIGNED_SHORT, avgLum );
	    glAccum( GL_ACCUM, _contrastRate );
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_LUMINANCE, GL_UNSIGNED_SHORT, black );
	    glAccum( GL_ACCUM, _brightnessRate );
	    glDrawPixels( width - (GLint)( offsetx / _zoomRate ), height - (GLint)( offsety / _zoomRate ), GL_LUMINANCE, GL_UNSIGNED_SHORT, pixel );
	    glAccum( GL_ACCUM, 1.0 - ( _brightnessRate + _contrastRate ) );
	    glAccum( GL_RETURN, 1.0 );
	    delete[] black;
	    delete[] avgLum;
	}
	break;

    }
    glPixelZoom( _zoomRate, _zoomRate );
    glTranslatef( _offset.x(), _offset.y(), 0 );
    glScalef( _zoomRate, _zoomRate, 0. );
    if ( _selectionMode ) drawBorder();
    for ( std::list< Selection::m4dShape<int> >::iterator it = _shapes.begin(); it != --(_shapes.end()); ++it )
        drawShape( *it, false );
    if ( !_shapes.empty() ) drawShape( *(--(_shapes.end())), true );
    glFlush();
}

void
m4dSliceViewerWidget::drawBorder()
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f(1., 0., 0.);
    glBegin(GL_LINES);
        glVertex2i( 0, 0 );
	glVertex2i( 0, this->height() - 1 );
	glVertex2i( 0, this->height() - 1);
	glVertex2i( this->width() - 1, this->height() - 1 );
	glVertex2i( this->width() - 1, this->height() - 1 );
	glVertex2i( this->width() - 1, 0 );
	glVertex2i( this->width() - 1, 0 );
	glVertex2i( 0, 0 );
        glVertex2i( 1, 1 );
	glVertex2i( 1, this->height() - 2 );
	glVertex2i( 1, this->height() - 2 );
	glVertex2i( this->width() - 2, this->height() - 2 );
	glVertex2i( this->width() - 2, this->height() - 2 );
	glVertex2i( this->width() - 2, 1 );
	glVertex2i( this->width() - 2, 1 );
	glVertex2i( 1, 1 );
    glEnd();
    glPopMatrix();
}

void
m4dSliceViewerWidget::drawShape( Selection::m4dShape<int>& s, bool last )
{
    if ( last ) glColor3f( 1., 0., 0. );
    else glColor3f( 0., 0., 1. );
    if ( s.shapeClosed() && s.shapeElements().size() > 1 &&
	  s.shapeElements().back().getParticularValue( 2 ) == _sliceNum )
    {
        glBegin(GL_LINES);
	    glVertex2i( s.shapeElements().front().getParticularValue( 0 ), s.shapeElements().front().getParticularValue( 1 ) );
	    glVertex2i(  s.shapeElements().back().getParticularValue( 0 ),  s.shapeElements().back().getParticularValue( 1 ) );
	glEnd();
        if ( _printShapeData )
	{
	    if ( last ) glColor3f( 1., 1., 0. );
            else glColor3f( 0., 1., 1. );
	    Selection::m4dPoint< int > mid = Selection::m4dPoint< int >::midpoint( s.shapeElements().front(), s.shapeElements().back() );
	    char dist[20];
	    snprintf( dist, 19, "%f", Selection::m4dPoint< int >::distance( s.shapeElements().front(), s.shapeElements().back() ) );
	    dist[19] = 0;
	    glRasterPos2i( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
	    unsigned i;
	    for ( i = 0; i < strlen(dist); ++i ) glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, dist[i] );
            if ( last ) glColor3f( 1., 0., 0. );
            else glColor3f( 0., 0., 1. );
	}
    }
    if ( _printShapeData )
    {
        Selection::m4dPoint< int > c = s.getCentroid();
	float a = s.getArea();
	if ( a > 0 && _sliceNum == c.getParticularValue( 2 ) )
	{
	    if ( last ) glColor3f( 1., 0.5, 0. );
	    else glColor3f( 0., 0.5, 1. );
            glBegin(GL_QUADS);
	        glVertex2i( c.getParticularValue( 0 ) - 3, c.getParticularValue( 1 ) - 3 );
	        glVertex2i( c.getParticularValue( 0 ) + 3, c.getParticularValue( 1 ) - 3 );
	        glVertex2i( c.getParticularValue( 0 ) + 3, c.getParticularValue( 1 ) + 3 );
	        glVertex2i( c.getParticularValue( 0 ) - 3, c.getParticularValue( 1 ) + 3 );
	    glEnd();
	    char area[20];
	    snprintf( area, 19, "%f", a );
	    area[19] = 0;
	    glRasterPos2i( c.getParticularValue( 0 ) - 5, c.getParticularValue( 1 ) + 5 );
	    unsigned i;
	    for ( i = 0; i < strlen(area); ++i ) glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, area[i] );
	    if ( last ) glColor3f( 1., 0., 0. );
	    else glColor3f( 0., 0., 1. );
	}
    }
    for ( std::list< Selection::m4dPoint<int> >::iterator it = s.shapeElements().begin(); it != s.shapeElements().end(); ++it )
        if  ( it->getParticularValue( 2 ) == _sliceNum )
        {
	    if ( &(*it) != &(s.shapeElements().back()) )
	    {
	        std::list< Selection::m4dPoint<int> >::iterator tmp = it;
		++tmp;
	        glBegin(GL_LINES);
		    glVertex2i(  it->getParticularValue( 0 ),  it->getParticularValue( 1 ) );
		    glVertex2i( tmp->getParticularValue( 0 ), tmp->getParticularValue( 1 ) );
		glEnd();
                if ( _printShapeData )
	        {
	            if ( last ) glColor3f( 1., 1., 0. );
                    else glColor3f( 0., 1., 1. );
	            Selection::m4dPoint< int > mid = Selection::m4dPoint< int >::midpoint( *it, *tmp );
	            char dist[20];
	            snprintf( dist, 19, "%f", Selection::m4dPoint< int >::distance( *it, *tmp ) );
	            dist[19] = 0;
	            glRasterPos2i( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
	            unsigned i;
	            for ( i = 0; i < strlen(dist); ++i ) glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, dist[i] );
                    if ( last ) glColor3f( 1., 0., 0. );
                    else glColor3f( 0., 0., 1. );
	        }
	    }
	    if ( last && &(*it) == &(s.shapeElements().back()) ) glColor3f( 1., 0., 1. );
            glBegin(GL_QUADS);
	        glVertex2i( it->getParticularValue( 0 ) - 3, it->getParticularValue( 1 ) - 3 );
	        glVertex2i( it->getParticularValue( 0 ) + 3, it->getParticularValue( 1 ) - 3 );
	        glVertex2i( it->getParticularValue( 0 ) + 3, it->getParticularValue( 1 ) + 3 );
	        glVertex2i( it->getParticularValue( 0 ) - 3, it->getParticularValue( 1 ) + 3 );
	    glEnd();
        }
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
    if ( _selectionMode )
    {
        if ( event->modifiers() & Qt::ControlModifier )
	{
	    if ( event->buttons() & Qt::LeftButton )
	        (this->*_selectMethods[1][0])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
	    else if ( event->buttons() & Qt::RightButton )
	        (this->*_selectMethods[1][1])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
	}
	else
	{
	    if ( event->buttons() & Qt::LeftButton )
	        (this->*_selectMethods[0][0])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
	    else if ( event->buttons() & Qt::RightButton )
	        (this->*_selectMethods[0][1])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
	}
    }

    updateGL();
}

void
m4dSliceViewerWidget::mouseReleaseEvent(QMouseEvent *event)
{
    _lastPos = QPoint( -1, -1 );
}

void
m4dSliceViewerWidget::mouseMoveEvent(QMouseEvent *event)
{

    if ( _selectionMode || _lastPos.x() == -1 || _lastPos.y() == -1 ) return;
    

    int dx = event->x() - _lastPos.x();
    int dy = event->y() - _lastPos.y();

    if ( ( event->buttons() & Qt::LeftButton ) &&
         ( event->buttons() & Qt::RightButton ) )
    {
	(this->*_buttonMethods[0][0])( dx);
	(this->*_buttonMethods[0][1])(-dy);
    }
    else if (event->buttons() & Qt::LeftButton)
    {
	(this->*_buttonMethods[1][0])( dx);
	(this->*_buttonMethods[1][1])(-dy);
    }
    else if (event->buttons() & Qt::RightButton)
    {
	(this->*_buttonMethods[2][0])( dx);
	(this->*_buttonMethods[2][1])(-dy);
    }
    _lastPos = event->pos();

    updateGL();

}

void
m4dSliceViewerWidget::zoomImage( int amount )
{
    _zoomRate += 0.01 * amount;
    /*size_t   width  = _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum,
    	     height = _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum;
    _offset.setX( _offset.x() - (int)( amount * (int)width  * 0.005 ) );
    _offset.setY( _offset.y() - (int)( amount * (int)height * 0.005 ) );*/
}

void
m4dSliceViewerWidget::wheelEvent(QWheelEvent *event)
{
    if ( event->buttons() & Qt::LeftButton )
    {
        _selectionMode = _selectionMode?false:true;
    }
    else if ( event->buttons() & Qt::RightButton )
    {
        _printShapeData = _printShapeData?false:true;
    }
    else
    {
        if ( _selectionMode ) _selectionMode = false;

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
    }
    updateGL();

}

void
m4dSliceViewerWidget::setSliceNum( size_t num )
{
    if ( num < _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ||
         num >= _inPort.GetAbstractImage().GetDimensionExtents(2).maximum )
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
    _offset.setY( _offset.y() + amount );
}

void
m4dSliceViewerWidget::adjustBrightness( int amount )
{
    _brightnessRate -= ((GLfloat)amount)/((GLfloat)height()/2.0);
}

void
m4dSliceViewerWidget::adjustContrast( int amount )
{
    _contrastRate -= ((GLfloat)amount)/((GLfloat)width()/2.0);
}

void
m4dSliceViewerWidget::none( int amount )
{
    /* auxiliary function to handle function pointers correctly. Does nothing. */
}

void
m4dSliceViewerWidget::nonePos( int x, int y, int z )
{
    /* auxiliary function to handle function pointers correctly. Does nothing. */
}

void
m4dSliceViewerWidget::newPoint( int x, int y, int z )
{
    if ( x < 0 || y < 0 ||
         x >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum ) ||
         y >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum ) )
	     return;

    if ( _shapes.empty() ) newShape( x, y, z );
    else
    {
        if ( !_shapes.back().shapeElements().empty() &&
	     abs( x - _shapes.back().shapeElements().front().getParticularValue( 0 ) ) < MINIMUM_SELECT_DISTANCE &&
             abs( y - _shapes.back().shapeElements().front().getParticularValue( 1 ) ) < MINIMUM_SELECT_DISTANCE &&
                  z == _shapes.back().shapeElements().front().getParticularValue( 2 ) ) _shapes.back().closeShape();
	else
	{
            Selection::m4dPoint<int> p( x, y, z );
            _shapes.back().addPoint( p );
	}
    }
}

void
m4dSliceViewerWidget::newShape( int x, int y, int z )
{
    if ( x < 0 || y < 0 ||
         x >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum ) ||
         y >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum ) )
	     return;

    Selection::m4dShape<int> s( 3 );
    _shapes.push_back( s );
    newPoint( x, y, z );
}

void
m4dSliceViewerWidget::deletePoint( int x, int y, int z )
{
    if ( _shapes.empty() ) return;
    
    if ( _shapes.back().shapeClosed() ) _shapes.back().openShape();
    else
    {
        _shapes.back().deleteLast();
        if ( _shapes.back().shapeElements().empty() ) deleteShape( x, y, z );
    }
}

void
m4dSliceViewerWidget::deleteShape( int x, int y, int z )
{
    if ( _shapes.empty() ) return;
    
    _shapes.pop_back();
}

void
m4dSliceViewerWidget::slotSetButtonHandlers( ButtonHandlers* hnd )
{
    setButtonHandlers( hnd );
}

void
m4dSliceViewerWidget::slotSetSelectHandlers( SelectHandlers* hnd )
{
    setSelectHandlers( hnd );
}

void
m4dSliceViewerWidget::slotSetSelectionMode( bool mode )
{
    setSelectionMode( mode );
}

void
m4dSliceViewerWidget::slotSetColorMode( ColorMode cm )
{
    setColorMode( cm );
}

void
m4dSliceViewerWidget::slotSetSliceNum( size_t num )
{
    setSliceNum( num );
}

void
m4dSliceViewerWidget::slotZoom( int amount )
{
    zoomImage( amount );
}

void
m4dSliceViewerWidget::slotMoveH( int amount )
{
    moveImageH( amount );
}

void
m4dSliceViewerWidget::slotMoveV( int amount )
{
    moveImageV( amount );
}

void
m4dSliceViewerWidget::slotAdjustBrightness( int amount )
{
    adjustBrightness( amount );
}

void
m4dSliceViewerWidget::slotAdjustContrast( int amount )
{
    adjustContrast( amount );
}

void
m4dSliceViewerWidget::slotNewPoint( int x, int y, int z )
{
    newPoint( x, y, z );
}

void
m4dSliceViewerWidget::slotNewShape( int x, int y, int z )
{
    newShape( x, y, z );
}

void
m4dSliceViewerWidget::slotDeletePoint()
{
    deletePoint( 0, 0, 0 );
}

void
m4dSliceViewerWidget::slotDeleteShape()
{
    deleteShape( 0, 0, 0 );
}

void
m4dSliceViewerWidget::slotRotateAxisX( int x )
{
}

void
m4dSliceViewerWidget::slotRotateAxisY( int y )
{
}

void
m4dSliceViewerWidget::slotRotateAxisZ( int z )
{
}

} /*namespace Viewer*/
} /*namespace M4D*/
