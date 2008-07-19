#include "GUI/m4dSliceViewerWidget.h"

#include <QtGui>
#include "GUI/ogl/fonts.h"
#include <sstream>

#define MINIMUM_SELECT_DISTANCE 5

#define FONT_WIDTH   8
#define FONT_HEIGHT 16

namespace M4D
{
namespace Viewer
{

m4dSliceViewerWidget::m4dSliceViewerWidget( unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    setInputPort( );
}

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn, unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    setInputPort( conn );
}

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn, unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    setInputPort( conn );
}

m4dSliceViewerWidget::m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn, unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
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
m4dSliceViewerWidget::setUnSelected()
{
    _selected = false;
    updateGL();
}

void
m4dSliceViewerWidget::setSelected()
{
    _selected = true;
    updateGL();
    emit signalSetSelected( _index, false );
}

bool
m4dSliceViewerWidget::getSelected()
{
    return _selected;
}

void
m4dSliceViewerWidget::setInputPort( )
{
    _inPort.UnPlug();
    setParameters();
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
    if ( _inPort.IsPlugged() ) _sliceNum = _inPort.GetAbstractImage().GetDimensionExtents(2).minimum;
    _offset = QPoint( 0, 0 );
    _lastPos = QPoint( -1, -1 );
    _zoomRate = 1.0;
    _brightnessRate = 0.0;
    _contrastRate = 0.0;
    setButtonHandler( moveI, left );
    setButtonHandler( switch_slice, right );
    _printShapeData = false;
    _printData = true;
    _selected = false;
    _oneSliceMode = true;
    _slicesPerRow = 1;
    _flipH = _flipV = 1;
    _availableSlots.clear();
    _availableSlots.push_back( SETBUTTONHANDLER );
    _availableSlots.push_back( SETSELECTED );
    _availableSlots.push_back( SETSLICENUM );
    _availableSlots.push_back( SETONESLICEMODE );
    _availableSlots.push_back( SETMORESLICEMODE );
    _availableSlots.push_back( TOGGLEFLIPVERTICAL );
    _availableSlots.push_back( TOGGLEFLIPHORIZONTAL );
    _availableSlots.push_back( ADDLEFTSIDEDATA );
    _availableSlots.push_back( ADDRIGHTSIDEDATA );
    _availableSlots.push_back( ERASELEFTSIDEDATA );
    _availableSlots.push_back( ERASERIGHTSIDEDATA );
    _availableSlots.push_back( CLEARLEFTSIDEDATA );
    _availableSlots.push_back( CLEARRIGHTSIDEDATA );
    _availableSlots.push_back( TOGGLEPRINTDATA );
    _availableSlots.push_back( ZOOM );
    _availableSlots.push_back( MOVE );
    _availableSlots.push_back( CONTRASTBRIGHTNESS );
    _availableSlots.push_back( NEWPOINT );
    _availableSlots.push_back( NEWSHAPE );
    _availableSlots.push_back( DELETEPOINT );
    _availableSlots.push_back( DELETESHAPE );
    _leftSideData.clear();
    _rightSideData.clear();
}

m4dSliceViewerWidget::AvailableSlots
m4dSliceViewerWidget::getAvailableSlots()
{
    return _availableSlots;
}

QWidget*
m4dSliceViewerWidget::operator()()
{
    return (QGLWidget*)this;
}

void
m4dSliceViewerWidget::setButtonHandler( ButtonHandler hnd, MouseButton btn )
{
    switch (hnd)
    {
	case zoomI:
	_buttonMethods[btn] = &M4D::Viewer::m4dSliceViewerWidget::zoomImage;
	_selectionMode[btn] = false;
	break;

	case moveI:
	_buttonMethods[btn] = &M4D::Viewer::m4dSliceViewerWidget::moveImage;
	_selectionMode[btn] = false;
	break;

	case adjust_bc:
	_buttonMethods[btn] = &M4D::Viewer::m4dSliceViewerWidget::adjustContrastBrightness;
	_selectionMode[btn] = false;
	break;

	case switch_slice:
	_buttonMethods[btn] = &M4D::Viewer::m4dSliceViewerWidget::switchSlice;
	_selectionMode[btn] = false;
	break;

	case new_point:
	_selectMethods[btn] = &M4D::Viewer::m4dSliceViewerWidget::newPoint;
	_selectionMode[btn] = true;
	break;

	case new_shape:
	_selectMethods[btn] = &M4D::Viewer::m4dSliceViewerWidget::newShape;
	_selectionMode[btn] = true;
	break;
    }

    emit signalSetButtonHandler( _index, hnd, btn );
}

void
m4dSliceViewerWidget::setOneSliceMode()
{
    _slicesPerRow = 1;
    _oneSliceMode = true;
    emit signalSetOneSliceMode( _index );
}

void
m4dSliceViewerWidget::setMoreSliceMode( unsigned slicesPerRow )
{
    _slicesPerRow = slicesPerRow;
    _oneSliceMode = false;
    emit signalSetMoreSliceMode( _index, slicesPerRow );
}

void
m4dSliceViewerWidget::toggleFlipHorizontal()
{
    _flipH *= -1;
    emit signalToggleFlipVertical();
    updateGL();
}

void
m4dSliceViewerWidget::toggleFlipVertical()
{
    _flipV *= -1;
    emit signalToggleFlipVertical();
    updateGL();
}

void
m4dSliceViewerWidget::addLeftSideData( std::string type, std::string data )
{
    _leftSideData[type] = data;
    emit signalAddLeftSideData( type, data );
    if ( _printData ) updateGL();
}

void
m4dSliceViewerWidget::addRightSideData( std::string type, std::string data )
{
    _rightSideData[type] = data;
    emit signalAddRightSideData( type, data );
    if ( _printData ) updateGL();
}

void
m4dSliceViewerWidget::eraseLeftSideData( std::string type )
{
    _leftSideData.erase( type );
    emit signalEraseLeftSideData( type );
    if ( _printData ) updateGL();
}

void
m4dSliceViewerWidget::eraseRightSideData( std::string type )
{
    _rightSideData.erase( type );
    emit signalEraseRightSideData( type );
    if ( _printData ) updateGL();
}

void
m4dSliceViewerWidget::clearLeftSideData()
{
    _leftSideData.clear();
    emit signalClearLeftSideData();
    if ( _printData ) updateGL();
}

void
m4dSliceViewerWidget::clearRightSideData()
{
    _rightSideData.clear();
    emit signalClearRightSideData();
    if ( _printData ) updateGL();
}

void
m4dSliceViewerWidget::togglePrintData()
{
    _printData = _printData?false:true;
    emit signalTogglePrintData();
    updateGL();
}

void
m4dSliceViewerWidget::paintGL()
{
    glClear( GL_COLOR_BUFFER_BIT | GL_ACCUM_BUFFER_BIT );
    if ( _inPort.IsPlugged() )
    {
        unsigned i;
        double   w = (double)_inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum,
                 h = (double)_inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum;
        if ( _oneSliceMode )
	{
	    QPoint offset;
	    offset.setX( (int)floor( (double)_offset.x() - ( _zoomRate - (double)width()/w ) * 0.5 * w ) );
	    offset.setY( (int)floor( (double)_offset.y() - ( _zoomRate - (double)height()/h ) * 0.5 * h ) );
	    drawSlice( _sliceNum, _zoomRate, offset );
	}
        else for ( i = 0; (int)( i / _slicesPerRow ) * (int)( ( width() - 1 ) / _slicesPerRow ) * ( h / w ) < ( height() - 1 ); ++i )
	         drawSlice( _sliceNum + i, (double)( width() - 1 )/( (double)w * (double)_slicesPerRow ),
        								      QPoint( (i % _slicesPerRow) * ( ( width() - 1 ) / _slicesPerRow ) , (int)( ( i / _slicesPerRow ) * ( ( width() - 1 ) / _slicesPerRow ) * ( h / w ) ) ) );
        if ( _selectionMode[ left ] || _selectionMode[ right ] ) drawSelectionModeBorder();
    }
    if ( _selected ) drawSelectedBorder();
    glFlush();
}

void
m4dSliceViewerWidget::drawSlice( int sliceNum, double zoomRate, QPoint offset )
{
    if ( !_inPort.IsPlugged() ) return;
    if ( sliceNum < (int)_inPort.GetAbstractImage().GetDimensionExtents(2).minimum ||
         sliceNum >= (int)_inPort.GetAbstractImage().GetDimensionExtents(2).maximum )
        return;
    glClear( GL_ACCUM_BUFFER_BIT );
    glLoadIdentity();
    int   w = (int)_inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum,
          h = (int)_inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum;
    if ( _flipH < 0 ) offset.setX( offset.x() + (int)( zoomRate * w ) );
    if ( _flipV < 0 ) offset.setY( offset.y() + (int)( zoomRate * h ) );
    unsigned i, offsetx = offset.x()<0?-offset.x():0, offsety = offset.y()<0?-offset.y():0;
    if ( _flipH < 0 ) offsetx = offset.x()>(width() - 1)?offset.x()-width()+1:0;
    if ( _flipV < 0 ) offsety = offset.y()>(height() - 1)?offset.y()-height()+1:0;
    unsigned o_x = offset.x()>0?offset.x():0, o_y = offset.y()>0?offset.y():0;
    if ( _flipH < 0 ) o_x = offset.x()<(width() - 1)?offset.x():(width()-1);
    if ( _flipV < 0 ) o_y = offset.y()<(height() - 1)?offset.y():(height()-1);
    glRasterPos2i( o_x, o_y );
    if ( _oneSliceMode )
    {
        glPixelStorei( GL_UNPACK_SKIP_PIXELS, (GLint)( offsetx / zoomRate ) );
        glPixelStorei( GL_UNPACK_SKIP_ROWS, (GLint)( offsety / zoomRate ) );
    }
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
	    pixel += ( sliceNum - _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ) * height * width * 4;
    	    if ( _oneSliceMode ) glPixelStorei( GL_UNPACK_ROW_LENGTH, width );
	    uint8* black = new uint8[ 4 * height * width ];
	    memset( black, 0, height * width * sizeof(uint32) );
	    uint8* avgLum = new uint8[ height * width * 4 ];
	    memset( avgLum, 0, height * width * sizeof(uint32) );
	    for ( i = 0; i < height * width; ++i )
	        avg += ( (float)pixel[4*i]*RW + (float)pixel[4*i + 1]*GW + (float)pixel[4*i + 2]*BW ) / maxvalue;
            avg = avg * maxvalue / (float)( width * height );
	    for ( i = 0; i < height * width; ++i )
	        avgLum[4*i] = avgLum[4*i + 1] = avgLum[4*i + 2] = (uint8)avg;
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_RGBA, GL_UNSIGNED_BYTE, avgLum );
	    glAccum( GL_ACCUM, _contrastRate );
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_RGBA, GL_UNSIGNED_BYTE, black );
	    glAccum( GL_ACCUM, _brightnessRate );
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_RGBA, GL_UNSIGNED_BYTE, pixel );
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
	    pixel += ( sliceNum - _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ) * height * width;
    	    if ( _oneSliceMode ) glPixelStorei( GL_UNPACK_ROW_LENGTH, width );
	    uint8* black = new uint8[ height * width ];
	    memset( black, 0, height * width * sizeof(uint8) );
	    uint8* avgLum = new uint8[ height * width ];
	    memset( avgLum, 0, height * width * sizeof(uint8) );
	    for ( i = 0; i < height * width; ++i )
	        avg += (float)pixel[i] / maxvalue;
            avg = avg * maxvalue / (float)( width * height );
	    for ( i = 0; i < height * width; ++i )
	        avgLum[i] = (uint8)avg;
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_LUMINANCE, GL_UNSIGNED_BYTE, avgLum );
	    glAccum( GL_ACCUM, _contrastRate );
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_LUMINANCE, GL_UNSIGNED_BYTE, black );
	    glAccum( GL_ACCUM, _brightnessRate );
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel );
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
	    pixel += ( sliceNum - _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ) * height * width;
    	    if ( _oneSliceMode ) glPixelStorei( GL_UNPACK_ROW_LENGTH, width );
	    uint16* black = new uint16[ height * width ];
	    memset( black, 0, height * width * sizeof(uint16) );
	    uint16* avgLum = new uint16[ height * width ];
	    memset( avgLum, 0, height * width * sizeof(uint16) );
	    for ( i = 0; i < height * width; ++i )
	        avg += (float)pixel[i] / maxvalue;
            avg = avg * maxvalue / (float)( width * height );
	    for ( i = 0; i < height * width; ++i )
	        avgLum[i] = (uint16)avg;
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_LUMINANCE, GL_UNSIGNED_SHORT, avgLum );
	    glAccum( GL_ACCUM, _contrastRate );
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_LUMINANCE, GL_UNSIGNED_SHORT, black );
	    glAccum( GL_ACCUM, _brightnessRate );
	    glDrawPixels( width - (GLint)( offsetx / zoomRate ), height - (GLint)( offsety / zoomRate ), GL_LUMINANCE, GL_UNSIGNED_SHORT, pixel );
	    glAccum( GL_ACCUM, 1.0 - ( _brightnessRate + _contrastRate ) );
	    glAccum( GL_RETURN, 1.0 );
	    delete[] black;
	    delete[] avgLum;
	}
	break;

    }
    glPixelZoom( _flipH * zoomRate, _flipV * zoomRate );
    glTranslatef( offset.x(), offset.y(), 0 );
    glScalef( _flipH * zoomRate, _flipV * zoomRate, 0. );
    for ( std::list< Selection::m4dShape<int> >::iterator it = _shapes.begin(); it != --(_shapes.end()); ++it )
        drawShape( *it, false, sliceNum, zoomRate );
    if ( !_shapes.empty() ) drawShape( *(--(_shapes.end())), true, sliceNum, zoomRate );
    if ( _flipH < 0 ) offset.setX( offset.x() - (int)( zoomRate * w ) );
    if ( _flipV < 0 ) offset.setY( offset.y() - (int)( zoomRate * h ) );
    if ( _printData ) drawData( zoomRate, offset );
    glFlush();
}

void
m4dSliceViewerWidget::drawSelectionModeBorder()
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f(1., 0., 0.);
    glBegin(GL_LINE_LOOP);
        glVertex2i( 0, 0 );
	glVertex2i( 0, this->height() - 1);
	glVertex2i( this->width() - 1, this->height() - 1 );
	glVertex2i( this->width() - 1, 0 );
    glEnd();
    glBegin(GL_LINE_LOOP);
	glVertex2i( 1, 1 );
	glVertex2i( 1, this->height() - 2 );
	glVertex2i( this->width() - 2, this->height() - 2 );
	glVertex2i( this->width() - 2, 1 );
    glEnd();
    glPopMatrix();
}

void
m4dSliceViewerWidget::drawSelectedBorder()
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f(0., 1., 0.);
    glBegin(GL_LINE_LOOP);
        glVertex2i( 3, 3 );
	glVertex2i( 3, this->height() - 4 );
	glVertex2i( this->width() - 4, this->height() - 4 );
	glVertex2i( this->width() - 4, 3 );
    glEnd();
    glPopMatrix();
}

void
m4dSliceViewerWidget::drawShape( Selection::m4dShape<int>& s, bool last, int sliceNum, float zoomRate )
{
    if ( last ) glColor3f( 1., 0., 0. );
    else glColor3f( 0., 0., 1. );
    if ( s.shapeClosed() && s.shapeElements().size() > 1 &&
	  s.shapeElements().back().getParticularValue( 2 ) == sliceNum )
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
	    std::ostringstream dist;
	    dist << Selection::m4dPoint< int >::distance( s.shapeElements().front(), s.shapeElements().back() );
	    setTextPosition( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
	    setTextCoords( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
            drawText( dist.str().c_str() );
	    unsetTextCoords();
	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
	    if ( last ) glColor3f( 1., 0., 0. );
            else glColor3f( 0., 0., 1. );
	}
    }
    if ( _printShapeData )
    {
        Selection::m4dPoint< int > c = s.getCentroid();
	float a = s.getArea();
	if ( a > 0 && sliceNum == c.getParticularValue( 2 ) )
	{
	    if ( last ) glColor3f( 1., 0.5, 0. );
	    else glColor3f( 0., 0.5, 1. );
            glBegin(GL_QUADS);
	        glVertex2i( c.getParticularValue( 0 ) - 3, c.getParticularValue( 1 ) - 3 );
	        glVertex2i( c.getParticularValue( 0 ) + 3, c.getParticularValue( 1 ) - 3 );
	        glVertex2i( c.getParticularValue( 0 ) + 3, c.getParticularValue( 1 ) + 3 );
	        glVertex2i( c.getParticularValue( 0 ) - 3, c.getParticularValue( 1 ) + 3 );
	    glEnd();
	    std::ostringstream area;
	    area << a;
	    setTextPosition( c.getParticularValue( 0 ) - 5, c.getParticularValue( 1 ) + 5 );
	    setTextCoords( c.getParticularValue( 0 ) - 5, c.getParticularValue( 1 ) + 5 );
	    drawText( area.str().c_str() );
	    unsetTextCoords();
	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
	    if ( last ) glColor3f( 1., 0., 0. );
	    else glColor3f( 0., 0., 1. );
	}
    }
    for ( std::list< Selection::m4dPoint<int> >::iterator it = s.shapeElements().begin(); it != s.shapeElements().end(); ++it )
        if  ( it->getParticularValue( 2 ) == sliceNum )
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
	            std::ostringstream dist;
	            dist << Selection::m4dPoint< int >::distance( *it, *tmp );
		    setTextPosition( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
		    setTextCoords( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
	            drawText( dist.str().c_str() );
		    unsetTextCoords();
	    	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
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
m4dSliceViewerWidget::drawData( double zoomRate, QPoint offset )
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f( 1., 1., 1. );
    std::map< std::string, std::string >::iterator it;
    int   w = (int)_inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum,
          h = (int)_inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum;
    int i, o_x, o_y, w_o;
    if ( _oneSliceMode )
    {
        i = height() - 5 - FONT_HEIGHT;
	o_x = 4;
	o_y = 5;
	w_o = width() - 8;
    }
    else
    {
        i =  offset.y() + (unsigned)( h * zoomRate ) - 1 - FONT_HEIGHT;
	o_x = offset.x();
	o_y = offset.y();
	w_o = (int)( w * zoomRate );
    }
    for ( it = _leftSideData.begin(); it != _leftSideData.end() && i >= o_y; ++it, i -= FONT_HEIGHT )
    {
        if ( ( (int)( it->first + " : " + it->second ).length() * FONT_WIDTH ) < w_o )
	{
            setTextPosition( o_x, i );
            setTextCoords( o_x + ( (int)( it->first + " : " + it->second ).length() * FONT_WIDTH ), i + FONT_HEIGHT );
            drawText( ( it->first + " : " + it->second ).c_str() );
            unsetTextCoords();
	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
        }
    }
    if ( _oneSliceMode )
    {
        i = height() - 5 - FONT_HEIGHT;
	w_o = width() - 8;
	o_y = 5;
    }
    else
    {
        i =  offset.y() + (unsigned)( h * zoomRate ) - 1 - FONT_HEIGHT;
	w_o = (int)( w * zoomRate );
	o_y = offset.y();
    }
    for ( it = _rightSideData.begin(); it != _rightSideData.end() && i >= o_y; ++it, i -= FONT_HEIGHT )
    {
        if ( ( (int)( it->first + " : " + it->second ).length() * FONT_WIDTH ) < w_o )
	{
	    o_x = w_o - (int)( it->first + " : " + it->second ).length() * FONT_WIDTH;
	    if ( !_oneSliceMode ) o_x += offset.x();
            setTextPosition( o_x, i );
            setTextCoords( o_x + ( (int)( it->first + " : " + it->second ).length() * FONT_WIDTH ), i + FONT_HEIGHT );
            drawText( ( it->first + " : " + it->second ).c_str() );
            unsetTextCoords();
	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
        }
    }
    glPopMatrix();
}

void
m4dSliceViewerWidget::resizeGL(int winW, int winH)
{
    glViewport(0, 0, width(), height());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (double)winW, 0.0, (double)winH, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    if ( _inPort.IsPlugged() )
    {
        int   w = (int)_inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum,
    	      h = (int)_inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum;
        if ( (double)width() / (double)w < (double)height() / (double)h ) _zoomRate = (double)width() / (double)w;
	else _zoomRate = (double)height() / (double)h;
    }
    updateGL();
}

void
m4dSliceViewerWidget::mousePressEvent(QMouseEvent *event)
{
    if ( !_selected )
    {
        setSelected();
	return;
    }
    if ( !_inPort.IsPlugged() ) return;
    _lastPos = event->pos();
    int   w = (int)_inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum,
          h = (int)_inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum;
    if ( ( event->buttons() & Qt::LeftButton ) && _selectionMode[ left ] )
    {
        if ( _oneSliceMode ) (this->*_selectMethods[ left ])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
        else (this->*_selectMethods[ left ])( (int)( ( event->x() % ( ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
   					    (int)( ( ( this->height() - event->y() ) % ( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
					    _sliceNum + event->x() / (int)( ( width() - 1 ) / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) );
    }
    else if ( event->buttons() & Qt::RightButton && _selectionMode[ right ] )
    {
        if ( _oneSliceMode ) (this->*_selectMethods[ right ])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
        else (this->*_selectMethods[ right ])( (int)( ( event->x() % ( ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
   					     (int)( ( ( this->height() - event->y() ) % ( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
					     _sliceNum + event->x() / (int)( ( width() - 1 ) / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) );
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
    if ( !_inPort.IsPlugged() ) return;

    if ( _lastPos.x() == -1 || _lastPos.y() == -1 ) return;
    

    int dx = event->x() - _lastPos.x();
    int dy = event->y() - _lastPos.y();

    if ( ( event->buttons() & Qt::LeftButton ) && !_selectionMode[ left ] )
    {
        (this->*_buttonMethods[ left ])( dx, -dy );
    }
    else if ( ( event->buttons() & Qt::RightButton ) && !_selectionMode[ right ] )
    {
        (this->*_buttonMethods[ right ])( dx, -dy );
    }
    _lastPos = event->pos();

    updateGL();

}

void
m4dSliceViewerWidget::zoomImage( int dummy, int amount )
{
    _zoomRate += 0.001 * amount;
    if ( _zoomRate < 0. ) _zoomRate = 0.;
    emit signalZoom( _index, amount );
}

void
m4dSliceViewerWidget::wheelEvent(QWheelEvent *event)
{
    if ( !_inPort.IsPlugged() ) return;

    int numDegrees = event->delta() / 8;
    int numSteps = numDegrees / 15;
    try
    {
        setSliceNum( _sliceNum + numSteps * _slicesPerRow );
    }
    catch (...)
    {
        //TODO handle
    }
    updateGL();

}

void
m4dSliceViewerWidget::switchSlice( int dummy, int amount )
{
    try
    {
        setSliceNum( _sliceNum + amount );
    }
    catch (...)
    {
        //TODO handle
    }
}

void
m4dSliceViewerWidget::setSliceNum( size_t num )
{
    if ( !_inPort.IsPlugged() ) return;
    if ( num < _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ||
         num >= _inPort.GetAbstractImage().GetDimensionExtents(2).maximum )
    {
    	throw ErrorHandling::ExceptionBase( "Index out of bounds." );
    }
    _sliceNum = num;
    emit signalSetSliceNum( _index, num );
}

void
m4dSliceViewerWidget::moveImage( int amountH, int amountV )
{
    _offset.setX( _offset.x() + amountH );
    _offset.setY( _offset.y() + amountV );
    emit signalMove( _index, amountH, amountV );
}

void
m4dSliceViewerWidget::adjustContrastBrightness( int amountC, int amountB )
{
    _brightnessRate -= ((GLfloat)amountB)/((GLfloat)height()/2.0);
    _contrastRate -= ((GLfloat)amountC)/((GLfloat)width()/2.0);
    emit signalAdjustContrastBrightness( _index, amountC, amountB );
}

void
m4dSliceViewerWidget::newPoint( int x, int y, int z )
{
    if ( !_inPort.IsPlugged() ) return;
    if ( _shapes.empty() ) newShape( x, y, z );
    else
    {
        if ( _flipH < 0 ) x = - ( x - (int)( _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum ));
        if ( _flipV < 0 ) y = - ( y - (int)( _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum ));
        if ( x < 0 || y < 0 ||
             x >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum ) ||
             y >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum ) )
	         return;

        if ( !_shapes.back().shapeElements().empty() &&
	     abs( x - _shapes.back().shapeElements().front().getParticularValue( 0 ) ) < MINIMUM_SELECT_DISTANCE &&
             abs( y - _shapes.back().shapeElements().front().getParticularValue( 1 ) ) < MINIMUM_SELECT_DISTANCE &&
                  z == _shapes.back().shapeElements().front().getParticularValue( 2 ) ) _shapes.back().closeShape();
	else
	{
            Selection::m4dPoint<int> p( x, y, z );
            _shapes.back().addPoint( p );
	}
        emit signalNewPoint( _index, x, y, z );
    }
}

void
m4dSliceViewerWidget::newShape( int x, int y, int z )
{
    if ( !_inPort.IsPlugged() ) return;
    if ( x < 0 || y < 0 ||
         x >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum ) ||
         y >= (int)( _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum ) )
	     return;

    Selection::m4dShape<int> s( 3 );
    _shapes.push_back( s );
    newPoint( x, y, z );
    if ( _flipH < 0 ) x = - ( x - (int)( _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum ));
    if ( _flipV < 0 ) y = - ( y - (int)( _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum ));
    emit signalNewShape( _index, x, y, z );
}

void
m4dSliceViewerWidget::deletePoint()
{
    if ( _shapes.empty() ) return;
    
    if ( _shapes.back().shapeClosed() ) _shapes.back().openShape();
    else
    {
        _shapes.back().deleteLast();
        if ( _shapes.back().shapeElements().empty() ) deleteShape();
    }
    emit signalDeletePoint( _index );
}

void
m4dSliceViewerWidget::deleteShape()
{
    if ( _shapes.empty() ) return;
    
    _shapes.pop_back();
    emit signalDeleteShape( _index );
}

void
m4dSliceViewerWidget::deleteAll()
{
    while ( !_shapes.empty() ) deleteShape();
    emit signalDeleteAll( _index );
}

void
m4dSliceViewerWidget::slotSetButtonHandler( ButtonHandler hnd, MouseButton btn )
{
    setButtonHandler( hnd, btn );
}

void
m4dSliceViewerWidget::slotToggleFlipHorizontal()
{
    toggleFlipHorizontal();
}

void
m4dSliceViewerWidget::slotToggleFlipVertical()
{
    toggleFlipVertical();
}

void
m4dSliceViewerWidget::slotAddLeftSideData( std::string type, std::string data )
{
    addLeftSideData( type, data );
}

void
m4dSliceViewerWidget::slotAddRightSideData( std::string type, std::string data )
{
    addRightSideData( type, data );
}

void
m4dSliceViewerWidget::slotEraseLeftSideData( std::string type )
{
    eraseLeftSideData( type );
}

void
m4dSliceViewerWidget::slotEraseRightSideData( std::string type )
{
    eraseRightSideData( type );
}

void
m4dSliceViewerWidget::slotClearLeftSideData()
{
    clearLeftSideData();
}

void
m4dSliceViewerWidget::slotClearRightSideData()
{
    clearRightSideData();
}

void
m4dSliceViewerWidget::slotTogglePrintData()
{
    togglePrintData();
}

void
m4dSliceViewerWidget::slotSetSliceNum( size_t num )
{
    setSliceNum( num );
}

void
m4dSliceViewerWidget::slotZoom( int amount )
{
    zoomImage( 0, amount );
}

void
m4dSliceViewerWidget::slotMove( int amountH, int amountV )
{
    moveImage( amountH, amountV );
}

void
m4dSliceViewerWidget::slotAdjustContrastBrightness( int amountC, int amountB )
{
    adjustContrastBrightness( amountC, amountB );
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
    deletePoint();
}

void
m4dSliceViewerWidget::slotDeleteShape()
{
    deleteShape();
}

void
m4dSliceViewerWidget::slotDeleteAll()
{
    deleteAll();
}

void
m4dSliceViewerWidget::slotSetSelected( bool selected )
{
    if ( selected ) setSelected();
    else setUnSelected();
}

void
m4dSliceViewerWidget::slotSetOneSliceMode()
{
    setOneSliceMode();
}

void
m4dSliceViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow )
{
    setMoreSliceMode( slicesPerRow );
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
