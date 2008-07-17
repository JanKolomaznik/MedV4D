#include "GUI/m4dSliceViewerWidget.h"

#include <QtGui>
#include "GUI/ogl/fonts.h"
#include <string.h>

#define MINIMUM_SELECT_DISTANCE 5

namespace M4D
{
namespace Viewer
{

m4dSliceViewerWidget::m4dSliceViewerWidget( QWidget *parent)
    : QGLWidget(parent)
{
    setInputPort( );
}

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
    emit signalSetColorMode( cm );
}

void
m4dSliceViewerWidget::setSelectionMode( bool mode )
{
    _selectionMode = mode;
    emit signalSetSelectionMode( mode );
}

bool
m4dSliceViewerWidget::getSelectionMode()
{
    return _selectionMode;
}

void
m4dSliceViewerWidget::setSelected( bool selected )
{
    _selected = selected;
    emit signalSetSelected( selected );
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
    _selectionMode = false;
    _printShapeData = false;
    _oneSliceMode = true;
    _selected = false;
    _slicesPerRow = 1;
    _availableSlots = SETBUTTONHANDLERS | SETSELECTHANDLERS | SETSELECTIONMODE | SETCOLORMODE | SETSLICENUM | ZOOM | MOVEH | MOVEV | ADJUSTBRIGHTNESS | ADJUSTCONTRAST | NEWPOINT | NEWSHAPE | DELETEPOINT | DELETESHAPE | SETSELECTED | SETONESLICEMODE | SETMORESLICEMODE;
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

QWidget*
m4dSliceViewerWidget::operator()()
{
    return (QWidget*)((QGLWidget*)this);
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
    emit signalSetButtonHandlers( hnd );
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
    emit signalSetSelectHandlers( hnd );
}

void
m4dSliceViewerWidget::setOneSliceMode()
{
    _slicesPerRow = 1;
    _oneSliceMode = true;
    emit signalSetOneSliceMode();
}

void
m4dSliceViewerWidget::setMoreSliceMode( unsigned slicesPerRow )
{
    _slicesPerRow = slicesPerRow;
    _oneSliceMode = false;
    emit signalSetMoreSliceMode( slicesPerRow );
}

void
m4dSliceViewerWidget::paintGL()
{
    glClear( GL_COLOR_BUFFER_BIT | GL_ACCUM_BUFFER_BIT );
    if ( _inPort.IsPlugged() )
    {
        unsigned i;
        if ( _oneSliceMode ) drawSlice( _sliceNum, _zoomRate, _offset );
        else for ( i = 0; i < _slicesPerRow * _slicesPerRow; ++i ) drawSlice( _sliceNum + i, 1./(double)_slicesPerRow,
        								      QPoint( (i % _slicesPerRow) * ( height() / _slicesPerRow ) , ( i / _slicesPerRow ) * ( width() / _slicesPerRow ) ) );
        if ( _selectionMode ) drawSelectionModeBorder();
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
    unsigned i, offsetx = offset.x()<0?-offset.x():0, offsety = offset.y()<0?-offset.y():0;
    glRasterPos2i( offset.x()>0?offset.x():0, offset.y()>0?offset.y():0 );
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
    glPixelZoom( zoomRate, zoomRate );
    glTranslatef( offset.x(), offset.y(), 0 );
    glScalef( zoomRate, zoomRate, 0. );
    for ( std::list< Selection::m4dShape<int> >::iterator it = _shapes.begin(); it != --(_shapes.end()); ++it )
        drawShape( *it, false, sliceNum, zoomRate );
    if ( !_shapes.empty() ) drawShape( *(--(_shapes.end())), true, sliceNum, zoomRate );
}

void
m4dSliceViewerWidget::drawSelectionModeBorder()
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
m4dSliceViewerWidget::drawSelectedBorder()
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f(0., 1., 0.);
    glBegin(GL_LINES);
        glVertex2i( 3, 3 );
	glVertex2i( 3, this->height() - 4 );
	glVertex2i( 3, this->height() - 4 );
	glVertex2i( this->width() - 4, this->height() - 4 );
	glVertex2i( this->width() - 4, this->height() - 4 );
	glVertex2i( this->width() - 4, 3 );
	glVertex2i( this->width() - 4, 3 );
	glVertex2i( 3, 3 );
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
	    char dist[20];
	    snprintf( dist, 19, "%f", Selection::m4dPoint< int >::distance( s.shapeElements().front(), s.shapeElements().back() ) );
	    dist[19] = 0;
	    setTextPosition( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
	    setTextCoords( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
            drawText( dist );
	    unsetTextCoords();
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
	    char area[20];
	    snprintf( area, 19, "%f", a );
	    area[19] = 0;
	    setTextPosition( c.getParticularValue( 0 ) - 5, c.getParticularValue( 1 ) + 5 );
	    setTextCoords( c.getParticularValue( 0 ) - 5, c.getParticularValue( 1 ) + 5 );
	    drawText( area );
	    unsetTextCoords();
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
	            char dist[20];
	            snprintf( dist, 19, "%f", Selection::m4dPoint< int >::distance( *it, *tmp ) );
	            dist[19] = 0;
		    setTextPosition( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
		    setTextCoords( mid.getParticularValue( 0 ), mid.getParticularValue( 1 ) );
	            drawText( dist );
		    unsetTextCoords();
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
    if ( !_inPort.IsPlugged() ) return;
    _lastPos = event->pos();
    if ( _selectionMode )
    {
        if ( event->modifiers() & Qt::ControlModifier )
	{
	    if ( event->buttons() & Qt::LeftButton )
	    {
	        if ( _oneSliceMode ) (this->*_selectMethods[1][0])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
		else (this->*_selectMethods[1][0])( (int)( ( event->x() % ( width() / _slicesPerRow ) ) * _slicesPerRow ),
						    (int)( ( ( this->height() - event->y() ) % ( height() / _slicesPerRow ) ) * _slicesPerRow ),
						    _sliceNum + event->x() / (int)( width() / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( height() / _slicesPerRow ) ) );
	    }
	    else if ( event->buttons() & Qt::RightButton )
	    {
	        if ( _oneSliceMode ) (this->*_selectMethods[1][1])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
		else (this->*_selectMethods[1][1])( (int)( ( event->x() % ( width() / _slicesPerRow ) ) * _slicesPerRow ),
						    (int)( ( ( this->height() - event->y() ) % ( height() / _slicesPerRow ) ) * _slicesPerRow ),
						    _sliceNum + event->x() / (int)( width() / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( height() / _slicesPerRow ) ) );
	    }
	}
	else
	{
	    if ( event->buttons() & Qt::LeftButton )
	    {
	        if ( _oneSliceMode ) (this->*_selectMethods[0][0])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
		else (this->*_selectMethods[0][0])( (int)( ( event->x() % ( width() / _slicesPerRow ) ) * _slicesPerRow ),
						    (int)( ( ( this->height() - event->y() ) % ( height() / _slicesPerRow ) ) * _slicesPerRow ),
						    _sliceNum + event->x() / (int)( width() / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( height() / _slicesPerRow ) ) );
	    }
	    else if ( event->buttons() & Qt::RightButton )
	    {
	        if ( _oneSliceMode ) (this->*_selectMethods[0][1])( (int)( ( event->x() - _offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - _offset.y() ) / _zoomRate ), _sliceNum );
		else (this->*_selectMethods[0][1])( (int)( ( event->x() % ( width() / _slicesPerRow ) ) * _slicesPerRow ),
						    (int)( ( ( this->height() - event->y() ) % ( height() / _slicesPerRow ) ) * _slicesPerRow ),
						    _sliceNum + event->x() / (int)( width() / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( height() / _slicesPerRow ) ) );
	    }
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
    if ( !_inPort.IsPlugged() ) return;

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
    _zoomRate += 0.001 * amount;
    if ( _zoomRate < 0. ) _zoomRate = 0.;
    /*size_t   width  = _inPort.GetAbstractImage().GetDimensionExtents(0).maximum - _inPort.GetAbstractImage().GetDimensionExtents(0).minimum,
    	     height = _inPort.GetAbstractImage().GetDimensionExtents(1).maximum - _inPort.GetAbstractImage().GetDimensionExtents(1).minimum;
    _offset.setX( _offset.x() - (int)( amount * (int)width  * 0.005 ) );
    _offset.setY( _offset.y() - (int)( amount * (int)height * 0.005 ) );*/
    emit signalZoom( amount );
}

void
m4dSliceViewerWidget::wheelEvent(QWheelEvent *event)
{
    if ( !_inPort.IsPlugged() ) return;
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
            setSliceNum( _sliceNum + numSteps * _slicesPerRow );
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
    if ( !_inPort.IsPlugged() ) return;
    if ( num < _inPort.GetAbstractImage().GetDimensionExtents(2).minimum ||
         num >= _inPort.GetAbstractImage().GetDimensionExtents(2).maximum )
    {
    	throw ErrorHandling::ExceptionBase( "Index out of bounds." );
    }
    _sliceNum = num;
    emit signalSetSliceNum( num );
}

void
m4dSliceViewerWidget::moveImageH( int amount )
{
    _offset.setX( _offset.x() + amount );
    emit signalMoveH( amount );
}

void
m4dSliceViewerWidget::moveImageV( int amount )
{
    _offset.setY( _offset.y() + amount );
    emit signalMoveV( amount );
}

void
m4dSliceViewerWidget::adjustBrightness( int amount )
{
    _brightnessRate -= ((GLfloat)amount)/((GLfloat)height()/2.0);
    emit signalAdjustBrightness( amount );
}

void
m4dSliceViewerWidget::adjustContrast( int amount )
{
    _contrastRate -= ((GLfloat)amount)/((GLfloat)width()/2.0);
    emit signalAdjustContrast( amount );
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
    if ( !_inPort.IsPlugged() ) return;
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
    emit signalNewPoint( x, y, z );
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
    emit signalNewShape( x, y, z );
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
    emit signalDeletePoint();
}

void
m4dSliceViewerWidget::deleteShape( int x, int y, int z )
{
    if ( _shapes.empty() ) return;
    
    _shapes.pop_back();
    emit signalDeleteShape();
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
m4dSliceViewerWidget::slotSetSelected( bool selected )
{
    setSelected( selected );
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
