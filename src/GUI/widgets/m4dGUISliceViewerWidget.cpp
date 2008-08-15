#include "GUI/m4dGUISliceViewerWidget.h"

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

m4dGUISliceViewerWidget::m4dGUISliceViewerWidget( unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    _inPort = new Imaging::InputPortAbstractImage();
    _inputPorts.AddPort( _inPort );
    setInputPort( );
}

m4dGUISliceViewerWidget::m4dGUISliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    _inPort = new Imaging::InputPortAbstractImage();
    _inputPorts.AddPort( _inPort );
    setInputPort( conn );
}

void
m4dGUISliceViewerWidget::setUnSelected()
{
    _selected = false;
    updateGL();
}

void
m4dGUISliceViewerWidget::setSelected()
{
    _selected = true;
    updateGL();
    emit signalSetSelected( _index, false );
}

void
m4dGUISliceViewerWidget::setInputPort( )
{
    _inPort->UnPlug();
    setParameters();
    updateGL();
}

void
m4dGUISliceViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
{
    conn->ConnectConsumer( *_inPort );
    setParameters();
    updateGL();
}

void
m4dGUISliceViewerWidget::setParameters()
{
    _ready = false;
    if ( _inPort->IsPlugged() )
    {
        try
	{
	    if ( _inPort->TryLockDataset() )
	    {
                try
	        {
        	    _imageID = _inPort->GetAbstractImage().GetElementTypeID();
		    _minimum[0] = _inPort->GetAbstractImage().GetDimensionExtents(0).minimum;
		    _minimum[1] = _inPort->GetAbstractImage().GetDimensionExtents(1).minimum;
		    _minimum[2] = _inPort->GetAbstractImage().GetDimensionExtents(2).minimum;
		    _maximum[0] = _inPort->GetAbstractImage().GetDimensionExtents(0).maximum;
		    _maximum[1] = _inPort->GetAbstractImage().GetDimensionExtents(1).maximum;
		    _maximum[2] = _inPort->GetAbstractImage().GetDimensionExtents(2).maximum;
	            _sliceNum = _inPort->GetAbstractImage().GetDimensionExtents(2).minimum;
		}
		catch (...) { _ready = false; }
	        _inPort->ReleaseDatasetLock();
	    }
	    else
	        return;
	}
	catch (...) { _ready = false; }
    }
    _offset = QPoint( 0, 0 );
    _lastPos = QPoint( -1, -1 );
    _zoomRate = 1.0;
    _brightnessRate = 0;
    _contrastRate = 1.0;
    slotSetButtonHandler( moveI, left );
    slotSetButtonHandler( switch_slice, right );
    _printShapeData = true;
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
    _availableSlots.push_back( MOVE );
    _availableSlots.push_back( CONTRASTBRIGHTNESS );
    _availableSlots.push_back( NEWPOINT );
    _availableSlots.push_back( NEWSHAPE );
    _availableSlots.push_back( DELETEPOINT );
    _availableSlots.push_back( DELETESHAPE );
    _availableSlots.push_back( DELETEALL );
    _availableSlots.push_back( ZOOM );
    _leftSideData.clear();
    _rightSideData.clear();
    _ready = true;
}

m4dGUISliceViewerWidget::AvailableSlots
m4dGUISliceViewerWidget::getAvailableSlots()
{
    return _availableSlots;
}

QWidget*
m4dGUISliceViewerWidget::operator()()
{
    return (QGLWidget*)this;
}

void
m4dGUISliceViewerWidget::ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction )
{
    switch( msg->msgID )
    {
        case Imaging::PMI_FILTER_UPDATED:
        case Imaging::PMI_PORT_PLUGGED:
	{
	    _ready = false;
	    if ( _inPort->IsPlugged() )
	    {
	        try
		{
		    if ( _inPort->TryLockDataset() )
	            {
		        try
			{
        		    _imageID = _inPort->GetAbstractImage().GetElementTypeID();
		    	    _minimum[0] = _inPort->GetAbstractImage().GetDimensionExtents(0).minimum;
		    	    _minimum[1] = _inPort->GetAbstractImage().GetDimensionExtents(1).minimum;
		    	    _minimum[2] = _inPort->GetAbstractImage().GetDimensionExtents(2).minimum;
		    	    _maximum[0] = _inPort->GetAbstractImage().GetDimensionExtents(0).maximum;
		    	    _maximum[1] = _inPort->GetAbstractImage().GetDimensionExtents(1).maximum;
		    	    _maximum[2] = _inPort->GetAbstractImage().GetDimensionExtents(2).maximum;
			    _sliceNum = _inPort->GetAbstractImage().GetDimensionExtents(2).minimum;
			}
			catch (...) { _ready = false; }
		        _inPort->ReleaseDatasetLock();
		        _ready = true;
	            }
		}
		catch (...) { _ready = false; }
	    }
	    updateGL();
	}
	break;
	
	default:
	break;
    }
}

void
m4dGUISliceViewerWidget::setButtonHandler( ButtonHandler hnd, MouseButton btn )
{
    switch (hnd)
    {
	case zoomI:
	_buttonMethods[btn] = &M4D::Viewer::m4dGUISliceViewerWidget::zoomImage;
	_selectionMode[btn] = false;
	break;

	case moveI:
	_buttonMethods[btn] = &M4D::Viewer::m4dGUISliceViewerWidget::moveImage;
	_selectionMode[btn] = false;
	break;

	case adjust_bc:
	_buttonMethods[btn] = &M4D::Viewer::m4dGUISliceViewerWidget::adjustContrastBrightness;
	_selectionMode[btn] = false;
	break;

	case switch_slice:
	_buttonMethods[btn] = &M4D::Viewer::m4dGUISliceViewerWidget::switchSlice;
	_selectionMode[btn] = false;
	break;

	case new_point:
	_selectMethods[btn] = &M4D::Viewer::m4dGUISliceViewerWidget::newPoint;
	_selectionMode[btn] = true;
	break;

	case new_shape:
	_selectMethods[btn] = &M4D::Viewer::m4dGUISliceViewerWidget::newShape;
	_selectionMode[btn] = true;
	break;
    }

    if ( _ready ) updateGL();
    emit signalSetButtonHandler( _index, hnd, btn );
}

void
m4dGUISliceViewerWidget::setOneSliceMode()
{
    _slicesPerRow = 1;
    _oneSliceMode = true;
    if ( _availableSlots.back() != ZOOM ) _availableSlots.push_back( ZOOM );
    updateGL();
    emit signalSetOneSliceMode( _index );
}

void
m4dGUISliceViewerWidget::setMoreSliceMode( unsigned slicesPerRow )
{
    _slicesPerRow = slicesPerRow;
    _oneSliceMode = false;
    if ( _availableSlots.back() == ZOOM ) _availableSlots.pop_back();
    updateGL();
    emit signalSetMoreSliceMode( _index, slicesPerRow );
}

void
m4dGUISliceViewerWidget::toggleFlipHorizontal()
{
    _flipH *= -1;
    emit signalToggleFlipVertical();
    updateGL();
}

void
m4dGUISliceViewerWidget::toggleFlipVertical()
{
    _flipV *= -1;
    emit signalToggleFlipVertical();
    updateGL();
}

void
m4dGUISliceViewerWidget::addLeftSideData( std::string type, std::string data )
{
    _leftSideData[type] = data;
    emit signalAddLeftSideData( type, data );
    if ( _printData ) updateGL();
}

void
m4dGUISliceViewerWidget::addRightSideData( std::string type, std::string data )
{
    _rightSideData[type] = data;
    emit signalAddRightSideData( type, data );
    if ( _printData ) updateGL();
}

void
m4dGUISliceViewerWidget::eraseLeftSideData( std::string type )
{
    _leftSideData.erase( type );
    emit signalEraseLeftSideData( type );
    if ( _printData ) updateGL();
}

void
m4dGUISliceViewerWidget::eraseRightSideData( std::string type )
{
    _rightSideData.erase( type );
    emit signalEraseRightSideData( type );
    if ( _printData ) updateGL();
}

void
m4dGUISliceViewerWidget::clearLeftSideData()
{
    _leftSideData.clear();
    emit signalClearLeftSideData();
    if ( _printData ) updateGL();
}

void
m4dGUISliceViewerWidget::clearRightSideData()
{
    _rightSideData.clear();
    emit signalClearRightSideData();
    if ( _printData ) updateGL();
}

void
m4dGUISliceViewerWidget::togglePrintData()
{
    _printData = _printData?false:true;
    emit signalTogglePrintData();
    updateGL();
}

void
m4dGUISliceViewerWidget::paintGL()
{
    glClear( GL_COLOR_BUFFER_BIT );
    if ( !_ready )
    {
        setParameters();
	if ( !_ready ) return;
    }
    if ( _inPort->IsPlugged() )
    {
        unsigned i;
	double w, h;
	w = (double)_maximum[0] - _minimum[0],
        h = (double)_maximum[1] - _minimum[1];
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
m4dGUISliceViewerWidget::drawSlice( int sliceNum, double zoomRate, QPoint offset )
{
    if ( !_ready ) return;
    if ( !_inPort->IsPlugged() ) return;
    int w, h;
    if ( sliceNum < (int)_minimum[2] ||
         sliceNum >= (int)_maximum[2] )
    {
        return;
    }
    else
    {
        w = (int)_maximum[0] - _minimum[0],
        h = (int)_maximum[1] - _minimum[1];
    }
    glLoadIdentity();
    if ( _flipH < 0 ) offset.setX( offset.x() + (int)( zoomRate * w ) );
    if ( _flipV < 0 ) offset.setY( offset.y() + (int)( zoomRate * h ) );
    uint32 height, width, depth;
    double maxvalue;
    int stride;
    GLuint texName;

    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &texName );

    glBindTexture ( GL_TEXTURE_2D, texName );
    glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

    glTranslatef( offset.x(), offset.y(), 0 );
    glScalef( _flipH * zoomRate, _flipV * zoomRate, 0. );
    switch ( _imageID )
    {
        case NTID_UINT_32:
	{
	    maxvalue = 255.;
	    uint8* pixel, *original;
	    try
	    {
	        if ( _inPort->TryLockDataset() )
	        {
		    try
		    {
		         original = (uint8*)Imaging::Image< uint32, 3 >::CastAbstractImage(_inPort->GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	                 original += ( sliceNum - _minimum[2] ) * height * width * 4;
		    }
		    catch (...) { _ready = false; }
		    _inPort->ReleaseDatasetLock();
	        }
	        else
	        {
	            _ready = false;
		    return;
	        }
	    }
	    catch (...) { _ready = false; }

	    pixel = new uint8[ height * width * 4 ];
	    unsigned i;
	    for ( i = 0; i < width * height * 4; i += 4 )
	    {
	        if ( ( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) > maxvalue ) pixel[i] = (uint8)maxvalue;
		else if ( ( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) < 0. ) pixel[i] = 0;
		else pixel[i] = (uint8)( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. );
	        if ( ( _contrastRate *   ( original[i+1]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) > maxvalue ) pixel[i+1] = (uint8)maxvalue;
		else if ( ( _contrastRate *   ( original[i+1]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) < 0. ) pixel[i+1] = 0;
		else pixel[i+1] = (uint8)( _contrastRate *   ( original[i+1]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. );
	        if ( ( _contrastRate *   ( original[i+2]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) > maxvalue ) pixel[i+2] = (uint8)maxvalue;
		else if ( ( _contrastRate *   ( original[i+2]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) < 0. ) pixel[i+2] = 0;
		else pixel[i+2] = (uint8)( _contrastRate *   ( original[i+2]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. );
		pixel[i+3] = original[i+3];
	    }
	    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
	                  GL_RGBA, GL_UNSIGNED_BYTE, pixel );
	    delete[] pixel;
	    
	}
	break;

        case NTID_INT_8:
	{    
	    maxvalue = 127.;
	    int8* pixel, *original;
	    try
	    {
	        if ( _inPort->TryLockDataset() )
	        {
		    try
		    {
		        original = Imaging::Image< int8, 3 >::CastAbstractImage(_inPort->GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	                original += ( sliceNum - _minimum[2] ) * height * width;
		    } catch (...) { _ready = false; }
		    _inPort->ReleaseDatasetLock();
	        }
	        else
	        {
	            _ready = false;
		    return;
	        }
	    }
	    catch (...) { _ready = false; }

	    pixel = new int8[ height * width ];
	    unsigned i;
	    for ( i = 0; i < width * height; ++i )
	    {
	        if ( ( _contrastRate *   ( original[i]+ _brightnessRate ) ) > maxvalue ) pixel[i] = (int8)maxvalue;
		else if ( ( _contrastRate *   ( original[i] + _brightnessRate ) ) < -maxvalue ) pixel[i] = (int8)(-maxvalue);
		else pixel[i] = (uint8)( _contrastRate *   ( original[i] + _brightnessRate ) );
	    }
	    glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
	                  GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel );
	    delete[] pixel;
	    
	}
	break;

        case NTID_UINT_8:
	{    
	    maxvalue = 255.;
	    uint8* pixel, *original;
	    try
	    {
	        if ( _inPort->TryLockDataset() )
	        {
		    try
		    {
		        original = Imaging::Image< uint8, 3 >::CastAbstractImage(_inPort->GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	                original += ( sliceNum - _minimum[2] ) * height * width;
		    } catch (...) { _ready = false; }
		    _inPort->ReleaseDatasetLock();
	        }
	        else
	        {
	            _ready = false;
		    return;
	        }
	    }
	    catch (...) { _ready = false; }

	    pixel = new uint8[ height * width ];
	    unsigned i;
	    for ( i = 0; i < width * height; ++i )
	    {
	        if ( ( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) > maxvalue ) pixel[i] = (uint8)maxvalue;
		else if ( ( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) < 0. ) pixel[i] = 0;
		else pixel[i] = (uint8)( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. );
	    }
	    glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
	                  GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel );
	    delete[] pixel;
	    
	}
	break;

        case NTID_UINT_16:
	{
	    maxvalue = 65535.;
	    uint16* pixel, *original;
	    try
	    {
	        if ( _inPort->TryLockDataset() )
	        {
		    try
		    {
		        original = Imaging::Image< uint16, 3 >::CastAbstractImage(_inPort->GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	                original += ( sliceNum - _minimum[2] ) * height * width;
		    } catch (...) { _ready = false; }
		    _inPort->ReleaseDatasetLock();
	        }
	        else
	        {
	            _ready = false;
		    return;
	        }
	    }
	    catch (...) { _ready = false; }

	    pixel = new uint16[ height * width ];
	    unsigned i;
	    for ( i = 0; i < width * height; ++i )
	    {
	        if ( ( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) > maxvalue ) pixel[i] = (uint16)maxvalue;
		else if ( ( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. ) < 0. ) pixel[i] = 0;
		else pixel[i] = (uint16)( _contrastRate *   ( original[i]   - maxvalue / 2. + _brightnessRate ) + maxvalue / 2. );
	    }
	    glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
	                  GL_LUMINANCE, GL_UNSIGNED_SHORT, pixel );
	    delete[] pixel;
	    
	}
	break;
        case NTID_INT_16:
	{
	    maxvalue = 32767.;
	    int16* pixel, *original;
	    try
	    {
	        if ( _inPort->TryLockDataset() )
	        {
		    try
		    {
		        original = Imaging::Image< int16, 3 >::CastAbstractImage(_inPort->GetAbstractImage()).GetPointer( height, width, depth, stride, stride, stride );
	                original += ( sliceNum - _minimum[2] ) * height * width;
		    } catch (...) { _ready = false; }
		    _inPort->ReleaseDatasetLock();
	        }
	        else
	        {
	            _ready = false;
		    return;
	        }
	    }
	    catch (...) { _ready = false; }

	    pixel = new int16[ height * width ];
	    unsigned i;
	    for ( i = 0; i < width * height; ++i )
	    {
	        if ( ( _contrastRate *   ( original[i] + _brightnessRate ) ) > maxvalue ) pixel[i] = (int16)maxvalue;
		else if ( ( _contrastRate *   ( original[i] + _brightnessRate ) ) < -maxvalue ) pixel[i] = (int16)(-maxvalue);
		else pixel[i] = (int16)( _contrastRate *   ( original[i] + _brightnessRate ) );
	    }
	    glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
	                  GL_LUMINANCE, GL_SHORT, pixel );
	    delete[] pixel;
	    
	}
	break;

	default:
	break;

    }
    glEnable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, texName );

    glBegin( GL_QUADS );
        glTexCoord2d(0.0,0.0); glVertex2d(  0.0,   0.0);
        glTexCoord2d(1.0,0.0); glVertex2d(width,   0.0);
        glTexCoord2d(1.0,1.0); glVertex2d(width,height);
        glTexCoord2d(0.0,1.0); glVertex2d(  0.0,height);
        glEnd();
    glDeleteTextures( 1, &texName );
    
    if ( !_shapes.empty() )
    {
        for ( std::list< Selection::m4dShape<int> >::iterator it = _shapes.begin(); it != --(_shapes.end()); ++it )
            drawShape( *it, false, sliceNum, zoomRate );
        drawShape( *(--(_shapes.end())), true, sliceNum, zoomRate );
    }
    if ( _flipH < 0 ) offset.setX( offset.x() - (int)( zoomRate * w ) );
    if ( _flipV < 0 ) offset.setY( offset.y() - (int)( zoomRate * h ) );
    if ( _printData ) drawData( zoomRate, offset );
    glFlush();
}

void
m4dGUISliceViewerWidget::drawSelectionModeBorder()
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
m4dGUISliceViewerWidget::drawSelectedBorder()
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
m4dGUISliceViewerWidget::drawShape( Selection::m4dShape<int>& s, bool last, int sliceNum, float zoomRate )
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
m4dGUISliceViewerWidget::drawData( double zoomRate, QPoint offset )
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f( 1., 1., 1. );
    std::map< std::string, std::string >::iterator it;
    int w, h;
    w = (int)(_maximum[0] - _minimum[0]),
    h = (int)(_maximum[1] - _minimum[1]);
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
m4dGUISliceViewerWidget::resizeGL(int winW, int winH)
{
    glViewport(0, 0, width(), height());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (double)winW, 0.0, (double)winH, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    if ( !_ready )
    {
        setParameters();
	if ( !_ready ) return;
    }
    if ( _inPort->IsPlugged() )
    {
        int w, h;
	w = (int)(_maximum[0] - _minimum[0]),
    	h = (int)(_maximum[1] - _minimum[1]);
        if ( (double)width() / (double)w < (double)height() / (double)h ) _zoomRate = (double)width() / (double)w;
	else _zoomRate = (double)height() / (double)h;
    }
    updateGL();
}

void
m4dGUISliceViewerWidget::mousePressEvent(QMouseEvent *event)
{
    if ( !_selected )
    {
        setSelected();
	return;
    }
    if ( !_inPort->IsPlugged() ) return;
    if ( !_ready )
    {
        setParameters();
	if ( !_ready ) return;
    }
    _lastPos = event->pos();
    int w, h;
    QPoint offset;
    w = (int)(_maximum[0] - _minimum[0]),
    h = (int)(_maximum[1] - _minimum[1]);
    offset.setX( (int)floor( (double)_offset.x() - ( _zoomRate - (double)width()/w ) * 0.5 * w ) );
    offset.setY( (int)floor( (double)_offset.y() - ( _zoomRate - (double)height()/h ) * 0.5 * h ) );
    if ( ( event->buttons() & Qt::LeftButton ) && _selectionMode[ left ] )
    {
        if ( _oneSliceMode ) (this->*_selectMethods[ left ])( (int)( ( event->x() - offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - offset.y() ) / _zoomRate ), _sliceNum );
        else (this->*_selectMethods[ left ])( (int)( ( event->x() % ( ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
   					    (int)( ( ( this->height() - event->y() ) % ( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
					    _sliceNum + event->x() / (int)( ( width() - 1 ) / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) );
    }
    else if ( event->buttons() & Qt::RightButton && _selectionMode[ right ] )
    {
        if ( _oneSliceMode ) (this->*_selectMethods[ right ])( (int)( ( event->x() - offset.x() ) / _zoomRate ), (int)( ( this->height() - event->y() - offset.y() ) / _zoomRate ), _sliceNum );
        else (this->*_selectMethods[ right ])( (int)( ( event->x() % ( ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
   					     (int)( ( ( this->height() - event->y() ) % ( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
					     _sliceNum + event->x() / (int)( ( width() - 1 ) / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - event->y() ) / (int)( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) );
    }

    updateGL();
}

void
m4dGUISliceViewerWidget::mouseReleaseEvent(QMouseEvent *event)
{
    _lastPos = QPoint( -1, -1 );
}

void
m4dGUISliceViewerWidget::mouseMoveEvent(QMouseEvent *event)
{
    if ( !_inPort->IsPlugged() ) return;

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
m4dGUISliceViewerWidget::zoomImage( int dummy, int amount )
{
    _zoomRate += 0.001 * amount;
    if ( _zoomRate < 0. ) _zoomRate = 0.;
    emit signalZoom( _index, amount );
}

void
m4dGUISliceViewerWidget::wheelEvent(QWheelEvent *event)
{
    if ( !_inPort->IsPlugged() ) return;

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
m4dGUISliceViewerWidget::switchSlice( int dummy, int amount )
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
m4dGUISliceViewerWidget::setSliceNum( size_t num )
{
    if ( !_inPort->IsPlugged() ) return;
        if ( num < _minimum[2] ||
             num >= _maximum[2] )
        {
            throw ErrorHandling::ExceptionBase( "Index out of bounds." );
        }
    _sliceNum = num;
    emit signalSetSliceNum( _index, num );
}

void
m4dGUISliceViewerWidget::moveImage( int amountH, int amountV )
{
    _offset.setX( _offset.x() + amountH );
    _offset.setY( _offset.y() + amountV );
    emit signalMove( _index, amountH, amountV );
}

void
m4dGUISliceViewerWidget::adjustContrastBrightness( int amountC, int amountB )
{
    _brightnessRate += amountB;
    _contrastRate += ((GLfloat)amountC)/((GLfloat)width()/2.0);
    emit signalAdjustContrastBrightness( _index, amountC, amountB );
}

void
m4dGUISliceViewerWidget::newPoint( int x, int y, int z )
{
    if ( !_inPort->IsPlugged() ) return;
    if ( _shapes.empty() ) newShape( x, y, z );
    else
    {
	if ( _flipH < 0 ) x = - ( x - (int)( _maximum[0] - _minimum[0] ));
        if ( _flipV < 0 ) y = - ( y - (int)( _maximum[1] - _minimum[1] ));
        if ( x < 0 || y < 0 ||
             x >= (int)( _maximum[0] - _minimum[0] ) ||
             y >= (int)( _maximum[1] - _minimum[1] ) )
	{
	    return;
	}
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
m4dGUISliceViewerWidget::newShape( int x, int y, int z )
{
    if ( !_inPort->IsPlugged() ) return;
    if ( x < 0 || y < 0 ||
         x >= (int)( _maximum[0] - _minimum[0] ) ||
         y >= (int)( _maximum[1] - _minimum[1] ) )
    {
        return;
    }

    Selection::m4dShape<int> s( 3 );
    _shapes.push_back( s );
    newPoint( x, y, z );
    if ( _flipH < 0 ) x = - ( x - (int)( _maximum[0] - _minimum[0] ));
    if ( _flipV < 0 ) y = - ( y - (int)( _maximum[1] - _minimum[1] ));
    emit signalNewShape( _index, x, y, z );
}

void
m4dGUISliceViewerWidget::deletePoint()
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
m4dGUISliceViewerWidget::deleteShape()
{
    if ( _shapes.empty() ) return;
    
    _shapes.pop_back();
    emit signalDeleteShape( _index );
}

void
m4dGUISliceViewerWidget::deleteAll()
{
    while ( !_shapes.empty() ) deleteShape();
    emit signalDeleteAll( _index );
}

void
m4dGUISliceViewerWidget::slotSetButtonHandler( ButtonHandler hnd, MouseButton btn )
{
    setButtonHandler( hnd, btn );
}

void
m4dGUISliceViewerWidget::slotToggleFlipHorizontal()
{
    toggleFlipHorizontal();
}

void
m4dGUISliceViewerWidget::slotToggleFlipVertical()
{
    toggleFlipVertical();
}

void
m4dGUISliceViewerWidget::slotAddLeftSideData( std::string type, std::string data )
{
    addLeftSideData( type, data );
}

void
m4dGUISliceViewerWidget::slotAddRightSideData( std::string type, std::string data )
{
    addRightSideData( type, data );
}

void
m4dGUISliceViewerWidget::slotEraseLeftSideData( std::string type )
{
    eraseLeftSideData( type );
}

void
m4dGUISliceViewerWidget::slotEraseRightSideData( std::string type )
{
    eraseRightSideData( type );
}

void
m4dGUISliceViewerWidget::slotClearLeftSideData()
{
    clearLeftSideData();
}

void
m4dGUISliceViewerWidget::slotClearRightSideData()
{
    clearRightSideData();
}

void
m4dGUISliceViewerWidget::slotTogglePrintData()
{
    togglePrintData();
}

void
m4dGUISliceViewerWidget::slotTogglePrintShapeData()
{
    if ( !_printShapeData ) _printShapeData = true;
    else _printShapeData = false;
    updateGL();
    emit signalTogglePrintShapeData();
}

void
m4dGUISliceViewerWidget::slotSetSliceNum( size_t num )
{
    setSliceNum( num );
}

void
m4dGUISliceViewerWidget::slotZoom( int amount )
{
    zoomImage( 0, amount );
}

void
m4dGUISliceViewerWidget::slotMove( int amountH, int amountV )
{
    moveImage( amountH, amountV );
}

void
m4dGUISliceViewerWidget::slotAdjustContrastBrightness( int amountC, int amountB )
{
    adjustContrastBrightness( amountC, amountB );
}

void
m4dGUISliceViewerWidget::slotNewPoint( int x, int y, int z )
{
    newPoint( x, y, z );
}

void
m4dGUISliceViewerWidget::slotNewShape( int x, int y, int z )
{
    newShape( x, y, z );
}

void
m4dGUISliceViewerWidget::slotDeletePoint()
{
    deletePoint();
    updateGL();
}

void
m4dGUISliceViewerWidget::slotDeleteShape()
{
    deleteShape();
    updateGL();
}

void
m4dGUISliceViewerWidget::slotDeleteAll()
{
    deleteAll();
    updateGL();
}

void
m4dGUISliceViewerWidget::slotSetSelected( bool selected )
{
    if ( selected ) setSelected();
    else setUnSelected();
}

void
m4dGUISliceViewerWidget::slotSetOneSliceMode()
{
    setOneSliceMode();
}

void
m4dGUISliceViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow )
{
    setMoreSliceMode( slicesPerRow );
}

void
m4dGUISliceViewerWidget::slotRotateAxisX( double x )
{
}

void
m4dGUISliceViewerWidget::slotRotateAxisY( double y )
{
}

void
m4dGUISliceViewerWidget::slotRotateAxisZ( double z )
{
}

} /*namespace Viewer*/
} /*namespace M4D*/
