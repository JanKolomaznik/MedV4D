#include "GUI/m4dGUISliceViewerWidget.h"

#include <QtGui>
#include "GUI/ogl/fonts.h"
#include <sstream>

#define MINIMUM_SELECT_DISTANCE			5

#define FONT_WIDTH				8
#define FONT_HEIGHT				16
#define BRIGHTNESS_MULTIPLICATOR		16

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class VoxelArrayCopier
{
public:
    static void copy( ElementType* dst, ElementType* src, uint32 width, uint32 height, uint32 depth, int32 xstride, int32 ystride, int32 zstride )
    {
        uint32 i;
	for ( i = 0; i < height * width; i++ )  dst[ i ] = src[ ( i % width ) * xstride + ( i / width ) * ystride + depth * zstride ];
    }
private:
    VoxelArrayCopier();
    VoxelArrayCopier( const VoxelArrayCopier& );
    const VoxelArrayCopier& operator=( const VoxelArrayCopier& );
};

template< typename ElementType >
class TexturePreparer
{
public:
    static GLenum oglType();
    static bool prepare( Imaging::InputPortAbstractImage* inPort, uint32& width, uint32& height, GLint brightnessRate, GLfloat contrastRate, m4dGUIAbstractViewerWidget::SliceOrientation so, uint32 slice )
    {
        uint32 depth;
        double maxvalue;
	bool unsgn;
	if ( typeid( ElementType ) == typeid( uint8 ) || typeid( ElementType ) == typeid( uint16 ) || typeid( ElementType ) == typeid( uint32 ) || typeid( ElementType ) == typeid( uint64 ) )
	{
	    maxvalue = pow( (double)256, (double)sizeof( ElementType ) ) - 1;
	    unsgn = true;
	}
	else
	{
	    maxvalue = (int)( pow( (double)256, (double)sizeof( ElementType ) ) / 2 - 1 );
	    unsgn = false;
	}
	double multiplicator = pow( (double)BRIGHTNESS_MULTIPLICATOR, (double)sizeof( ElementType ) - 1 );
        int32 xstride, ystride, zstride;
        bool ready = true;
	ElementType* pixel, *original;
	try
	{
	    if ( inPort->TryLockDataset() )
	    {
		try
		{
		    if ( inPort->GetAbstractImage().GetDimension() == 2 )
		    {
		        original = Imaging::Image< ElementType, 2 >::CastAbstractImage(inPort->GetAbstractImage()).GetPointer( width, height, xstride, ystride );
			depth = zstride = 0;
			slice = 0;
		    }
		    else if ( inPort->GetAbstractImage().GetDimension() == 3 )
		    {
		        switch ( so )
		        {
			    case m4dGUIAbstractViewerWidget::xy:
			    {
		                original = Imaging::Image< ElementType, 3 >::CastAbstractImage(inPort->GetAbstractImage()).GetPointer( width, height, depth, xstride, ystride, zstride );
			        break;
			    }

			    case m4dGUIAbstractViewerWidget::yz:
			    {
		                original = Imaging::Image< ElementType, 3 >::CastAbstractImage(inPort->GetAbstractImage()).GetPointer( depth, width, height, zstride, xstride, ystride );
			        break;
			    }

			    case m4dGUIAbstractViewerWidget::zx:
			    {
		                original = Imaging::Image< ElementType, 3 >::CastAbstractImage(inPort->GetAbstractImage()).GetPointer( height, depth, width, ystride, zstride, xstride );
			        break;
			    }
		        }
		    }
		} catch (...) { ready = false; }
		inPort->ReleaseDatasetLock();
		if ( !ready ) return ready;
	    }
	    else
	    {
	        ready = false;
		return ready;
	    }
	}
	catch (...) { ready = false; }
	if ( !ready ) return ready;

	pixel = new ElementType[ height * width ];
	VoxelArrayCopier<ElementType>::copy( pixel, original, width, height, slice, xstride, ystride, zstride );
	unsigned i;
	double mean;
	mean = 0.;
	for ( i = 0; i < width * height; i++ ) mean += (double)pixel[i] / (double)(width*height);
	mean += brightnessRate * multiplicator;
	for ( i = 0; i < width * height; ++i )
	{
	    if ( ( contrastRate *   ( pixel[i]- mean + brightnessRate * multiplicator ) + mean ) > maxvalue ) pixel[i] = (ElementType)maxvalue;
	    else if ( ( contrastRate *   ( pixel[i] - mean + brightnessRate * multiplicator ) + mean ) < ( unsgn ? 0 : -maxvalue ) ) pixel[i] = ( unsgn ? 0 : (ElementType)(-maxvalue) );
	    else pixel[i] = (ElementType)( contrastRate *   ( pixel[i] - mean + brightnessRate * multiplicator ) + mean );
	}
	glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
	              GL_LUMINANCE, oglType(), pixel );
	delete[] pixel;
	return ready;
    }
private:
    TexturePreparer();
    TexturePreparer( const TexturePreparer& );
    const TexturePreparer& operator=( const TexturePreparer& );
	    
};

template<>
GLenum
TexturePreparer<uint8>::oglType()
{
    return GL_UNSIGNED_BYTE;
}

template<>
GLenum
TexturePreparer<int8>::oglType()
{
    return GL_BYTE;
}

template<>
GLenum
TexturePreparer<uint16>::oglType()
{
    return GL_UNSIGNED_SHORT;
}

template<>
GLenum
TexturePreparer<int16>::oglType()
{
    return GL_SHORT;
}

template<>
GLenum
TexturePreparer<uint32>::oglType()
{
    return GL_UNSIGNED_INT;
}

template<>
GLenum
TexturePreparer<int32>::oglType()
{
    return GL_INT;
}

template<>
GLenum
TexturePreparer<uint64>::oglType()
{
    throw ErrorHandling::ExceptionBase( "64-bit numbers are not supported." );
    return GL_UNSIGNED_INT;
}

template<>
GLenum
TexturePreparer<int64>::oglType()
{
    throw ErrorHandling::ExceptionBase( "64-bit numbers are not supported." );
    return GL_INT;
}

m4dGUISliceViewerWidget::m4dGUISliceViewerWidget( unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    _inPort = new Imaging::InputPortAbstractImage();
    resetParameters();
    _inputPorts.AddPort( _inPort );
    _selected = false;
    _sliceOrientation = xy;
    setInputPort( );
}

m4dGUISliceViewerWidget::m4dGUISliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    _inPort = new Imaging::InputPortAbstractImage();
    resetParameters();
    _inputPorts.AddPort( _inPort );
    _selected = false;
    _sliceOrientation = xy;
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
    _ready = false;
    _inPort->UnPlug();
    setParameters();
    resizeGL( width(), height() );
    updateGL();
}

void
m4dGUISliceViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
{
    _ready = false;
    conn->ConnectConsumer( *_inPort );
    setParameters();
    resizeGL( width(), height() );
    updateGL();
}

void
m4dGUISliceViewerWidget::resetParameters()
{
    qRegisterMetaType<Imaging::PipelineMsgID>( "Imaging::PipelineMsgID" );
    m4dGUIAbstractViewerWidget::connect( (m4dGUIAbstractViewerWidget*)this, SIGNAL(signalMessageHandler( Imaging::PipelineMsgID )), (m4dGUIAbstractViewerWidget*)this, SLOT(slotMessageHandler( Imaging::PipelineMsgID )), Qt::QueuedConnection );
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
		    _minimum[ 0 ] = _inPort->GetAbstractImage().GetDimensionExtents(0).minimum;
		    _minimum[ 1 ] = _inPort->GetAbstractImage().GetDimensionExtents(1).minimum;
		    _minimum[ 2 ] = _inPort->GetAbstractImage().GetDimensionExtents(2).minimum;
		    _maximum[ 0 ] = _inPort->GetAbstractImage().GetDimensionExtents(0).maximum;
		    _maximum[ 1 ] = _inPort->GetAbstractImage().GetDimensionExtents(1).maximum;
		    _maximum[ 2 ] = _inPort->GetAbstractImage().GetDimensionExtents(2).maximum;
	            _sliceNum = _minimum[ ( _sliceOrientation + 2 ) % 3 ];
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
    _oneSliceMode = true;
    _slicesPerRow = 1;
    _flipH = _flipV = 1;
    _slicePicked = 0;
    _colorPicker = false;
    _colorPicked = 0;
    _pickedPosition = QPoint( -1, -1 );
    _shapes.clear();
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
    _availableSlots.push_back( SETSLICEORIENTATION );
    _availableSlots.push_back( COLORPICKER );
    _leftSideData.clear();
    _rightSideData.clear();
    _ready = true;
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
	    	    _minimum[ 0 ] = _inPort->GetAbstractImage().GetDimensionExtents(0).minimum;
	    	    _minimum[ 1 ] = _inPort->GetAbstractImage().GetDimensionExtents(1).minimum;
	    	    _minimum[ 2 ] = _inPort->GetAbstractImage().GetDimensionExtents(2).minimum;
	    	    _maximum[ 0 ] = _inPort->GetAbstractImage().GetDimensionExtents(0).maximum;
	    	    _maximum[ 1 ] = _inPort->GetAbstractImage().GetDimensionExtents(1).maximum;
	    	    _maximum[ 2 ] = _inPort->GetAbstractImage().GetDimensionExtents(2).maximum;
		    _sliceNum = _minimum[ ( _sliceOrientation + 2 ) % 3 ];
		}
		catch (...) { _ready = false; }
	        _inPort->ReleaseDatasetLock();
	        _ready = true;
            }
	}
	catch (...) { _ready = false; }
    }
    _shapes.clear();
    _leftSideData.clear();
    _rightSideData.clear();
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
    emit signalMessageHandler( msg->msgID );
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

	case color_picker:
	_selectMethods[btn] = &M4D::Viewer::m4dGUISliceViewerWidget::colorPicker;
	_selectionMode[btn] = true;
	break;

	default:
	throw ErrorHandling::ExceptionBase( "Unsupported button handler." );
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
	w = (double)(_maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ]),
        h = (double)(_maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ]);
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
    if ( _inPort->IsPlugged() ) drawPluggedBorder();
    glFlush();
}

void
m4dGUISliceViewerWidget::drawSlice( int sliceNum, double zoomRate, QPoint offset )
{
    if ( !_ready ) return;
    if ( !_inPort->IsPlugged() ) return;
    int w, h;
    if ( sliceNum < (int)_minimum[ ( _sliceOrientation + 2 ) % 3 ] ||
         sliceNum >= (int)_maximum[ ( _sliceOrientation + 2 ) % 3 ] )
    {
        return;
    }
    else
    {
        w = (int)_maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ],
        h = (int)_maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ];
    }
    glLoadIdentity();
    if ( _flipH < 0 ) offset.setX( offset.x() + (int)( zoomRate * w ) );
    if ( _flipV < 0 ) offset.setY( offset.y() + (int)( zoomRate * h ) );
    uint32 height, width;
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
    
    INTEGER_TYPE_TEMPLATE_SWITCH_MACRO(
    	_imageID, _ready = TexturePreparer<TTYPE>::prepare( _inPort, width, height, _brightnessRate, _contrastRate, _sliceOrientation, sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ]) )
    
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
    if ( _colorPicker && sliceNum == _slicePicked ) drawPicked();
    glFlush();
}

void
m4dGUISliceViewerWidget::borderDrawer(GLfloat red, GLfloat green, GLfloat blue, unsigned pos)
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f(red, green, blue);
    glBegin(GL_LINE_LOOP);
        glVertex2i( pos, pos );
	glVertex2i( pos, this->height() - 1 - pos );
	glVertex2i( this->width() - 1 - pos, this->height() - 1 - pos );
	glVertex2i( this->width() - 1 - pos, pos );
    glEnd();
    glPopMatrix();
}

void
m4dGUISliceViewerWidget::drawPluggedBorder()
{
    borderDrawer( 0., 0., 1., 2 );
}

void
m4dGUISliceViewerWidget::drawSelectionModeBorder()
{
    borderDrawer(1., 0., 0., 0);
}

void
m4dGUISliceViewerWidget::drawSelectedBorder()
{
    borderDrawer(0., 1., 0., 1);
}

void
m4dGUISliceViewerWidget::drawShape( Selection::m4dShape<int>& s, bool last, int sliceNum, float zoomRate )
{
    if ( last ) glColor3f( 1., 0., 0. );
    else glColor3f( 0., 0., 1. );
    if ( s.shapeClosed() && s.shapeElements().size() > 1 &&
	  ( s.shapeElements().back().getParticularValue( 2 ) == sliceNum || s.shapeElements().front().getParticularValue( 2 ) == sliceNum ) )
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
    std::list< Selection::m4dPoint<int> >::iterator it, tmp;
    for ( it = s.shapeElements().begin(); it != s.shapeElements().end(); ++it )
    {
    	tmp = it;
	++tmp;
	if ( &(*it) != &(s.shapeElements().back()) &&
	   ( it->getParticularValue( 2 ) == sliceNum || tmp->getParticularValue( 2 ) == sliceNum ) )
	{
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
        if  ( it->getParticularValue( 2 ) == sliceNum )
        {
	    if ( last && &(*it) == &(s.shapeElements().back()) ) glColor3f( 1., 0., 1. );
            glBegin(GL_QUADS);
	        glVertex2i( it->getParticularValue( 0 ) - 3, it->getParticularValue( 1 ) - 3 );
	        glVertex2i( it->getParticularValue( 0 ) + 3, it->getParticularValue( 1 ) - 3 );
	        glVertex2i( it->getParticularValue( 0 ) + 3, it->getParticularValue( 1 ) + 3 );
	        glVertex2i( it->getParticularValue( 0 ) - 3, it->getParticularValue( 1 ) + 3 );
	    glEnd();
        }
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
    w = (int)(_maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ]),
    h = (int)(_maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ]);
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
m4dGUISliceViewerWidget::drawPicked()
{
    glColor3f( 1., 1., 1. );
    setTextPosition( _pickedPosition.x(), _pickedPosition.y() );
    setTextCoords( _pickedPosition.x(), _pickedPosition.y() );
    std::ostringstream pick;
    pick << _colorPicked;
    drawText( pick.str().c_str() );
    unsetTextCoords();
    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
}

void
m4dGUISliceViewerWidget::ImagePositionSelectionCaller( int x, int y, SelectMethods f )
{
    int w, h;
    QPoint offset;
    w = (int)(_maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ]),
    h = (int)(_maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ]);
    offset.setX( (int)floor( (double)_offset.x() - ( _zoomRate - (double)width()/w ) * 0.5 * w ) );
    offset.setY( (int)floor( (double)_offset.y() - ( _zoomRate - (double)height()/h ) * 0.5 * h ) );
    if ( _oneSliceMode ) (this->*f)( (int)( ( x - offset.x() ) / _zoomRate ), (int)( ( this->height() - y - offset.y() ) / _zoomRate ), _sliceNum );
    else (this->*f)( (int)( ( x % ( ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
   				    (int)( ( ( this->height() - y ) % ( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) * _slicesPerRow * w / ( width() - 1 ) ),
				    _sliceNum + x / (int)( ( width() - 1 ) / _slicesPerRow ) + _slicesPerRow * ( ( this->height() - y ) / (int)( ( h / w ) * ( width() - 1 ) / _slicesPerRow ) ) );
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
	w = (int)(_maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ]),
    	h = (int)(_maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ]);
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
    if ( ( event->buttons() & Qt::LeftButton ) && _selectionMode[ left ] )
    	ImagePositionSelectionCaller( event->x(), event->y(), _selectMethods[ left ] );
    else if ( event->buttons() & Qt::RightButton && _selectionMode[ right ] )
    	ImagePositionSelectionCaller( event->x(), event->y(), _selectMethods[ right ] );

    updateGL();
}

void
m4dGUISliceViewerWidget::mouseReleaseEvent(QMouseEvent *event)
{
    _lastPos = QPoint( -1, -1 );
    if ( _colorPicker )
    {
        _colorPicker = false;
        updateGL();
    }
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
    if ( _colorPicker )
        ImagePositionSelectionCaller( event->x(), event->y(), &M4D::Viewer::m4dGUISliceViewerWidget::colorPicker );

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
    if ( _colorPicker )
        ImagePositionSelectionCaller( event->x(), event->y(), &M4D::Viewer::m4dGUISliceViewerWidget::colorPicker );
    
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
        if ( num < _minimum[ ( _sliceOrientation + 2 ) % 3 ] ||
             num >= _maximum[ ( _sliceOrientation + 2 ) % 3 ] )
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
	if ( _flipH < 0 ) x = - ( x - (int)( _maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ] ));
        if ( _flipV < 0 ) y = - ( y - (int)( _maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ] ));
        if ( x < 0 || y < 0 ||
             x >= (int)( _maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ] ) ||
             y >= (int)( _maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ] ) )
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
         x >= (int)( _maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ] ) ||
         y >= (int)( _maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ] ) )
    {
        return;
    }

    Selection::m4dShape<int> s( 3 );
    _shapes.push_back( s );
    newPoint( x, y, z );
    if ( _flipH < 0 ) x = - ( x - (int)( _maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ] ));
    if ( _flipV < 0 ) y = - ( y - (int)( _maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ] ));
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
m4dGUISliceViewerWidget::colorPicker( int x, int y, int z )
{
    if ( !_inPort->IsPlugged() ) return;
    int64 result;
    if ( _flipH < 0 ) x = - ( x - (int)( _maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ] ));
    if ( _flipV < 0 ) y = - ( y - (int)( _maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ] ));
    if ( x < 0 || y < 0 ||
         x >= (int)( _maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ] ) ||
         y >= (int)( _maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ] ) )
    {
        return;
    }
    if ( !_ready ) setParameters();
    if ( !_ready ) return;
    try
    {
	if ( _inPort->TryLockDataset() )
	{
            try
	    {
		if ( _inPort->GetAbstractImage().GetDimension() == 3 )
		{
		    INTEGER_TYPE_TEMPLATE_SWITCH_MACRO(
		        _imageID, result = Imaging::Image< TTYPE, 3 >::CastAbstractImage(_inPort->GetAbstractImage()).GetElement( x, y, z ) );
		}
	        else if ( _inPort->GetAbstractImage().GetDimension() == 2 )
	        {
		    INTEGER_TYPE_TEMPLATE_SWITCH_MACRO(
		        _imageID, result = Imaging::Image< TTYPE, 2 >::CastAbstractImage(_inPort->GetAbstractImage()).GetElement( x, y ) );
	        }
	        else
	            result = 0;
	    }
	    catch (...) { _ready = false; }
	    _inPort->ReleaseDatasetLock();
	}
        else
            return;
    }
    catch (...) { _ready = false; }
    if ( !_ready ) return;
    _slicePicked = z;
    _colorPicker = true;
    _colorPicked = result;
    _pickedPosition = QPoint( x, y );
    emit signalColorPicker( _index, result );
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

void
m4dGUISliceViewerWidget::slotToggleSliceOrientation()
{
    switch ( _sliceOrientation )
    {
        case xy:
	_sliceOrientation = yz;
	break;

        case yz:
	_sliceOrientation = zx;
	break;

        case zx:
	_sliceOrientation = xy;
	break;
    }
    if ( _sliceNum >= (int)_maximum[ ( _sliceOrientation + 2 ) % 3 ] ) _sliceNum = _maximum[ ( _sliceOrientation + 2 ) % 3 ] - 1;
    if ( _sliceNum < (int)_minimum[ ( _sliceOrientation + 2 ) % 3 ] ) _sliceNum = _minimum[ ( _sliceOrientation + 2 ) % 3 ];
    updateGL();
    emit signalToggleSliceOrientation( _index );
}

void
m4dGUISliceViewerWidget::slotColorPicker( int x, int y, int z )
{
    colorPicker( x, y, z );
}

void
m4dGUISliceViewerWidget::slotMessageHandler( Imaging::PipelineMsgID msgID )
{
    switch( msgID )
    {
        case Imaging::PMI_DATASET_PUT:
	case Imaging::PMI_PORT_PLUGGED:
        case Imaging::PMI_FILTER_UPDATED:
	{
	    setParameters();
            resizeGL( width(), height() );
            updateGL();
	}
	break;
	
	default:
	break;
    }
}

} /*namespace Viewer*/
} /*namespace M4D*/
