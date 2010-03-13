/**
 *  @ingroup gui
 *  @file m4dGUISliceViewerWidget.cpp
 *  @brief some brief
 */
#include "GUI/widgets/m4dGUISliceViewerWidget.h"
#include "GUI/widgets/components/RGBSliceViewerTexturePreparer.h"

#include <QtGui>
#include "GUI/widgets/ogl/fonts.h"
#include <sstream>
#include <cmath>

#define MINIMUM_SELECT_DISTANCE			5

#define FONT_WIDTH				8
#define FONT_HEIGHT				16


namespace M4D
{
namespace Viewer
{

m4dGUISliceViewerWidget::m4dGUISliceViewerWidget( unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    _inPort = new Imaging::InputPortTyped<Imaging::AImage>();
    resetParameters();
    _inputPorts.AppendPort( _inPort );
    for ( uint32 i = 1; i < SLICEVIEWER_INPUT_NUMBER; ++i ) _inputPorts.AppendPort( new Imaging::InputPortTyped<Imaging::AImage>() );
    setInputPort( );
}

m4dGUISliceViewerWidget::m4dGUISliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent)
    : QGLWidget(parent)
{
    _index = index;
    _inPort = new Imaging::InputPortTyped<Imaging::AImage>();
    resetParameters();
    _inputPorts.AppendPort( _inPort );
    for ( uint32 i = 1; i < SLICEVIEWER_INPUT_NUMBER; ++i ) _inputPorts.AppendPort( new Imaging::InputPortTyped<Imaging::AImage>() );
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
    calculateOptimalZoomRate();
    updateGL();
}

void
m4dGUISliceViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
{
    if ( !conn )
    {
        setInputPort();
	return;
    }
    _ready = false;
    conn->ConnectConsumer( *_inPort );
    setParameters();
    calculateOptimalZoomRate();
    updateGL();
}

void
m4dGUISliceViewerWidget::setTexturePreparerToSimple()
{
    texturePreparerType = simple;
}

void
m4dGUISliceViewerWidget::setTexturePreparerToRGB()
{
    texturePreparerType = rgb;
}

void
m4dGUISliceViewerWidget::setTexturePreparerToCustom(AbstractSliceViewerTexturePreparer* ctp)
{
    if ( ctp )
    {
        texturePreparerType = custom;
        customTexturePreparer = ctp;
    }
}

void
m4dGUISliceViewerWidget::resetParameters()
{
    _selected = false;
    _sliceOrientation = xy;
    _ready = false;
    _dimension = 0;
    texturePreparerType = simple;
    if ( _inPort->IsPlugged() )
    {
        try
	{
	    if ( _inPort->TryLockDataset() )
	    {
                try
	        {
        	    _imageID = _inPort->GetDatasetTyped().GetElementTypeID();
		    _minimum[ 0 ] = _inPort->GetDatasetTyped().GetDimensionExtents(0).minimum;
		    _minimum[ 1 ] = _inPort->GetDatasetTyped().GetDimensionExtents(1).minimum;
		    _maximum[ 0 ] = _inPort->GetDatasetTyped().GetDimensionExtents(0).maximum;
		    _maximum[ 1 ] = _inPort->GetDatasetTyped().GetDimensionExtents(1).maximum;
	    	    _extents[ 0 ] = _inPort->GetDatasetTyped().GetDimensionExtents(0).elementExtent;
	    	    _extents[ 1 ] = _inPort->GetDatasetTyped().GetDimensionExtents(1).elementExtent;
		    if ( _inPort->GetDatasetTyped().GetDimension() == 3 )
		    {
		        _minimum[ 2 ] = _inPort->GetDatasetTyped().GetDimensionExtents(2).minimum;
		        _maximum[ 2 ] = _inPort->GetDatasetTyped().GetDimensionExtents(2).maximum;
	    	        _extents[ 2 ] = _inPort->GetDatasetTyped().GetDimensionExtents(2).elementExtent;
	                _sliceNum = _minimum[ ( _sliceOrientation + 2 ) % 3 ];
			_dimension = 3;
		    }
		    else
		    {
		        _minimum[ 2 ] = 0;
		        _maximum[ 2 ] = 1;
	    	        _extents[ 2 ] = 1;
	                _sliceNum = 0;
			_sliceOrientation = xy;
			_dimension = 2;
		    }
		    _oldTimeStamp = _inPort->GetDatasetTyped().GetStructureTimestamp();
		    unsigned typeSize;
		    bool unsgn;
		    NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( _imageID, { typeSize = sizeof( TTYPE ); unsgn = ( typeid( TTYPE ) == typeid( uint8 ) || typeid( TTYPE ) == typeid( uint16 ) || typeid( TTYPE ) == typeid( uint32 ) || typeid( TTYPE ) == typeid( uint64 ) ); } );
		    typeSize = (unsigned)std::pow( (double)256, (double)(typeSize - 1) );
                    _brightnessRate = 12 * typeSize;
                    _contrastRate = 20 * typeSize;
		    if ( unsgn ) _brightnessRate *= 2;
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
    slotSetButtonHandler( moveI, left );
    slotSetButtonHandler( switch_slice, right );
    _printShapeData = true;
    _printData = true;
    _oneSliceMode = true;
    _slicesPerRow = 1;
    _slicesPerColumn = 1;
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
    _dimension = 0;
    if ( _inPort->IsPlugged() )
    {
        try
	{
	    if ( _inPort->TryLockDataset() )
            {
	        try
		{
        	    _imageID = _inPort->GetDatasetTyped().GetElementTypeID();
		    _minimum[ 0 ] = _inPort->GetDatasetTyped().GetDimensionExtents(0).minimum;
		    _minimum[ 1 ] = _inPort->GetDatasetTyped().GetDimensionExtents(1).minimum;
		    _maximum[ 0 ] = _inPort->GetDatasetTyped().GetDimensionExtents(0).maximum;
		    _maximum[ 1 ] = _inPort->GetDatasetTyped().GetDimensionExtents(1).maximum;
	    	    _extents[ 0 ] = _inPort->GetDatasetTyped().GetDimensionExtents(0).elementExtent;
	    	    _extents[ 1 ] = _inPort->GetDatasetTyped().GetDimensionExtents(1).elementExtent;
		    if ( _inPort->GetDatasetTyped().GetDimension() == 3 )
		    {
		        _minimum[ 2 ] = _inPort->GetDatasetTyped().GetDimensionExtents(2).minimum;
		        _maximum[ 2 ] = _inPort->GetDatasetTyped().GetDimensionExtents(2).maximum;
	    	        _extents[ 2 ] = _inPort->GetDatasetTyped().GetDimensionExtents(2).elementExtent;
	                _sliceNum = _minimum[ ( _sliceOrientation + 2 ) % 3 ];
			_dimension = 3;
		    }
		    else
		    {
		        _minimum[ 2 ] = 0;
		        _maximum[ 2 ] = 1;
	    	        _extents[ 2 ] = 1;
	                _sliceNum = 0;
			_sliceOrientation = xy;
			_dimension = 2;
		    }
		    _oldTimeStamp = _inPort->GetDatasetTyped().GetStructureTimestamp();
		    unsigned typeSize;
		    bool unsgn;
		    NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( _imageID, { typeSize = sizeof( TTYPE ); unsgn = ( typeid( TTYPE ) == typeid( uint8 ) || typeid( TTYPE ) == typeid( uint16 ) || typeid( TTYPE ) == typeid( uint32 ) || typeid( TTYPE ) == typeid( uint64 ) ); } );
		    typeSize = (unsigned)std::pow( (double)256, (double)(typeSize - 1) );
                    _brightnessRate = 12 * typeSize;
                    _contrastRate = 20 * typeSize;
		    if ( unsgn ) _brightnessRate *= 2;
		}
		catch (...) { _ready = false; }
	        _inPort->ReleaseDatasetLock();
	        _ready = true;
            }
	}
	catch (...) { _ready = false; }
    }
    _shapes.clear();
    //_leftSideData.clear();
    //_rightSideData.clear();
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
m4dGUISliceViewerWidget::setButtonMethod( ButtonMethods bm, MouseButton btn )
{
    _buttonMethods[btn] = bm;
    _selectMode[btn] = false;
    _buttonMode[btn] = true;
}

void
m4dGUISliceViewerWidget::setSelectMethod( SelectMethods sm, MouseButton btn )
{
    _selectMethods[btn] = sm;
    _selectMode[btn] = true;
    _buttonMode[btn] = false;
}

void
m4dGUISliceViewerWidget::setButtonHandler( ButtonHandler hnd, MouseButton btn )
{
    switch (hnd)
    {
	case zoomI:
	setButtonMethod( &M4D::Viewer::m4dGUISliceViewerWidget::zoomImage, btn );
	break;

	case moveI:
	setButtonMethod( &M4D::Viewer::m4dGUISliceViewerWidget::moveImage, btn );
	break;

	case adjust_bc:
	setButtonMethod( &M4D::Viewer::m4dGUISliceViewerWidget::adjustContrastBrightness, btn );
	break;

	case switch_slice:
	setButtonMethod( &M4D::Viewer::m4dGUISliceViewerWidget::switchSlice, btn );
	break;

	case new_point:
	setSelectMethod( &M4D::Viewer::m4dGUISliceViewerWidget::newPoint, btn );
	break;

	case new_shape:
	setSelectMethod( &M4D::Viewer::m4dGUISliceViewerWidget::newShape, btn );
	break;

	case color_picker:
	setSelectMethod( &M4D::Viewer::m4dGUISliceViewerWidget::colorPicker, btn );
	break;

  case point_picker:
	setSelectMethod( &M4D::Viewer::m4dGUISliceViewerWidget::pointPicker, btn );
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
    _slicesPerColumn = 1;
    _oneSliceMode = true;
    if ( _availableSlots.back() != ZOOM ) _availableSlots.push_back( ZOOM );
    updateGL();
    emit signalSetOneSliceMode( _index );
}

void
m4dGUISliceViewerWidget::setMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn )
{
    _slicesPerRow = slicesPerRow;
    _slicesPerColumn = slicesPerColumn;
    _oneSliceMode = false;
    if ( _availableSlots.back() == ZOOM ) _availableSlots.pop_back();
    updateGL();
    emit signalSetMoreSliceMode( _index, slicesPerRow, slicesPerColumn );
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
m4dGUISliceViewerWidget::addLeftSideData( std::string data )
{
    _leftSideData.push_back( data );
    emit signalAddLeftSideData( data );
    if ( _printData ) updateGL();
}

void
m4dGUISliceViewerWidget::addRightSideData( std::string data )
{
    _rightSideData.push_back( data );
    emit signalAddRightSideData( data );
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
m4dGUISliceViewerWidget::calculateWidthHeight( double& w, double& h )
{
    w = (double)( (_maximum[ _sliceOrientation ] - _minimum[ _sliceOrientation ]) * _extents[ _sliceOrientation ] ),
    h = (double)( (_maximum[ ( _sliceOrientation + 1 ) % 3 ] - _minimum[ ( _sliceOrientation + 1 ) % 3 ]) * _extents[ ( _sliceOrientation + 1 ) % 3 ] );    
}

void
m4dGUISliceViewerWidget::paintGL()
{
    glClear( GL_COLOR_BUFFER_BIT );

    try
    {
         if ( !_ready ||
             _inPort->GetDatasetTyped().GetStructureTimestamp() != _oldTimeStamp ) setParameters();
    } catch (...) { _ready = false; }

    if ( _inPort->IsPlugged() && _ready )
    {
        unsigned i;
	double w, h;
	calculateWidthHeight( w, h );

        // check if in one slice mode or more slice mode
	if ( _oneSliceMode )
	{

	    // set new offset according to zoom rate
	    QPoint offset;
	    offset.setX( (int)floor( (double)_offset.x() - ( _zoomRate - (double)width()/w ) * 0.5 * w ) );
	    offset.setY( (int)floor( (double)_offset.y() - ( _zoomRate - (double)height()/h ) * 0.5 * h ) );
	    drawSlice( _sliceNum, _zoomRate, offset );
	
	}
        
	else
	{
        
	    double xgap = 0, ygap = 0;
	    double zoomRate = 1.;
	    
	    // set zoom rate and offset according to how many slices per row and per column are to be displayed
	    if ( ( (double)width() / (double)_slicesPerRow ) / w < ( (double)height() / (double)_slicesPerColumn ) / h )
	    {
	        ygap = ( ( (double)height() / (double)_slicesPerColumn ) - h * ( (double)width() / (double)_slicesPerRow ) / w ) / 2.;
	        zoomRate = ( (double)width() / (double)_slicesPerRow ) / w;
	    }
	    else
	    {
	        xgap = ( ( (double)width() / (double)_slicesPerRow ) - w * ( (double)height() / (double)_slicesPerColumn ) / h ) / 2.;
	        zoomRate = ( (double)height() / (double)_slicesPerColumn ) / h;
	    }
	    
	    // draw slices at given offsets
	    for ( i = 0; i < _slicesPerRow * _slicesPerColumn && _ready; ++i )
	         drawSlice( _sliceNum + i, zoomRate, QPoint( (int)( ( i % _slicesPerRow) * ( width() / _slicesPerRow ) + xgap ) , (int)( ( ( i / _slicesPerRow ) * ( height() / _slicesPerColumn ) + ygap ) ) ) );
	}
        if ( _selectMode[ left ] || _selectMode[ right ] ) drawSelectionModeBorder();
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
    double w, h;
    if ( sliceNum < (int)_minimum[ ( _sliceOrientation + 2 ) % 3 ] ||
         sliceNum >= (int)_maximum[ ( _sliceOrientation + 2 ) % 3 ] )
    {
        return;
    }
    else
    {
	calculateWidthHeight( w, h );
    }
    glLoadIdentity();
    
    // manage flips
    if ( _flipH < 0 ) offset.setX( offset.x() + (int)( zoomRate * w ) );
    if ( _flipV < 0 ) offset.setY( offset.y() + (int)( zoomRate * h ) );
    uint32 height, width;
    GLuint texName;

    // opengl texture setup functions
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
    
    // prepare texture
    if ( texturePreparerType == custom || texturePreparerType == rgb )
	for ( uint32 i = 1; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	{
	    if ( this->InputPort()[i].IsPlugged() ) break;
	    if ( i == SLICEVIEWER_INPUT_NUMBER - 1 ) texturePreparerType = simple;
	}

    switch ( texturePreparerType )
    {

	case rgb:
        NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
    	    _imageID, { RGBSliceViewerTexturePreparer<TTYPE> texturePreparer; _ready = texturePreparer.prepare( this->InputPort(), width, height, _brightnessRate, _contrastRate, _sliceOrientation, sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], _dimension ); } );
	break;

	case custom:
	_ready = customTexturePreparer->prepare( this->InputPort(), width, height, _brightnessRate, _contrastRate, _sliceOrientation, sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], _dimension );
	break;

	default:
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
            _imageID, { SimpleSliceViewerTexturePreparer<TTYPE> texturePreparer; _ready = texturePreparer.prepare( this->InputPort(), width, height, _brightnessRate, _contrastRate, _sliceOrientation, sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], _dimension ); } );
        break;

    }

    if ( !_ready ) return;
    
    glEnable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, texName );

    // draw surface and map texture on it
    glBegin( GL_QUADS );
        glTexCoord2d(0.0,0.0); glVertex2d(  0.0,   0.0);
        glTexCoord2d(1.0,0.0); glVertex2d(width * _extents[ _sliceOrientation ],   0.0);
        glTexCoord2d(1.0,1.0); glVertex2d(width * _extents[ _sliceOrientation ], height * _extents[ ( _sliceOrientation + 1 ) % 3 ] );
        glTexCoord2d(0.0,1.0); glVertex2d(  0.0,height * _extents[ ( _sliceOrientation + 1 ) % 3 ] );
        glEnd();
    glDeleteTextures( 1, &texName );
    
	drawSliceAdditionals( sliceNum, zoomRate );

	if ( _flipH < 0 ) offset.setX( offset.x() - (int)( zoomRate * w ) );
	if ( _flipV < 0 ) offset.setY( offset.y() - (int)( zoomRate * h ) );

	drawHUD( sliceNum, zoomRate, offset );
/*
    // if there are selected shapes, draw them
    if ( !_shapes.empty() )
    {
        for ( std::list< Selection::m4dShape<double> >::iterator it = _shapes.begin(); it != --(_shapes.end()); ++it )
            drawShape( *it, false, sliceNum, zoomRate );
        drawShape( *(--(_shapes.end())), true, sliceNum, zoomRate );
    }
    if ( _flipH < 0 ) offset.setX( offset.x() - (int)( zoomRate * w ) );
    if ( _flipV < 0 ) offset.setY( offset.y() - (int)( zoomRate * h ) );
    
    // print text data if requested
    if ( _printData ) drawData( zoomRate, offset, sliceNum );

    // print color value of the picked voxel
    if ( _colorPicker && sliceNum == _slicePicked ) drawPicked();
*/

    glFlush();
}

void
m4dGUISliceViewerWidget::drawSliceAdditionals( int sliceNum, double zoomRate )
{
	// if there are selected shapes, draw them
	if ( !_shapes.empty() )	{
		for ( std::list< Selection::m4dShape<double> >::iterator it = _shapes.begin(); it != --(_shapes.end()); ++it ) {
			drawShape( *it, false, sliceNum, zoomRate ); 
		}
		drawShape( *(--(_shapes.end())), true, sliceNum, zoomRate );
	}
}

void
m4dGUISliceViewerWidget::drawHUD( int sliceNum, double zoomRate, QPoint offset )
{
	// print text data if requested
	if ( _printData ) drawData( zoomRate, offset, sliceNum );

	// print color value of the picked voxel
	if ( _colorPicker && sliceNum == _slicePicked ) drawPicked();
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
m4dGUISliceViewerWidget::drawShape( Selection::m4dShape<double>& s, bool last, int sliceNum, double zoomRate )
{
    if ( last ) glColor3f( 1., 0., 0. );
    else glColor3f( 0., 0., 1. );
    
    // check if at least either the starting point or the ending point of the shape lies on the current slice and the shape is closed
    if ( s.shapeClosed() && s.shapeElements().size() > 1 &&
	  ( (int)( s.shapeElements().back().getParticularValue( ( _sliceOrientation + 2 ) % 3 ) / _extents[ ( _sliceOrientation + 2 ) % 3 ] ) == sliceNum ||
	    (int)(s.shapeElements().front().getParticularValue( ( _sliceOrientation + 2 ) % 3 ) / _extents[ ( _sliceOrientation + 2 ) % 3 ] ) == sliceNum ) )
    {
        // draw the line between the points
        glBegin(GL_LINES);
	    glVertex2i( (int)s.shapeElements().front().getParticularValue( _sliceOrientation ), (int)s.shapeElements().front().getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
	    glVertex2i(  (int)s.shapeElements().back().getParticularValue( _sliceOrientation ),  (int)s.shapeElements().back().getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
	glEnd();

	// print the length of the segment if requested
        if ( _printShapeData )
	{
	    if ( last ) glColor3f( 1., 1., 0. );
            else glColor3f( 0., 1., 1. );
	    Selection::m4dPoint< double > mid = Selection::m4dPoint< double >::midpoint( s.shapeElements().front(), s.shapeElements().back() );
	    std::ostringstream dist;
	    dist << Selection::m4dPoint< double >::distance( s.shapeElements().front(), s.shapeElements().back() );
	    setTextPosition( (int)mid.getParticularValue( _sliceOrientation ), (int)mid.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
	    setTextCoords( (int)mid.getParticularValue( _sliceOrientation ), (int)mid.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
            drawText( dist.str().c_str() );
	    unsetTextCoords();
	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
	    if ( last ) glColor3f( 1., 0., 0. );
            else glColor3f( 0., 0., 1. );
	}
    }

    // print the area size if requested
    if ( _printShapeData )
    {
        Selection::m4dPoint< double > c = s.getCentroid();
	float a = s.getArea();
	if ( a > 0 && sliceNum == (int)(c.getParticularValue( ( _sliceOrientation + 2 ) % 3 ) / _extents[ ( _sliceOrientation + 2 ) % 3 ] ) )
	{
	    if ( last ) glColor3f( 1., 0.5, 0. );
	    else glColor3f( 0., 0.5, 1. );
            glBegin(GL_QUADS);
	        glVertex2i( (int)c.getParticularValue( _sliceOrientation ) - 3, (int)c.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) - 3 );
	        glVertex2i( (int)c.getParticularValue( _sliceOrientation ) + 3, (int)c.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) - 3 );
	        glVertex2i( (int)c.getParticularValue( _sliceOrientation ) + 3, (int)c.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) + 3 );
	        glVertex2i( (int)c.getParticularValue( _sliceOrientation ) - 3, (int)c.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) + 3 );
	    glEnd();
	    std::ostringstream area;
	    area << a;
	    setTextPosition( (int)c.getParticularValue( _sliceOrientation ) - 5, (int)c.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) + 5 );
	    setTextCoords( (int)c.getParticularValue( _sliceOrientation ) - 5, (int)c.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) + 5 );
	    drawText( area.str().c_str() );
	    unsetTextCoords();
	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
	    if ( last ) glColor3f( 1., 0., 0. );
	    else glColor3f( 0., 0., 1. );
	}
    }
    std::list< Selection::m4dPoint<double> >::iterator it, tmp;
    
    // draw all the other segments and points
    for ( it = s.shapeElements().begin(); it != s.shapeElements().end(); ++it )
    {
    	tmp = it;
	++tmp;

	// check if at least one enpoint of the given segment lies on the current slice
	if ( &(*it) != &(s.shapeElements().back()) &&
	   ( (int)( it->getParticularValue( ( _sliceOrientation + 2 ) % 3 ) / _extents[ ( _sliceOrientation + 2 ) % 3 ] ) == sliceNum ||
	     (int)( tmp->getParticularValue( ( _sliceOrientation + 2 ) % 3 ) / _extents[ ( _sliceOrientation + 2 ) % 3 ] ) == sliceNum ) )
	{
	    glBegin(GL_LINES);
		glVertex2i(  (int)it->getParticularValue( _sliceOrientation ),  (int)it->getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
		glVertex2i( (int)tmp->getParticularValue( _sliceOrientation ), (int)tmp->getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
	    glEnd();
            if ( _printShapeData )
	    {
	        if ( last ) glColor3f( 1., 1., 0. );
                else glColor3f( 0., 1., 1. );
	        Selection::m4dPoint< double > mid = Selection::m4dPoint< double >::midpoint( *it, *tmp );
	        std::ostringstream dist;
	        dist << Selection::m4dPoint< double >::distance( *it, *tmp );
		setTextPosition( (int)mid.getParticularValue( _sliceOrientation ), (int)mid.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
		setTextCoords( (int)mid.getParticularValue( _sliceOrientation ), (int)mid.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) );
	        drawText( dist.str().c_str() );
		unsetTextCoords();
	        glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
                if ( last ) glColor3f( 1., 0., 0. );
                else glColor3f( 0., 0., 1. );
	    }
	}

	// check if the current point lies on the current slice
        if  ( (int)( it->getParticularValue( ( _sliceOrientation + 2 ) % 3 ) / _extents[ ( _sliceOrientation + 2 ) % 3 ] )  == sliceNum )
        {
	    if ( last && &(*it) == &(s.shapeElements().back()) ) glColor3f( 1., 0., 1. );
            glBegin(GL_QUADS);
	        glVertex2i( (int)it->getParticularValue( _sliceOrientation ) - 3, (int)it->getParticularValue( ( _sliceOrientation + 1 ) % 3 ) - 3 );
	        glVertex2i( (int)it->getParticularValue( _sliceOrientation ) + 3, (int)it->getParticularValue( ( _sliceOrientation + 1 ) % 3 ) - 3 );
	        glVertex2i( (int)it->getParticularValue( _sliceOrientation ) + 3, (int)it->getParticularValue( ( _sliceOrientation + 1 ) % 3 ) + 3 );
	        glVertex2i( (int)it->getParticularValue( _sliceOrientation ) - 3, (int)it->getParticularValue( ( _sliceOrientation + 1 ) % 3 ) + 3 );
	    glEnd();
        }
    }
}

void
m4dGUISliceViewerWidget::textDrawer( int xpos, int ypos, const char* text )
{
    setTextPosition( xpos, ypos );
    setTextCoords( xpos, ypos );
    drawText( text );
    unsetTextCoords();
    glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 );
}

void
m4dGUISliceViewerWidget::drawData( double zoomRate, QPoint offset, int sliceNum )
{
    glPushMatrix();
    glLoadIdentity();
    glColor3f( 1., 1., 1. );
    std::list< std::string >::iterator it;
    double w, h;
    calculateWidthHeight( w, h );
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
    std::ostringstream snum, contrast, brightness, resolution, zoom;
    std::string sor;
    snum << sliceNum + 1 << " / " << ( _maximum[ ( _sliceOrientation + 2 ) % 3 ] - _minimum[ ( _sliceOrientation + 2 ) % 3 ] );
    resolution << w << "mm x " << h << "mm";
    zoom << zoomRate << "x";
    contrast << "Contrast : " << _contrastRate;
    brightness << "Brightness : " << _brightnessRate;
    switch ( _sliceOrientation )
    {
        case xy:
	sor = "xy";
	break;

	case yz:
	sor = "yz";
	break;

	case zx:
	sor = "zx";
	break;
    }
    if ( i - o_y > 2 * FONT_HEIGHT && (int)( snum.str().length() * FONT_WIDTH ) < w_o && (int)( resolution.str().length() * FONT_WIDTH ) < w_o &&
         (int)( zoom.str().length() * FONT_WIDTH ) < w_o && (int)( contrast.str().length() * FONT_WIDTH ) < w_o && (int)( brightness.str().length() * FONT_WIDTH ) < w_o )
    {
        textDrawer( o_x + w_o / 2 - FONT_WIDTH, o_y, sor.c_str() );
        textDrawer( o_x + w_o / 2 - FONT_WIDTH * snum.str().length() / 2, o_y + FONT_HEIGHT, snum.str().c_str() );
        textDrawer( o_x + w_o / 4 - FONT_WIDTH * zoom.str().length() / 2, o_y, zoom.str().c_str() );
        textDrawer( o_x + w_o / 4 - FONT_WIDTH * resolution.str().length() / 2, o_y + FONT_HEIGHT, resolution.str().c_str() );
        textDrawer( o_x + 3 * w_o / 4 - FONT_WIDTH * brightness.str().length() / 2, o_y, brightness.str().c_str() );
        textDrawer( o_x + 3 * w_o / 4 - FONT_WIDTH * contrast.str().length() / 2, o_y + FONT_HEIGHT, contrast.str().c_str() );
    }
    for ( it = _leftSideData.begin(); it != _leftSideData.end() && i >= o_y; ++it, i -= FONT_HEIGHT )
    {
        if ( (int)(it->length() * FONT_WIDTH) < w_o )
	{
            setTextPosition( o_x, i );
            setTextCoords( o_x + it->length() * FONT_WIDTH, i + FONT_HEIGHT );
            drawText( it->c_str() );
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
        if ( (int)(it->length() * FONT_WIDTH) < w_o )
	{
	    o_x = w_o - it->length() * FONT_WIDTH;
	    if ( !_oneSliceMode ) o_x += offset.x();
            setTextPosition( o_x, i );
            setTextCoords( o_x + it->length() * FONT_WIDTH, i + FONT_HEIGHT );
            drawText( it->c_str() );
            unsetTextCoords();
	    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
        }
    }
    glPopMatrix();
}

void
m4dGUISliceViewerWidget::drawPicked()
{
    int x, y;
    double w, h;
    calculateWidthHeight( w, h );
    if ( _flipH < 0 ) x = - ( _pickedPosition.x() - (int)w);
    else x = _pickedPosition.x();
    if ( _flipV < 0 ) y = - ( _pickedPosition.y() - (int)h);
    else y = _pickedPosition.y();
    glColor3f( 1., 1., 1. );
    setTextPosition( x, y );
    setTextCoords( x, y );
    std::ostringstream pick;
    pick << _colorPicked;
    drawText( pick.str().c_str() );
    unsetTextCoords();
    glPixelStorei( GL_UNPACK_ROW_LENGTH,  0 );
}

void
m4dGUISliceViewerWidget::ImagePositionSelectionCaller( int x, int y, SelectMethods f )
{
    double w, h;
    QPoint offset;
    calculateWidthHeight( w, h );
    offset.setX( (int)floor( (double)_offset.x() - ( _zoomRate - (double)width()/w ) * 0.5 * w ) );
    offset.setY( (int)floor( (double)_offset.y() - ( _zoomRate - (double)height()/h ) * 0.5 * h ) );
    double coords[3];
    if ( _oneSliceMode )
    {
        coords[ _sliceOrientation             ] = ( ( x - offset.x() ) / _zoomRate );
	coords[ ( _sliceOrientation + 1 ) % 3 ] = ( ( this->height() - y - offset.y() ) / _zoomRate );
	coords[ ( _sliceOrientation + 2 ) % 3 ] = ( _sliceNum * _extents[ ( _sliceOrientation + 2 ) % 3 ] );
    }
    else
    {
        double xgap = 0, ygap = 0;
	double zoomRate = 1.;
	if ( ( (double)width() / (double)_slicesPerRow ) / w < ( (double)height() / (double)_slicesPerColumn ) / h )
	{
	    ygap = ( ( (double)height() / (double)_slicesPerColumn ) - h * ( (double)width() / (double)_slicesPerRow ) / w ) / 2.;
	    zoomRate = ( (double)width() / (double)_slicesPerRow ) / w;
	}
	else
	{
	    xgap = ( ( (double)width() / (double)_slicesPerRow ) - w * ( (double)height() / (double)_slicesPerColumn ) / h ) / 2.;
	    zoomRate = ( (double)height() / (double)_slicesPerColumn ) / h;
	}
        coords[ _sliceOrientation             ] = (double)( x % (int)( 2 * xgap + zoomRate * w ) - xgap ) / zoomRate;
	coords[ ( _sliceOrientation + 1 ) % 3 ] = (double)( ( height() - y ) % (int)( 2 * ygap + zoomRate * h ) - ygap ) / zoomRate;
	coords[ ( _sliceOrientation + 2 ) % 3 ] = ( ( _sliceNum + x / (int)( (double)width() / _slicesPerRow ) + _slicesPerRow * (int)( ( height() - y ) / ( (double)height() / _slicesPerColumn ) ) ) * _extents[ ( _sliceOrientation + 2 ) % 3 ] );
    }
    (this->*f)( coords[0], coords[1], coords[2] );
}

void
m4dGUISliceViewerWidget::calculateOptimalZoomRate()
{
    if ( !_ready )
    {
        setParameters();
	if ( !_ready ) return;
    }
    if ( _inPort->IsPlugged() )
    {
        double w, h;
    	calculateWidthHeight( w, h );
        if ( (double)width() / w < (double)height() / h ) _zoomRate = (double)width() / w;
	else _zoomRate = (double)height() / h;
    }
}

void
m4dGUISliceViewerWidget::resizeGL(int winW, int winH)
{
    glViewport(0, 0, width(), height());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (double)winW, 0.0, (double)winH, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    calculateOptimalZoomRate();
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
    if ( _eventHandler )
    {
        _eventHandler->mousePressEvent(event);
    	updateGL();
        return;
    }
    if ( !_inPort->IsPlugged() ) return;
    if ( !_ready )
    {
        setParameters();
	if ( !_ready ) return;
    }
    _lastPos = event->pos();
    if ( ( event->buttons() & Qt::LeftButton ) && _selectMode[ left ] )
    	ImagePositionSelectionCaller( event->x(), event->y(), _selectMethods[ left ] );
    else if ( event->buttons() & Qt::RightButton && _selectMode[ right ] )
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
    if ( _eventHandler )
    {
        _eventHandler->mouseReleaseEvent(event);
    	updateGL();
        return;
    }
}

void
m4dGUISliceViewerWidget::mouseMoveEvent(QMouseEvent *event)
{
    if ( _eventHandler )
    {
        _eventHandler->mouseMoveEvent(event);
    	updateGL();
        return;
    }
    if ( !_inPort->IsPlugged() ) return;

    if ( _lastPos.x() == -1 || _lastPos.y() == -1 ) return;
    

    int dx = event->x() - _lastPos.x();
    int dy = event->y() - _lastPos.y();

    if ( ( event->buttons() & Qt::LeftButton ) && _buttonMode[ left ] )
    {
        (this->*_buttonMethods[ left ])( dx, -dy );
    }
    else if ( ( event->buttons() & Qt::RightButton ) && _buttonMode[ right ] )
    {
        (this->*_buttonMethods[ right ])( dx, -dy );
    }
    _lastPos = event->pos();
    if ( _colorPicker )
        ImagePositionSelectionCaller( event->x(), event->y(), &M4D::Viewer::m4dGUISliceViewerWidget::colorPicker );

    updateGL();

}

void
m4dGUISliceViewerWidget::mouseDoubleClickEvent ( QMouseEvent * event )
{
    if ( _eventHandler ) _eventHandler->mouseDoubleClickEvent(event);
    else QGLWidget::mouseDoubleClickEvent(event);
    updateGL();
}

void
m4dGUISliceViewerWidget::keyPressEvent ( QKeyEvent * event )
{
    if ( _eventHandler ) _eventHandler->keyPressEvent(event);
    else QGLWidget::keyPressEvent(event);
    updateGL();
}

void
m4dGUISliceViewerWidget::keyReleaseEvent ( QKeyEvent * event )
{
    if ( _eventHandler ) _eventHandler->keyReleaseEvent(event);
    else QGLWidget::keyReleaseEvent(event);
    updateGL();
}

void
m4dGUISliceViewerWidget::wheelEvent(QWheelEvent *event)
{
    if ( _eventHandler )
    {
        _eventHandler->wheelEvent(event);
    	updateGL();
        return;
    }
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
m4dGUISliceViewerWidget::zoomImage( int dummy, int amount )
{
    _zoomRate += 0.001 * amount;
    if ( _zoomRate < 0. ) _zoomRate = 0.;
    emit signalZoom( _index, amount );
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
    unsigned typeSize;
    NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( _imageID, typeSize = sizeof( TTYPE ) );
    typeSize = (unsigned)std::pow( (double)256, (double)(typeSize - 1.5) );
    _brightnessRate += amountB * typeSize;
    _contrastRate += amountC * typeSize;
    emit signalAdjustContrastBrightness( _index, amountC, amountB );
}

void
m4dGUISliceViewerWidget::newPoint( double x, double y, double z )
{
    if ( !_inPort->IsPlugged() ) return;
    if ( _shapes.empty() ) newShape( x, y, z );
    else
    {
    	double w, h;
	calculateWidthHeight( w, h );
	double coords[3];
	coords[0] = x;
	coords[1] = y;
	coords[2] = z;
        resolveFlips( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] );
        if ( checkOutOfBounds( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] ) ) return;
        if ( !_shapes.back().shapeElements().empty() &&
	     abs( (int)( coords[0] - _shapes.back().shapeElements().front().getParticularValue( 0 ) ) ) < MINIMUM_SELECT_DISTANCE &&
             abs( (int)(coords[1] - _shapes.back().shapeElements().front().getParticularValue( 1 ) ) ) < MINIMUM_SELECT_DISTANCE &&
             abs( (int)(coords[2] - _shapes.back().shapeElements().front().getParticularValue( 2 ) ) ) < MINIMUM_SELECT_DISTANCE ) _shapes.back().closeShape();
	else
	{
            Selection::m4dPoint<double> p( coords[0], coords[1], coords[2] );
            _shapes.back().addPoint( p );
	}
        emit signalNewPoint( _index, coords[0], coords[1], coords[2] );
    }
}

void
m4dGUISliceViewerWidget::newShape( double x, double y, double z )
{
    double w, h;
    calculateWidthHeight( w, h );
    if ( !_inPort->IsPlugged() ) return;
    double coords[3];
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
    if ( checkOutOfBounds( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] ) ) return;

    Selection::m4dShape<double> s( 3, _sliceOrientation );
    _shapes.push_back( s );
    newPoint( x, y, z );
    resolveFlips( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] );
    emit signalNewShape( _index, coords[0], coords[1], coords[2] );
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
m4dGUISliceViewerWidget::colorPicker( double x, double y, double z )
{
    double coords[3];
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
    _pickedPosition = QPoint( (int)coords[ _sliceOrientation ], (int)coords[ ( _sliceOrientation + 1 ) % 3 ] );
    if ( !_inPort->IsPlugged() ) return;
    int64 result = 0;
    resolveFlips( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] );
    if ( checkOutOfBounds( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] ) ) return;
    if ( !_ready ) setParameters();
    if ( !_ready ) return;
    coords[ 0 ] = coords[ 0 ] / _extents[ 0 ];
    coords[ 1 ] = coords[ 1 ] / _extents[ 1 ];
    coords[ 2 ] = coords[ 2 ] / _extents[ 2 ];
    try
    {
	if ( _inPort->TryLockDataset() )
	{
            try
	    {
		if ( _inPort->GetDatasetTyped().GetDimension() == 3 )
		{
		    NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
		        _imageID, result = Imaging::Image< TTYPE, 3 >::CastAImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >( (int)coords[0], (int)coords[1], (int)coords[2] ) ) );
		}
	        else if ( _inPort->GetDatasetTyped().GetDimension() == 2 )
	        {
		    NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
		        _imageID, result = Imaging::Image< TTYPE, 2 >::CastAImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >( (int)coords[0], (int)coords[1] ) ) );
	        }
	        else
		{
		    _ready = false;
	            result = 0;
		}
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
    if ( !_ready ) return;
    _colorPicker = true;
    _colorPicked = result;
    _slicePicked = (int)coords[ ( _sliceOrientation + 2 ) % 3 ];
    emit signalColorPicker( _index, result );
}

void
m4dGUISliceViewerWidget::pointPicker( double x, double y, double z )
{
    double coords[3];
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;

    _pickedPosition = QPoint( (int)coords[ _sliceOrientation ], (int)coords[ ( _sliceOrientation + 1 ) % 3 ] );
    
    if ( !_inPort->IsPlugged() ) return;

    resolveFlips( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] );
    
    if ( checkOutOfBounds( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] ) ) return;
    
    if ( !_ready ) setParameters();
    if ( !_ready ) return;
    
    coords[ 0 ] = coords[ 0 ] / _extents[ 0 ];
    coords[ 1 ] = coords[ 1 ] / _extents[ 1 ];
    coords[ 2 ] = coords[ 2 ] / _extents[ 2 ];
    
    try
    {
	    if ( _inPort->TryLockDataset() )
	    {
        try
	      {
		      if ( _inPort->GetDatasetTyped().GetDimension() != 3 && _inPort->GetDatasetTyped().GetDimension() != 2 )
		      {
		        _ready = false;
          }
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
    
    if ( !_ready ) return;

    _slicePicked = (int)coords[ ( _sliceOrientation + 2 ) % 3 ];
    emit signalDataPointPicker( _index, (int)coords[0], (int)coords[1], (int)coords[2] );
}

bool
m4dGUISliceViewerWidget::checkOutOfBounds( double x, double y )
{
    double w, h;
    calculateWidthHeight( w, h );
    if ( x < 0 || y < 0 || x >= w || y >= h ) return true;
    return false;
}

void
m4dGUISliceViewerWidget::resolveFlips( double& x, double& y )
{
    double w, h;
    calculateWidthHeight( w, h );
    if ( _flipH < 0 ) x = - ( x - w);
    if ( _flipV < 0 ) y = - ( y - h);
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
m4dGUISliceViewerWidget::slotAddLeftSideData( std::string data )
{
    addLeftSideData( data );
}

void
m4dGUISliceViewerWidget::slotAddRightSideData( std::string data )
{
    addRightSideData( data );
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
m4dGUISliceViewerWidget::slotNewPoint( double x, double y, double z )
{
    newPoint( x, y, z );
}

void
m4dGUISliceViewerWidget::slotNewShape( double x, double y, double z )
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
m4dGUISliceViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn )
{
    setMoreSliceMode( slicesPerRow, slicesPerColumn );
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
    if ( _dimension < 3 ) return;
    
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
    calculateOptimalZoomRate();
    
    // modify shape orientations for centroid and area calculation and displaying
    for ( std::list< Selection::m4dShape<double> >::iterator it = _shapes.begin(); it != _shapes.end(); ++it ) it->setOrientation( _sliceOrientation );
    updateGL();
    emit signalToggleSliceOrientation( _index );
}

void
m4dGUISliceViewerWidget::slotColorPicker( double x, double y, double z )
{
    colorPicker( x, y, z );
}

void
m4dGUISliceViewerWidget::updateViewer()
{
	updateGL();
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
            calculateOptimalZoomRate();
            updateGL();
	    if ( ! _ready ) return;
	    if ( msgID != Imaging::PMI_FILTER_UPDATED )
	    {
		unsigned typeSize;
		bool unsgn;
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( _imageID, { typeSize = sizeof( TTYPE ); unsgn = ( typeid( TTYPE ) == typeid( uint8 ) || typeid( TTYPE ) == typeid( uint16 ) || typeid( TTYPE ) == typeid( uint32 ) || typeid( TTYPE ) == typeid( uint64 ) ); } );
		typeSize = (unsigned)std::pow( (double)256, (double)(typeSize - 1) );
                _brightnessRate = 12 * typeSize;
                _contrastRate = 20 * typeSize;
		if ( unsgn ) _brightnessRate *= 2;
	    }
	}
	break;
	
	default:
	break;
    }
}

} /*namespace Viewer*/
} /*namespace M4D*/
