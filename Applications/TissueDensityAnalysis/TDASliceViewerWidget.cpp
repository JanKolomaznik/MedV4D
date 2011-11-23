/**
 *  @ingroup TDA
 *	@author Milan Lepik
 *  @file TDASliceViewerWidget.cpp
 */

#include "TDASliceViewerWidget.h"
#include "GUI/widgets/components/RGBSliceViewerTexturePreparer.h"
#include "TDASliceViewerTexturePreparer.cpp"

using namespace M4D;
using namespace M4D::Viewer;


TDASliceViewerWidget::TDASliceViewerWidget( unsigned index, QWidget *parent)
{
	//TODO: smazat port list
    _index = index;
    _inPort = new Imaging::InputPortTyped<Imaging::AImage>();
	_inMaskPort = new Imaging::InputPortTyped<Imaging::AImage>();
    resetParameters();
    _inputPorts.AppendPort( _inPort );
	_inputPorts.AppendPort( _inMaskPort );
}

TDASliceViewerWidget::TDASliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent)
    : m4dGUISliceViewerWidget(conn, index, parent)
{
}

void
TDASliceViewerWidget::setMaskConnection(Imaging::ConnectionInterface* connMask)
{
  connMask->ConnectConsumer( *_inMaskPort );
}

void 
TDASliceViewerWidget::specialStateButtonMethodLeft( int amountA, int amountB )
{
	if( _specialState ) {
		//_specialState->ButtonMethodLeft( amountA, amountB, _zoomRate );
		//emit signalSphereRadius( amountA, amountB, _zoomRate);
	}
}

void 
TDASliceViewerWidget::specialStateSelectMethodLeft( double x, double y, double z )
{
	if( _specialState ) {
		resolveFlips( x, y );
		int sliceNum = z/_extents[2];
		emit signalSphereCenter(x,  y,  (double)sliceNum);
		//_specialState->SelectMethodLeft( x + _extents[0]*_minimum[0], y + _extents[1]*_minimum[1], sliceNum, _zoomRate );
	}
}

void 
TDASliceViewerWidget::drawHUD( int sliceNum, double zoomRate, QPoint offset )
{
	PredecessorType::drawHUD( sliceNum, zoomRate, offset );
}

void 
TDASliceViewerWidget::setButtonHandler( ButtonHandler hnd, MouseButton btn )
{
	if( hnd != specialState ) {
		PredecessorType::setButtonHandler( hnd, btn );
		return;
	}

	_selectMethods[ left ] = (SelectMethods)&TDASliceViewerWidget::specialStateSelectMethodLeft;
	_selectMode[ left ] = true;

	_buttonMethods[ left ] = (ButtonMethods)&TDASliceViewerWidget::specialStateButtonMethodLeft;
	_buttonMode[ left ] = true;

	if ( _ready ) updateGL();
		emit signalSetButtonHandler( _index, hnd, btn );
}

void
TDASliceViewerWidget::sphereCenter( double x, double y, double z )
{
   /* double w, h;
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
    resolveFlips( coords[ _sliceOrientation ], coords[ ( _sliceOrientation + 1 ) % 3 ] );*/
    //emit signalSphereCenter( _index, coords[0], coords[1], coords[2] );
}


void 
TDASliceViewerWidget::slotSetSpecialStateSelectMethodLeft()
{
	setButtonHandler( specialState, left );
	TDASliceViewerSpecialStateOperator *sState = new TDASliceViewerSpecialStateOperator();
	_specialState = TDASliceViewerSpecialStateOperatorPtr( sState );
}


void
TDASliceViewerWidget::drawSlice( int sliceNum, double zoomRate, QPoint offset )
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
    	    _imageID, 
			{ 
				RGBSliceViewerTexturePreparer<TTYPE> texturePreparer; 
				_ready = texturePreparer.prepare( 
						this->InputPort(), 
						width, 
						height, 
						_brightnessRate, 
						_contrastRate, 
						_sliceOrientation, 
						sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], 
						_dimension 
						); 
			}
		);
	break;

	case custom:
	_ready = customTexturePreparer->prepare( this->InputPort(), width, height, _brightnessRate, _contrastRate, _sliceOrientation, sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], _dimension );
	break;

	default:
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
				_imageID, 
				{
					TDASliceViewerTexturePreparer<TTYPE> texturePreparer;
					_ready = texturePreparer.prepare( 
						this->InputPort(), 
						width, 
						height, 
						_brightnessRate, 
						_contrastRate, 
						_sliceOrientation, 
						sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], 
						_dimension ); 
				} 
		);
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