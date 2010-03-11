/**
 *  @ingroup gui
 *  @file m4dGUISliceViewerWidget2.cpp
 *  @brief some brief
 */

#include <TF/m4dTFSliceViewerWidget.h>
#include "GUI/widgets/components/RGBSliceViewerTexturePreparer.h"

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
bool TFSimpleSliceViewerTexturePreparer< ElementType >::prepare(
	const Imaging::InputPortList& inputPorts,
	uint32& width,
	uint32& height,
	GLint brightnessRate,
	GLint contrastRate,
	SliceOrientation so,
	uint32 slice,
	unsigned& dimension){

	// get the input datasets
	ElementType** pixel = getDatasetArrays( inputPorts, 1, width, height, so, slice, dimension );

	if ( ! *pixel )
	{
	    delete[] pixel;
	    return false;
	}

	// equalize the first input array
	adjustArrayContrastBrightness( *pixel, width, height, brightnessRate, contrastRate );
	ElementType* pix = *pixel;
	for(int i = 0; i < width*height; ++i)
	{
		*pix = TypeTraits<ElementType>::Max;
		++pix;
	}
	/*
	if(currentTransferFunction != NULL)
	{
		currentTransferFunction->adjustByTransferFunction(
			(int*)(*pixel),
			TypeTraits<ElementType>::Min,
			TypeTraits<ElementType>::Max,
			width, height,
			brightnessRate, contrastRate);
	}
	*/

	// prepare texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
                      GL_LUMINANCE, this->oglType(), *pixel );

	// free temporary allocated space
	delete[] *pixel;

	delete[] pixel;

    return true;
}
/*
template< typename ElementType >
bool TFSimpleSliceViewerTexturePreparer< ElementType >::adjustByTransferFunction(
	ElementType* pixel,
	uint32 &width,
	uint32 &height,
	GLint brightnessRate,
	GLint contrastRate){

	if ( ! pixel )
	{
	    return false;
	}

	uint32 i, j;

	for ( i = 0; i < height; ++i )
	{
		for ( j = 0; j < width; ++j )
		{
			ElementType pixelValue = (pixel[ i * width + j ]/contrastRate)-brightnessRate;
			int newValue = currentTransferFunction->getValue(pixelValue, TypeTraits<ElementType>::Min, TypeTraits<ElementType>::Max);
			pixel[ i * width + j ] = newValue;
		}
	}

	return true;
}*/

template< typename ElementType >
void TFSimpleSliceViewerTexturePreparer< ElementType >::setTransferFunction(TFAFunction *transferFunction){
	currentTransferFunction = transferFunction;
}



void m4dTFSliceViewerWidget::AdjustByTransferFunction(TFAFunction *transferFunction){	

    uint32 height, width;
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
				_imageID, 
				{					
					TFSimpleSliceViewerTexturePreparer<TTYPE> texturePreparer;
					texturePreparer.setTransferFunction(transferFunction);
					_ready = texturePreparer.prepare( 
						this->InputPort(), 
						width, 
						height, 
						_brightnessRate, 
						_contrastRate, 
						_sliceOrientation, 
						_sliceNum,//sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], 
						_dimension ); 
				} 
		);
}

void m4dTFSliceViewerWidget::drawSlice( int sliceNum, double zoomRate, QPoint offset ){

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
    /*
    // prepare texture
    if ( texturePreparerType == custom || texturePreparerType == rgb || texturePreparerType == tf )
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

	case tf:*/
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
				_imageID, 
				{					
					TFSimpleSliceViewerTexturePreparer<TTYPE> texturePreparer;
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
		);/*
    break;

	case custom:
	_ready = customTexturePreparer->prepare( this->InputPort(), width, height, _brightnessRate, _contrastRate, _sliceOrientation, sliceNum - _minimum[ ( _sliceOrientation + 2 ) % 3 ], _dimension );
	break;
    }*/

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

} /*namespace Viewer*/
} /*namespace M4D*/
