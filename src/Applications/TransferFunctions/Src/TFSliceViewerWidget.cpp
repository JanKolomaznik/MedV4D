/**
 *  @ingroup gui
 *  @file m4dGUISliceViewerWidget2.cpp
 *  @brief some brief
 */

#include "TFSliceViewerWidget.h"
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

	ElementType* pixelValue = *pixel;
	/*
	if(_histSlice != slice)
	{
		double range = TypeTraits<ElementType>::Max - TypeTraits<ElementType>::Min;
		_histogram = std::vector<int>(range, 0);
		for(unsigned i = 0; i < width*height; ++i)
		{
			++_histogram[*pixelValue];
			++pixelValue;
		}
	}
	pixelValue = *pixel;
	*/
	if(_currentTransferFunction.size() > 0)
	{
		for(unsigned i = 0; i < width*height; ++i)
		{
			*pixelValue = _currentTransferFunction[*pixelValue];
			++pixelValue;
		}
	}
	else
	{
		//no TF used
		// equalize the first input array
		adjustArrayContrastBrightness( *pixel, width, height, brightnessRate, contrastRate );
	}

	// prepare texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
                      GL_LUMINANCE, this->oglType(), *pixel );

	// free temporary allocated space
	delete[] *pixel;

	delete[] pixel;

    return true;
}

template< typename ElementType >
void TFSimpleSliceViewerTexturePreparer< ElementType >::setTransferFunction(TFAbstractFunction &transferFunction){

	_currentTransferFunction = adjustByTransferFunction<ElementType>(
		TypeTraits<ElementType>::Min,
		TypeTraits<ElementType>::Max,
		transferFunction);
}

template< typename ElementType >
std::vector<int> TFSimpleSliceViewerTexturePreparer< ElementType >::getHistogram(){
	return _histogram;
}


void TFSliceViewerWidget::adjust_by_transfer_function(TFAbstractFunction &transferFunction){	

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
				_imageID, 
				{									
					if(!texturePreparer)
					{
						texturePreparer = new TFSimpleSliceViewerTexturePreparer<TTYPE>();
						currentImageID = _imageID;
					}
					else if(currentImageID != _imageID)
					{
						delete texturePreparer;
						texturePreparer = new TFSimpleSliceViewerTexturePreparer<TTYPE>();
						currentImageID = _imageID;
					}
					(dynamic_cast<TFSimpleSliceViewerTexturePreparer<TTYPE>*>(texturePreparer))->setTransferFunction(transferFunction);
					
					setTexturePreparerToCustom(texturePreparer);

					updateGL();
				} 
		);
}

} /*namespace Viewer*/
} /*namespace M4D*/
