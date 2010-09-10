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
	if(histSlice_ != slice)
	{
		double range = TypeTraits<ElementType>::Max - TypeTraits<ElementType>::Min;
		histogram_ = std::vector<int>(range, 0);
		for(unsigned i = 0; i < width*height; ++i)
		{
			++histogram_[*pixelValue];
			++pixelValue;
		}
	}
	pixelValue = *pixel;
	*/
	if(tfUsed_)
	{
		for(unsigned i = 0; i < width*height; ++i)
		{
			*pixelValue = currentTransferFunction_[*pixelValue];
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

	tfUsed_ = adjustByTransferFunction<ElementType>(
		&transferFunction,
		&currentTransferFunction_,
		0,
		TypeTraits<ElementType>::Max,
		TypeTraits<ElementType>::Max);
}

template< typename ElementType >
std::vector<int> TFSimpleSliceViewerTexturePreparer< ElementType >::getHistogram(){
	return histogram_;
}


void TFSliceViewerWidget::adjust_by_transfer_function(TFAbstractFunction &transferFunction){	

	if(_imageID == -1)
	{		
		QMessageBox::critical(this, QObject::tr("Transfer Functions"),
						  QObject::tr("No data loaded!"));
		return;
	}

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
				_imageID, 
				{									
					if(!texturePreparer_)
					{
						texturePreparer_ = new TFSimpleSliceViewerTexturePreparer<TTYPE>();
						currentImageID_ = _imageID;
					}
					else if(currentImageID_ != _imageID)
					{
						delete texturePreparer_;
						texturePreparer_ = new TFSimpleSliceViewerTexturePreparer<TTYPE>();
						currentImageID_ = _imageID;
					}
					(dynamic_cast<TFSimpleSliceViewerTexturePreparer<TTYPE>*>(texturePreparer_))->setTransferFunction(transferFunction);
					
					setTexturePreparerToCustom(texturePreparer_);

					updateGL();
				} 
		);
}

} /*namespace Viewer*/
} /*namespace M4D*/
