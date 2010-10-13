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
bool TFSliceViewerTexturePreparer< ElementType >::prepare(
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
void TFSliceViewerTexturePreparer< ElementType >::setTransferFunction(TFAbstractFunction &transferFunction){

	tfUsed_ = TFApplicator::apply(
		&transferFunction,
		currentTransferFunction_.begin(),
		currentTransferFunction_.size(),
		(TFSize)TypeTraits<ElementType>::Max);
}

template< typename ElementType >
const TFHistogram& TFSliceViewerTexturePreparer< ElementType >::getHistogram(){

	//TODO
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
			if(!texturePreparer_ || currentImageID_ != _imageID)
			{
				if(texturePreparer_) delete texturePreparer_;
				texturePreparer_ = new TFSliceViewerTexturePreparer<TTYPE>();
				currentImageID_ = _imageID;
				setTexturePreparerToCustom(texturePreparer_);
			}
			(dynamic_cast<TFSliceViewerTexturePreparer<TTYPE>*>(texturePreparer_))->setTransferFunction(transferFunction);
			updateGL();
		} 
	);
}

void TFSliceViewerWidget::histogram_request(){

	if(_imageID == -1)
	{		
		QMessageBox::critical(this, QObject::tr("Transfer Functions"),
						  QObject::tr("No data loaded!"));
		return;
	}

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
		_imageID, 
		{		
			if(!texturePreparer_ || currentImageID_ != _imageID)
			{
				if(texturePreparer_) delete texturePreparer_;
				texturePreparer_ = new TFSliceViewerTexturePreparer<TTYPE>();
				currentImageID_ = _imageID;
				setTexturePreparerToCustom(texturePreparer_);
			}
			emit Histogram((dynamic_cast<TFSliceViewerTexturePreparer<TTYPE>*>(texturePreparer_))->getHistogram());
		}
	);
}


} /*namespace Viewer*/
} /*namespace M4D*/
