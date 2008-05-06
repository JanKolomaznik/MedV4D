#include "m4dImageDataSource.h"

#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "Log.h"
#include "Debug.h"
#include "DataConversion.h"
#include "ImageFactory.h"

namespace M4D
{
namespace vtkIntegration
{

vtkCxxRevisionMacro(m4dImageDataSource, "$Revision: 1.00 $");

m4dImageDataSource* 
m4dImageDataSource::New()
{
	return new m4dImageDataSource; 
}

m4dImageDataSource::m4dImageDataSource()
{
	this->SetNumberOfInputPorts(0);

	/*_wholeExtent[0] = 0;  _wholeExtent[1] = 0;
  	_wholeExtent[2] = 0;  _wholeExtent[3] = 0;
	_wholeExtent[4] = 0;  _wholeExtent[5] = 0;
	Modified();*/

	//Test version

	SetImageData( Images::ImageFactory::CreateEmptyImage3D< unsigned char >( 30, 30, 30 ) );
}

m4dImageDataSource::~m4dImageDataSource()
{

}

void
m4dImageDataSource::SetImageData( Images::AbstractImage::APtr imageData )
{
	//TODO Check
	_imageData = imageData;

	if( !_imageData ) {
		_wholeExtent[0] = 0;  _wholeExtent[1] = 0;
  		_wholeExtent[2] = 0;  _wholeExtent[3] = 0;
		_wholeExtent[4] = 0;  _wholeExtent[5] = 0;
	} else {
		_wholeExtent[0] = 0;  _wholeExtent[1] = _imageData->GetDimensionInfo( 0 ).size-1;
  		_wholeExtent[2] = 0;  _wholeExtent[3] = _imageData->GetDimensionInfo( 1 ).size-1;
		_wholeExtent[4] = 0;  _wholeExtent[5] = _imageData->GetDimensionInfo( 2 ).size-1;
	}
	Modified();
}

int 
m4dImageDataSource::RequestInformation (
		vtkInformation 		*vtkNotUsed(request),
		vtkInformationVector	**vtkNotUsed( inputVector ),
		vtkInformationVector	*outputVector
		)
{
	vtkInformation* outInfo = outputVector->GetInformationObject(0);
	if( !_imageData ) {
		vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 1);
		return 1;
	}

	outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
               this->_wholeExtent,6);

	vtkDataObject::SetPointDataActiveScalarInfo(
			outInfo, 
			ConvertNumericTypeIDToVTKScalarType( _imageData->GetElementTypeID() ), 
			1
			);
	return 1;
}	

int 
m4dImageDataSource::RequestData(
		vtkInformation* vtkNotUsed( request ),
		vtkInformationVector** vtkNotUsed(inputVector),
		vtkInformationVector* outputVector
		)
{
	// get the data object
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	vtkImageData *output = vtkImageData::SafeDownCast(
					outInfo->Get(vtkDataObject::DATA_OBJECT())
					);

	// Set the extent of the output and allocate memory.
	output->SetExtent(
			outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT())
			);
	output->AllocateScalars();

	//We don't have data for convesion.
	if( !_imageData ) {
		return 1;
	}

	//Fill data set
	FillVTKImageFromM4DImage( output, _imageData );

	return 1;
}

} /*namespace vtkIntegration*/
} /*namespace M4D*/

