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

	_wholeExtent[0] = 0;  _wholeExtent[1] = 0;
  	_wholeExtent[2] = 0;  _wholeExtent[3] = 0;
	_wholeExtent[4] = 0;  _wholeExtent[5] = 0;
	Modified();

	//Test version

	//SetImageData( Images::ImageFactory::CreateEmptyImage3D< unsigned char >( 30, 30, 30 ) );
}

m4dImageDataSource::~m4dImageDataSource()
{

}

void
m4dImageDataSource::SetImageData( Images::AbstractImage::APtr imageData )
{
	D_PRINT( LogDelimiter( '*' ) );
	D_PRINT( "-- Entering m4dImageDataSource::SetImageData()." );
	
	//TODO - check dimension	
	size_t imageDimension = 3;

	if( !imageData ) {
		D_PRINT( "---- Obtained invalid image pointer." );
		//Setting to NULL
		_imageData = Images::AbstractImage::APtr();

		for( size_t dim = 0; dim < imageDimension; ++dim ) {
			_wholeExtent[2*dim]		= 0;  
			_wholeExtent[2*dim + 1] = 0;
		}
	} else {
		D_PRINT( "---- Obtained valid image pointer :" );
		_imageData = imageData;

		for( size_t dim = 0; dim < imageDimension; ++dim ) {
			D_PRINT( "-------- Size in dimension " << dim << " = " 
				<< _imageData->GetDimensionInfo( dim ).size );

			_wholeExtent[2*dim]		= 0;  
			_wholeExtent[2*dim + 1] = _imageData->GetDimensionInfo( dim ).size-1;  		
		}
	}
	Modified();
	Update();

	D_PRINT( "-- Leaving m4dImageDataSource::SetImageData()." );
	D_PRINT( LogDelimiter( '+' ) );
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
	D_PRINT( LogDelimiter( '*' ) );
	D_PRINT( "-- Entering m4dImageDataSource::RequestData()." );

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
		D_PRINT( "-- Leaving m4dImageDataSource::RequestData(). We didn't have data for conversion!" );
		D_PRINT( LogDelimiter( '+' ) );
		return 1;
	}
	D_PRINT( "---- Filling requested VTK image dataset." );
	//Fill data set
	FillVTKImageFromM4DImage( output, _imageData );

	D_PRINT( "-- Leaving m4dImageDataSource::RequestData(). Everything OK" );
	D_PRINT( LogDelimiter( '+' ) );

	return 1;
}

} /*namespace vtkIntegration*/
} /*namespace M4D*/

