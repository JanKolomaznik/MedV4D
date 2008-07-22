#include "m4dImageDataSource.h"

#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "Log.h"
#include "Debug.h"
#include "DataConversion.h"
#include "Imaging/ImageFactory.h"


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
	_spacing[0] = _spacing[1] = _spacing[2] = 1.0;
	_tmpImageData = NULL;
	Modified();

	//Test version

	//SetImageData( Imaging::ImageFactory::CreateEmptyImage3D< unsigned char >( 30, 30, 30 ) );
}

m4dImageDataSource::~m4dImageDataSource()
{

}

void
m4dImageDataSource::SetImageData( Imaging::AbstractImage::AImagePtr imageData )
{
	D_PRINT( LogDelimiter( '*' ) );
	D_PRINT( "-- Entering m4dImageDataSource::SetImageData()." );
	
	if( !imageData ) {
		D_PRINT( "---- Obtained invalid image pointer." );

	} else {
		D_PRINT( "---- Obtained valid image pointer :" );

		TemporarySetImageData( *(imageData.get()) );
		_imageData = imageData;
	}
	//TODO - check dimension	
	/*size_t imageDimension = 3;

	if( !imageData ) {
		D_PRINT( "---- Obtained invalid image pointer." );
		//Setting to NULL
		_imageData = Imaging::AbstractImage::AImagePtr();
		_tmpImageData = NULL;

		for( size_t dim = 0; dim < imageDimension; ++dim ) {
			_wholeExtent[2*dim]		= 0;  
			_wholeExtent[2*dim + 1] = 0;
			_spacing[dim] = 1.0;
		}
	} else {
		D_PRINT( "---- Obtained valid image pointer :" );
		_imageData = imageData;
		_tmpImageData = imageData.get();
		
		for( size_t dim = 0; dim < imageDimension; ++dim ) {
			const Imaging::DimensionExtents &dimExtents = _imageData->GetDimensionExtents( dim );
			
				D_PRINT( "-------- Size in dimension " << dim << " = " << dimExtents.maximum - dimExtents.minimum );
			
			_wholeExtent[2*dim]	= dimExtents.minimum;  
			_wholeExtent[2*dim + 1] = dimExtents.maximum-1; 
			
			_spacing[dim] = dimExtents.elementExtent;
		}
	}
	Modified();
	Update();*/

	D_PRINT( "-- Leaving m4dImageDataSource::SetImageData()." );
	D_PRINT( LogDelimiter( '+' ) );
}

void
m4dImageDataSource::TemporarySetImageData( const Imaging::AbstractImage & imageData )
{
	
	D_PRINT( LogDelimiter( '*' ) );
	D_PRINT( "-- Entering m4dImageDataSource::TemporarySetImageData()." );
	
	//TODO - check dimension	
	size_t imageDimension = 3;
	_tmpImageData = &imageData;
	_imageData = Imaging::AbstractImage::AImagePtr();

		
	for( size_t dim = 0; dim < imageDimension; ++dim ) {
		const Imaging::DimensionExtents &dimExtents = _tmpImageData->GetDimensionExtents( dim );
		
			D_PRINT( "-------- Size in dimension " << dim << " = " << dimExtents.maximum - dimExtents.minimum );
		
		_wholeExtent[2*dim]	= dimExtents.minimum;  
		_wholeExtent[2*dim + 1] = dimExtents.maximum-1; 
		
		_spacing[dim] = dimExtents.elementExtent;
	}

	Modified();
	Update();

	D_PRINT( "-- Leaving m4dImageDataSource::TemporarySetImageData()." );
	D_PRINT( LogDelimiter( '+' ) );
}

void
m4dImageDataSource::TemporaryUnsetImageData()
{
	_tmpImageData = NULL;
	_imageData = Imaging::AbstractImage::AImagePtr();
}

int 
m4dImageDataSource::RequestInformation (
		vtkInformation 		*vtkNotUsed(request),
		vtkInformationVector	**vtkNotUsed( inputVector ),
		vtkInformationVector	*outputVector
		)
{
	vtkInformation* outInfo = outputVector->GetInformationObject(0);
	if( !_tmpImageData ) {
		vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 1);
		return 1;
	}

	outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
               this->_wholeExtent,6);

	outInfo->Set(vtkDataObject::SPACING(), this->_spacing, 3);

	vtkDataObject::SetPointDataActiveScalarInfo(
			outInfo, 
			ConvertNumericTypeIDToVTKScalarType( _tmpImageData->GetElementTypeID() ), 
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
	//Set voxel size
	output->SetSpacing( _spacing );

	output->AllocateScalars();

	//We don't have data for convesion.
	if( !_tmpImageData ) {
		D_PRINT( "-- Leaving m4dImageDataSource::RequestData(). We didn't have data for conversion!" );
		D_PRINT( LogDelimiter( '+' ) );
		return 1;
	}

	//TODO - check whether OK
	D_PRINT( "---- Setting voxel size to : " << std::endl << "\t\tw = " << 	_tmpImageData->GetDimensionExtents( 0 ).elementExtent <<
			std::endl << "\t\th = " << _tmpImageData->GetDimensionExtents( 1 ).elementExtent <<
			std::endl << "\t\td = " << _tmpImageData->GetDimensionExtents( 2 ).elementExtent );

	D_PRINT( "---- Filling requested VTK image dataset." );
	//Fill data set
	FillVTKImageFromM4DImage( output, *_tmpImageData );

	D_PRINT( "-- Leaving m4dImageDataSource::RequestData(). Everything OK" );
	D_PRINT( LogDelimiter( '+' ) );

	return 1;
}

} /*namespace vtkIntegration*/
} /*namespace M4D*/

