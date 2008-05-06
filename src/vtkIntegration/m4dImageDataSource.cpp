#include "m4dImageDataSource.h"

#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"


#include "DataConversion.h"

namespace M4D
{
namespace vtkIntegration
{

void
m4dImageDataSource::SetImageData( Images::AbstractImage::APtr imageData )
{
	//TODO Check
	_imageData = imageData;

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
		/*outInfo->Set(vtkDataObject::SCALAR_TYPE(),VTK_DOUBLE);
		outInfo->Set(vtkDataObject::SPACING(),spacing,3);
		outInfo->Set(vtkDataObject::ORIGIN(),origin,3)*/
		vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 1);
		return 1;
	}

	vtkDataObject::SetPointDataActiveScalarInfo(
			outInfo, 
			ConvertNumericTypeIDToVTKScalarType( _imageData->GetElementTypeID() ), 
			1//_imageData->GetElementCount()
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

	//TODO Fill data set
	
	return 1;
}

} /*namespace vtkIntegration*/
} /*namespace M4D*/

