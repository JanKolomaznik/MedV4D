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

vtkCxxRevisionMacro(m4dImageDataSource, "$Revision: 1.00 $");

m4dImageDataSource* 
m4dImageDataSource::New()
{
	return new m4dImageDataSource; 
}

m4dImageDataSource::m4dImageDataSource()
{
	this->SetNumberOfInputPorts(0);

	_wholeExtent[0] = 0;  _wholeExtent[1] = 200;
  	_wholeExtent[2] = 0;  _wholeExtent[3] = 200;
	_wholeExtent[4] = 0;  _wholeExtent[5] = 200;
	Modified();
}

m4dImageDataSource::~m4dImageDataSource()
{

}

void
m4dImageDataSource::SetImageData( Images::AbstractImage::APtr imageData )
{
	//TODO Check
	_imageData = imageData;

	_wholeExtent[0] = 0;  _wholeExtent[1] = _imageData->GetDimensionInfo( 0 ).size-1;
  	_wholeExtent[2] = 0;  _wholeExtent[3] = _imageData->GetDimensionInfo( 1 ).size-1;
	_wholeExtent[4] = 0;  _wholeExtent[5] = _imageData->GetDimensionInfo( 2 ).size-1;

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

	outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
               this->_wholeExtent,6);

	vtkDataObject::SetPointDataActiveScalarInfo(
			outInfo, 
			VTK_UNSIGNED_SHORT,
			1
			);
	/*vtkDataObject::SetPointDataActiveScalarInfo(
			outInfo, 
			ConvertNumericTypeIDToVTKScalarType( _imageData->GetElementTypeID() ), 
			1
			);*/
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


	vtkIdType IncX, IncY, IncZ;
	
	output->GetIncrements(IncX, IncY, IncZ);
	unsigned short *iPtr = (unsigned short*)output->GetScalarPointer();

	for(size_t idxZ = 0; idxZ < _wholeExtent[5]; ++idxZ)
	{
		for(size_t idxY = 0; idxY < _wholeExtent[3]; ++idxY)
		{
			for(size_t idxX = 0; idxX < _wholeExtent[1]; ++idxX)
			{
				*iPtr = (idxZ + idxY + idxX)%255;
				++iPtr;
			}
			iPtr += IncY;
		}
		iPtr += IncZ;
	}

	return 1;
}

} /*namespace vtkIntegration*/
} /*namespace M4D*/

