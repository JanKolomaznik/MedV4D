#include "aggregationFilterForVTK.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"

vtkStandardNewMacro(aggregationFilterForVtk);
vtkCxxRevisionMacro(aggregationFilterForVtk, "$Revision: 1.70 $");

aggregationFilterForVtk::aggregationFilterForVtk()
{
	aggregationData.clear();
}

void aggregationFilterForVtk::SetAggregationPoint(unsigned short downRangeBorder, unsigned short upRangeBorder, unsigned short valueToSet)
{
	aggregationUnit au(downRangeBorder, upRangeBorder, valueToSet);
	aggregationData.push_back(au);
}

void aggregationFilterForVtk::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

int aggregationFilterForVtk::RequestData(vtkInformation *vtkNotUsed(request), vtkInformationVector **inputVector, vtkInformationVector *outputVector)
{
	vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	vtkImageData *input = vtkImageData::SafeDownCast(
		inInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkImageData *output = vtkImageData::SafeDownCast(
		outInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
	image->DeepCopy(input);

	if (!aggregationData.empty())
	{
		int extents[6];
		image->GetExtent(extents);

		for (int i = extents[0]; i < extents[1]; ++i)
		{
			for (int j = extents[2]; j < extents[3]; ++j)
			{
				for (int k = extents[4]; k < extents[5]; ++k)
				{
					for (int c = 0; c < image->GetNumberOfScalarComponents(); ++c)
					{
						for(std::size_t a = 0; a < aggregationData.size(); ++a)
						{
							unsigned short pointValue = (unsigned short)image->GetScalarComponentAsDouble(i, j, k, c);
							if ((pointValue >= aggregationData[a].downRangeBorder) && (pointValue <= aggregationData[a].upRangeBorder))
							{
								image->SetScalarComponentFromDouble(i, j, k, c, aggregationData[a].valueToSet);
							}
						}
					}
				}
			}
		}
	}

	output->ShallowCopy(image);

	return 1; // May be bug - not documented what should this method return
}
