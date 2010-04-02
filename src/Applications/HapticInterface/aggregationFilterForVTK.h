#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_aggregation_FILTER_FOR_VTK
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_aggregation_FILTER_FOR_VTK

#include "vtkImageAlgorithm.h"
#include <vector>

class aggregationFilterForVtk : public vtkImageAlgorithm
{
public:
	vtkTypeRevisionMacro(aggregationFilterForVtk, vtkImageAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	void SetAggregationPoint(unsigned short downRangeBorder, unsigned short upRangeBorder, unsigned short valueToSet);
	static aggregationFilterForVtk *New();
protected:
	
	struct aggregationUnit
	{
	public:
		aggregationUnit(unsigned short downRangeBorder, unsigned short upRangeBorder, unsigned short valueToSet)
		{
			this->downRangeBorder = downRangeBorder;
			this->upRangeBorder = upRangeBorder;
			this->valueToSet = valueToSet;
		}
		unsigned short downRangeBorder, upRangeBorder, valueToSet;
	};

	aggregationFilterForVtk();
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *); // Here is real filter logic

	std::vector< aggregationUnit > aggregationData;

private:
	aggregationFilterForVtk( const aggregationFilterForVtk&); // Not implemented
	void operator=(const aggregationFilterForVtk&); // Not implemented
};

#endif