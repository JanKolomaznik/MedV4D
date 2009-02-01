#ifndef KIDNEY_SEGMENTATION_MANAGER_H
#define KIDNEY_SEGMENTATION_MANAGER_H

#include "Imaging.h"
#include "GUI/m4dGUISliceViewerWidget2.h"
#include "MainManager.h"

typedef M4D::Imaging::Geometry::BSpline< float32, 2 >	CurveType;
typedef CurveType::PointType				PointType;

typedef M4D::Imaging::SlicedGeometry< float32, M4D::Imaging::Geometry::BSpline >	GDataSet;

struct PoleDefinition {
	PoleDefinition(): defined( false ), radius( 10.0 )
		{}

	bool		defined;
	float32		radius;
	int32		slice;
	PointType	coordinates;
};

class KidneySegmentationManager
{
public:
	static void
	Initialize();

	static void
	Finalize();

	static void
	UserInputFinished();

	static ImageConnectionType *
	GetInputConnection()
		{ return _inConnection; }

	static M4D::Viewer::SliceViewerSpecialStateOperatorPtr
	GetSpecialState()
	{
		return _specialState;
	}
protected:
	static ImageConnectionType				*_inConnection;
	static M4D::Viewer::SliceViewerSpecialStateOperatorPtr 	_specialState;
	static InputImagePtr				 	_inputImage;
	static GDataSet::Ptr					_dataset;
	static PoleDefinition					_poles[2];

	static bool						_wasInitialized;
};

#endif //KIDNEY_SEGMENTATION_MANAGER_H


