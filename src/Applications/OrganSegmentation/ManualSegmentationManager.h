#ifndef MANUAL_SEGMENTATION_MANAGER_H
#define MANUAL_SEGMENTATION_MANAGER_H

#include "Imaging.h"
#include "GUI/m4dGUISliceViewerWidget2.h"
#include "MainManager.h"

typedef M4D::Imaging::SlicedGeometry< M4D::Imaging::Geometry::BSpline<float32,2> >	GDataSet;

class ManualSegmentationManager
{
public:
	static void
	Initialize();

	static void
	Finalize();

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

	static bool						_wasInitialized;
};

#endif //MANUAL_SEGMENTATION_MANAGER_H


