#ifndef MANUAL_SEGMENTATION_MANAGER_H
#define MANUAL_SEGMENTATION_MANAGER_H

#include "Imaging.h"
#include "GUI/m4dGUISliceViewerWidget2.h"
#include "MainManager.h"
#include "Common.h"

#include <QtCore>

typedef M4D::Imaging::SlicedGeometry< M4D::Imaging::Geometry::BSpline<float32,2> >	GDataSet;

class ManualSegmentationManager;

class ManualSegmentationManager: public QObject
{
	Q_OBJECT
public:
	static ManualSegmentationManager &
	Instance();

	void
	Initialize();

	void
	Finalize();

	ImageConnectionType *
	GetInputConnection()
		{ return _inConnection; }

	M4D::Viewer::SliceViewerSpecialStateOperatorPtr
	GetSpecialState()
	{
		return _specialState;
	}
protected:
	ImageConnectionType				*_inConnection;
	M4D::Viewer::SliceViewerSpecialStateOperatorPtr 	_specialState;
	InputImagePtr				 	_inputImage;
	GDataSet::Ptr					_dataset;

	static ManualSegmentationManager		*_instance;

	bool						_wasInitialized;
};

class EInstanceUnavailable : public M4D::ErrorHandling::ExceptionBase
{
public:
	EInstanceUnavailable() throw() : M4D::ErrorHandling::ExceptionBase( "Singleton instance unavailable" )
		{}
};

#endif //MANUAL_SEGMENTATION_MANAGER_H


