#ifndef MANUAL_SEGMENTATION_MANAGER_H
#define MANUAL_SEGMENTATION_MANAGER_H

#include "Imaging/Imaging.h"
#include "MainManager.h"
#include "ManagerViewerSpecialState.h"
#include "MedV4D/Common/Common.h"
#include "SegmentationManager.h"

#include <QtCore>

static const float32 DISTANCE_TOLERATION_SQUARED = 3.0f;

class ManualSegmentationManager: public SegmentationManager
{
	Q_OBJECT
public:
	enum InternalState {
		SELECT,
		SELECTED,
		CREATING,
		SELECT_POINT,
		SELECTED_POINT
	};

	static ManualSegmentationManager &
	Instance();
	
	std::string
	GetName()
		{ return "Manual Segmentation"; }

	void
	Initialize();

	void
	Finalize();

	M4D::Viewer::SliceViewerSpecialStateOperatorPtr
	GetSpecialState()
	{
		return _specialState;
	}

	GDataset::Ptr
	GetOutputGeometry()
		{ return _dataset; }

	InputImageType::Ptr
	GetInputImage()
		{ return _inputImage; }

	InternalState
	GetInternalState()const
		{ return _state; }

	void
	Draw( int32 sliceNum, double zoomRate );
	void
	LeftButtonMove( Vector< float32, 2 > diff );
	void
	RightButtonMove( Vector< float32, 2 > diff ){}
	void
	RightButtonDown( Vector< float32, 2 > pos, int32 sliceNum );
	void
	LeftButtonDown( Vector< float32, 2 > pos, int32 sliceNum );

	void
	Activate( InputImageType::Ptr inImage );
	void
	Activate( InputImageType::Ptr inImage, GDataset::Ptr geometry );
	void
	ActivateManagerInit( InputImageType::Ptr inImage, GDataset::Ptr geometry );
public slots:
	void
	ActivateManager();

	void
	SetCreatingState( bool enable );

	void
	SetEditPointsState( bool enable );

	void
	DeleteSelectedCurve();

signals:
	void StateUpdated();
protected:

	InternalState	_state;
	CurveType	*_curve;
	int32		_curveSlice;
	int32		_curveIdx;
	int32		_curvePointIndex;

	void
	SetState( InternalState state );
	void
	FinishCurveCreating();
	void
	PrepareNewCurve( int32 sliceNum );

	InputImageType::Ptr			 	_inputImage;
	GDataset::Ptr					_dataset;

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


