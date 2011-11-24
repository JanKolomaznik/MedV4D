#ifndef SHOULDER_MEASUREMENT_CONTROLLER_H
#define SHOULDER_MEASUREMENT_CONTROLLER_H

#include <QtCore>
#include "MedV4D/GUI/widgets/GeneralViewer.h"
#include "MedV4D/GUI/utils/ApplicationManager.h"
#include "MedV4D/GUI/utils/ProxyViewerController.h"
#include "MedV4D/GUI/utils/QtModelViewTools.h"
#include "MedV4D/GUI/utils/PrimitiveCreationEventController.h"
#include "ShoulderMeasurementModule/Computations.hpp"


class ShoulderMeasurementController: public ModeViewerController, public M4D::GUI::Viewer::RenderingExtension
{
	Q_OBJECT;
public:
	enum MeasurementMode {
		mmNONE,
		mmHUMERAL_HEAD,
		mmPROXIMAL_SHAFT,

		mmSENTINEL //use for valid interval testing 
	};

	typedef QList<QAction *> QActionList;
	typedef boost::shared_ptr< ShoulderMeasurementController > Ptr;
	typedef M4D::GUI::Viewer::ViewerController ControllerPredecessor;
	typedef M4D::GUI::Viewer::RenderingExtension RenderingExtPredecessor;

	ShoulderMeasurementController();
	~ShoulderMeasurementController();

	void
	activated()
	{

	}

	void
	deactivated()
	{
		//std::for_each( mChosenToolActions.begin(), mChosenToolActions.end(), boost::bind( &QAction::setChecked, _1, false ) );
	}

	void
	setModeId( M4D::Common::IDNumber aId )
	{
		mModeId = aId;
	}

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event );

	unsigned
	getAvailableViewTypes()const;

	void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane );

	void
	preRender3D();

	void
	postRender3D();

	void
	render3D();

	/*QActionList &
	getActions();

	QWidget *
	getAnnotationView();*/
	
	PointSet &
	getPointModel()
	{
		return mHumeralHeadPoints;
	}

	PointSet &
	getProximalShaftPointModel()
	{
		return mProximalShaftPoints;
	}

	void
	setMeasurementMode( int aMode )
	{
		ASSERT( aMode < mmSENTINEL );
		mMeasurementMode = static_cast< MeasurementMode >( aMode );
	}
signals:
	void
	updateRequest();

public slots:
	void
	analyseHumeralHead();

	void
	analyseProximalShaftOfHumerus();

protected:

public:

	PointSet mHumeralHeadPoints;
	PointSet mProximalShaftPoints;
	Qt::MouseButton	mVectorEditorInteractionButton;

	bool mOverlay;
	MeasurementMode mMeasurementMode;

	QActionList mActions;

	QActionList mChosenToolActions;

	std::map< MeasurementMode, APrimitiveCreationEventController::Ptr > mMeasurementHandlers;

	M4D::Common::IDNumber mModeId;


	HeadMeasurementData mHeadMeasurementData;
	ProximalShaftMeasurementData mProximalShaftMeasurementData;
};


#endif /*SHOULDER_MEASUREMENT_CONTROLLER_H*/
