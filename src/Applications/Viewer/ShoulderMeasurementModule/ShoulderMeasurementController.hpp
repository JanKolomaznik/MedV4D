#ifndef SHOULDER_MEASUREMENT_CONTROLLER_H
#define SHOULDER_MEASUREMENT_CONTROLLER_H


class ShoulderMeasurementController: public ModeViewerController, public M4D::GUI::Viewer::RenderingExtension
{
	Q_OBJECT;
public:
	enum MeasurementMode {
		mmNONE,

		mmSENTINEL //use for valid interval testing 
	};

	typedef QList<QAction *> QActionList;
	typedef boost::shared_ptr< ShoulderMeasurementController > Ptr;
	typedef M4D::GUI::Viewer::ViewerController ControllerPredecessor;
	typedef M4D::GUI::Viewer::RenderingExtension RenderingExtPredecessor;

	AnnotationEditorController();
	~AnnotationEditorController();

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

	QActionList &
	getActions();

	QWidget *
	getAnnotationView();

public slots:
	void
	setAnnotationEditMode( int aMode );

	void
	abortEditInProgress();

	void
	showSettingsDialog();

	void
	setOverlay( bool aOverlay )
	{
		mSettings.overlayed = aOverlay;
		emit updateRequest();
	}
signals:
	void
	updateRequest();

protected slots:
	void
	editModeActionToggled( QAction *aAction );

	void
	updateActions();

	void
	applySettings();
protected:
	void
	renderPoints2D();

	void
	renderSpheres2D();

	void
	renderLines2D();

	void
	renderAngles2D();

	void
	renderPoints3D();

	void
	renderSpheres3D();

	void
	renderLines3D();

	void
	renderAngles3D();
public:

	PointSet mPoints;
	LineSet mLines;
	SphereSet mSpheres;
	Qt::MouseButton	mVectorEditorInteractionButton;

	bool mOverlay;
	AnnotationEditMode mEditMode;

	QActionList mActions;

	QActionList mChosenToolActions;

	std::map< AnnotationEditMode, APrimitiveCreationEventController::Ptr > mAnnotationPrimitiveHandlers;

	AnnotationSettings mSettings;

	AnnotationSettingsDialog *mSettingsDialog;

	AnnotationWidget *mAnnotationView;

	M4D::Common::IDNumber mModeId;

};


#endif /*SHOULDER_MEASUREMENT_CONTROLLER_H*/
