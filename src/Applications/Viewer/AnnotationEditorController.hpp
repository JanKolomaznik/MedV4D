#ifndef ANNOTATION_EDITOR_CONTROLLER_HPP
#define ANNOTATION_EDITOR_CONTROLLER_HPP

#include <QtGui>

#include "common/GeometricPrimitives.h"
#include "common/Sphere.h"
#include "GUI/widgets/GeneralViewer.h"
#include "GUI/utils/OGLTools.h"
#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/QtModelViewTools.h"
#include "GUI/utils/PrimitiveCreationEventController.h"
#include <algorithm>
#include "GUI/utils/ApplicationManager.h"

class AnnotationSettingsDialog;
class AnnotationWidget;

typedef VectorItemModel< M4D::Point3Df > PointSet;
typedef VectorItemModel< M4D::Line3Df > LineSet;
typedef VectorItemModel< M4D::Sphere3Df > SphereSet;

/*class AnnotationBasicViewerController: public M4D::GUI::Viewer::AViewerController
{
public:
	typedef boost::shared_ptr< AnnotationBasicViewerController > Ptr;

	Qt::MouseButton	mVectorEditorInteractionButton;

	AnnotationBasicViewerController(): mVectorEditorInteractionButton( Qt::LeftButton ) {}

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) { return false; }

	virtual void
	abortEditation(){}
};*/

template<typename TPrimitive>
class AnnotationPrimitiveController: public TemplatedPrimitiveCreationEventController< TPrimitive >
{
public:
	typedef VectorItemModel< TPrimitive > PrimitiveSet;

	AnnotationPrimitiveController( PrimitiveSet &aPrimitives ): mPrimitives( aPrimitives ) {}

protected:

	virtual TPrimitive *
	createPrimitive( const TPrimitive & aPrimitive )
	{
		mPrimitives.push_back( aPrimitive );
		return &(mPrimitives[mPrimitives.size()-1]);
	}

	virtual void
	primitiveFinished( TPrimitive *aPrimitive )
	{
	}

	virtual void
	disposePrimitive( TPrimitive *aPrimitive )
	{
		mPrimitives.resize( mPrimitives.size()-1 );
	}

	PrimitiveSet &mPrimitives;
};

class AnnotationEditorController: public M4D::GUI::Viewer::ViewerController, public M4D::GUI::Viewer::RenderingExtension
{
	Q_OBJECT;
public:
	struct AnnotationSettings {
		//General
		bool annotationsEnabled;

		bool overlayed;

		//Points
		QColor pointColor;

		//Lines

		//Spheres
		bool sphereContourVisible2D;
		QColor sphereContourColor2D;
		bool sphereFill2D;
		QColor sphereFillColor2D;

		QColor sphereColor3D;
		bool sphereEnableShading3D;
	};

	enum AnnotationEditMode {
		aemNONE,
		aemPOINTS,
		aemSPHERES,
		aemLINES,
		aemANGLES,

		aemSENTINEL //use for valid interval testing 
	};

	typedef QList<QAction *> QActionList;
	typedef boost::shared_ptr< AnnotationEditorController > Ptr;
	typedef M4D::GUI::Viewer::ViewerController ControllerPredecessor;
	typedef M4D::GUI::Viewer::RenderingExtension RenderingExtPredecessor;

	AnnotationEditorController();
	~AnnotationEditorController();

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

#endif /*ANNOTATION_EDITOR_CONTROLLER_HPP*/
