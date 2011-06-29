#ifndef ANNOTATION_EDITOR_CONTROLLER_HPP
#define ANNOTATION_EDITOR_CONTROLLER_HPP

#include <QtGui>

#include "common/GeometricPrimitives.h"
#include "common/Sphere.h"
#include "GUI/widgets/GeneralViewer.h"
#include "GUI/utils/OGLTools.h"
#include "GUI/utils/OGLDrawing.h"
#include <algorithm>



//using namespace M4D;
/*class PointSet
{
public:
	void
	addPoint( Vector3f aCoord )
	{
		D_PRINT( "Adding point - " << aCoord );
		mPoints.push_back( aCoord );
		mSelectedIdx = mPoints.size() - 1;
		mSelected = true;
	}

	std::vector< Vector3f > mPoints;

	size_t mSelectedIdx;
	bool mSelected;
};*/

class AnnotationSettingsDialog;

class AnnotationBasicViewerController: public M4D::GUI::Viewer::AViewerController
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
};

typedef std::vector< M4D::Point3Df > PointSet;
typedef std::vector< M4D::Line3Df > LineSet;
typedef std::vector< M4D::Sphere3Df > SphereSet;

class AnnotationEditorController: public M4D::GUI::Viewer::ViewerController, public M4D::GUI::Viewer::RenderingExtension
{
	Q_OBJECT;
public:
	struct AnnotationSettings {
		//General
		bool annotationsEnabled;

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

public slots:
	void
	setAnnotationEditMode( int aMode );

	void
	abortEditInProgress();

	void
	showSettingsDialog();
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

	std::map< AnnotationEditMode, AnnotationBasicViewerController::Ptr > mAnnotationPrimitiveHandlers;

	AnnotationSettings mSettings;

	AnnotationSettingsDialog *mSettingsDialog;

};

#endif /*ANNOTATION_EDITOR_CONTROLLER_HPP*/
