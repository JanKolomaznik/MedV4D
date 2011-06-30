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
class AnnotationWidget;

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

//typedef std::vector< M4D::Point3Df > PointSet;
//typedef std::vector< M4D::Line3Df > LineSet;
//typedef std::vector< M4D::Sphere3Df > SphereSet;
//

class AVectorItemModel: public QAbstractListModel
{
public:
	AVectorItemModel( QObject *parent = 0): QAbstractListModel(parent)
	{}

	virtual void clear()=0;
};

template< typename TType >
class VectorItemModel:public AVectorItemModel, public std::vector< TType >
{
public:
	typedef std::vector< TType > Container;

	VectorItemModel( QObject *parent = 0): AVectorItemModel(parent)
	{}

	int 
	rowCount(const QModelIndex &parent = QModelIndex()) const
	{
		return this->size();
	}

	void
	push_back( const typename Container::value_type &aVal )
	{
		QAbstractListModel::beginInsertRows(QModelIndex(), this->size(), this->size() );
		Container::push_back( aVal );
		QAbstractListModel::endInsertRows();
	}

	void
	clear()
	{
		QAbstractListModel::beginRemoveRows(QModelIndex(), 0, this->size()-1 );
		Container::clear();
		QAbstractListModel::endRemoveRows();
	}
};

class PointSet : public VectorItemModel< M4D::Point3Df >
{
	Q_OBJECT;
public:
	
	PointSet( QObject *parent = 0): VectorItemModel< M4D::Point3Df >(parent) 
	{}

	int 
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{return 3;}

	QVariant 
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)size() || index.column() > 2 )
			return QVariant();
		
		if (role == Qt::DisplayRole) {
			return at( index.row() )[index.column()];
		}
		
		return QVariant();
	}

	QVariant 
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const
	{
		if (role != Qt::DisplayRole)
			return QVariant();

		if( orientation == Qt::Vertical ) {
			return section;
		}
		switch( section ) {
		case 0:
			return QString("X");
		case 1:
			return QString("Y");
		case 2:
			return QString("Z");
		default:
			return QVariant();
		}
		return QVariant();
	}
};

class SphereSet : public VectorItemModel< M4D::Sphere3Df >
{
	Q_OBJECT;
public:
	SphereSet( QObject *parent = 0): VectorItemModel< M4D::Sphere3Df >(parent) 
	{
	
	}

	int 
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{
		return 4;
	}

	QVariant 
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)size() || index.column() > 3 )
			return QVariant();
		
		if (role == Qt::DisplayRole) {
			if( index.column() <= 2 ) {
				return at( index.row() ).center()[index.column()];
			} else {
				return at( index.row() ).radius();
			}
		}
		
		return QVariant();
	}

	QVariant 
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const
	{
		if (role != Qt::DisplayRole)
			return QVariant();

		if( orientation == Qt::Vertical ) {
			return section;
		}
		switch( section ) {
		case 0:
			return QString("X");
		case 1:
			return QString("Y");
		case 2:
			return QString("Z");
		case 3:
			return QString("Radius");
		default:
			return QVariant();
		}
		return QVariant();
	}
};

class LineSet : public VectorItemModel<  M4D::Line3Df >
{
	Q_OBJECT;
public:
	LineSet( QObject *parent = 0): VectorItemModel<  M4D::Line3Df >(parent) 
	{
	
	}

	int 
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{
		return 6;
	}

	QVariant 
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)size() || index.column() > 5 )
			return QVariant();
		
		if (role == Qt::DisplayRole) {
			if ( index.column() < 3 ) {
				return at( index.row() ).firstPoint()[index.column()];
			} else {
				return at( index.row() ).secondPoint()[index.column()-3];
			}
		}
		
		return QVariant();
	}

	QVariant 
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const
	{
		if (role != Qt::DisplayRole)
			return QVariant();

		if( orientation == Qt::Vertical ) {
			return section;
		}
		switch( section ) {
		case 0:
			return QString("X1");
		case 1:
			return QString("Y1");
		case 2:
			return QString("Z1");
		case 3:
			return QString("X2");
		case 4:
			return QString("Y2");
		case 5:
			return QString("Z2");
		default:
			return QVariant();
		}
		return QVariant();
	}
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

	std::map< AnnotationEditMode, AnnotationBasicViewerController::Ptr > mAnnotationPrimitiveHandlers;

	AnnotationSettings mSettings;

	AnnotationSettingsDialog *mSettingsDialog;

	AnnotationWidget *mAnnotationView;

};

#endif /*ANNOTATION_EDITOR_CONTROLLER_HPP*/
