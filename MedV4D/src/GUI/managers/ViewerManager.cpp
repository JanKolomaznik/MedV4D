#include "MedV4D/GUI/managers/ViewerManager.h"
#include "MedV4D/GUI/utils/ViewerAction.h"
#include <QtCore>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/bind/bind.hpp>
#include <algorithm>

#include "MedV4D/GUI/managers/ApplicationManager.h"

ViewerManager *viewerManagerInstance = NULL;
//*******************************************************************************************

struct ViewerManagerPimpl
{
	ViewerManagerPimpl() : mSelectedViewer( NULL ), mViewerActionsCreated( false )
	{}
	M4D::GUI::Viewer::AGLViewer* mSelectedViewer;

	ViewerActionSet mViewerActions;
	bool mViewerActionsCreated;

	std::vector<M4D::GUI::Viewer::AGLViewer*> mRegisteredViewers;
	ViewerListModel mRegisteredViewersModel;
};



ViewerManager *
ViewerManager::getInstance()
{
	ASSERT( viewerManagerInstance );
	return viewerManagerInstance;
}

ViewerManager::ViewerManager( ViewerManager *aInstance )
{
	ASSERT( aInstance );
	viewerManagerInstance = aInstance;
}

ViewerManager::~ViewerManager()
{

}

void
ViewerManager::initialize()
{
	mPimpl = new ViewerManagerPimpl;


}

void
ViewerManager::finalize()
{
	delete mPimpl;
}

M4D::GUI::Viewer::AGLViewer*
ViewerManager::getSelectedViewer()
{
	ASSERT( mPimpl );
	return mPimpl->mSelectedViewer;
}

void
ViewerManager::deselectCurrentViewer()
{
	selectViewer( NULL );
}

void
ViewerManager::selectViewer( M4D::GUI::Viewer::AGLViewer *aViewer )
{
	ASSERT( mPimpl );
	if ( aViewer != NULL && mPimpl->mSelectedViewer == aViewer ) { // prevent infinite cycling
		return;
	}
	M4D::GUI::Viewer::AGLViewer *tmp = mPimpl->mSelectedViewer;
	if ( mPimpl->mSelectedViewer ) {
		mPimpl->mSelectedViewer = NULL;
		tmp->deselect();
	}
	if ( aViewer ) {
		mPimpl->mSelectedViewer = aViewer;
		aViewer->select();
	}
	viewerSelectionChangedHelper();
}

void
ViewerManager::deselectViewer( M4D::GUI::Viewer::AGLViewer *aViewer )
{
	ASSERT( mPimpl );
	if ( aViewer != NULL && mPimpl->mSelectedViewer == aViewer ) {
		mPimpl->mSelectedViewer = NULL;
		aViewer->deselect();
	}
	viewerSelectionChangedHelper();
}

void ViewerManager::registerViewer(M4D::GUI::Viewer::AGLViewer *aViewer)
{
	//TODO - is locking needed?
	mPimpl->mRegisteredViewers.push_back(aViewer);
}

void ViewerManager::unregisterViewer(M4D::GUI::Viewer::AGLViewer *aViewer)
{
	mPimpl->mRegisteredViewers.erase(
		std::remove(
			std::begin(mPimpl->mRegisteredViewers),
			std::end(mPimpl->mRegisteredViewers),
			aViewer),
		std::end(mPimpl->mRegisteredViewers));
}

ViewerActionSet &
ViewerManager::getViewerActionSet()
{
	ASSERT( mPimpl );
	if (!mPimpl->mViewerActionsCreated) {
		createViewerActions();
	}
	return mPimpl->mViewerActions;
}

M4D::GUI::Viewer::AGLViewer *ViewerManager::getViewer(int aIndex)
{
	if (aIndex < 0 || aIndex >= mPimpl->mRegisteredViewers.size()) {
		return nullptr;
	}

	return mPimpl->mRegisteredViewers[aIndex];
}

int ViewerManager::getViewerIndex(M4D::GUI::Viewer::AGLViewer *aViewer)
{
	if (aViewer == nullptr) {
		return -1;
	}
	for (int i = 0; i < mPimpl->mRegisteredViewers.size(); ++i) {
		if (mPimpl->mRegisteredViewers[i] == aViewer) {
			return i;
		}
	}
	return -1;
}

ViewerListModel *
ViewerManager::registeredViewers() const
{
	return &(mPimpl->mRegisteredViewersModel);
}

void
ViewerManager::createViewerActions()
{
	M4D::ApplicationManager &app = *(M4D::ApplicationManager::getInstance());


	QActionGroup *viewGroup = new QActionGroup( NULL );
	QAction *action2D = createGeneralViewerAction(
			QString( "2D" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setViewType, _1, M4D::GUI::Viewer::vt2DAlignedSlices ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt2DAlignedSlices) ),
			true,
			app.getIcon( "2D" )
			);
	QAction *action3D = createGeneralViewerAction(
			QString( "3D" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setViewType, _1, M4D::GUI::Viewer::vt3D ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "3D" )
			);
	viewGroup->addAction( action2D );
	viewGroup->addAction( action3D );
	mPimpl->mViewerActions.addActionGroup( viewGroup );
	mPimpl->mViewerActions.addSeparator();
	//****************************
	QAction *actionEnableJittering = createGeneralViewerAction(
			QString( "Jittering" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::enableJittering, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::isJitteringEnabled, _1 ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "jittering" )
			);
	mPimpl->mViewerActions.addAction( actionEnableJittering );
	//****************************
	QAction *actionEnableShading = createGeneralViewerAction(
			QString( "Shading" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::enableShading, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::isShadingEnabled, _1 ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "shading" )
			);
	mPimpl->mViewerActions.addAction( actionEnableShading );
	//****************************
	QAction *actionEnableBoundingBox = createGeneralViewerAction(
			QString( "Bounding Box" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::enableBoundingBox, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::isBoundingBoxEnabled, _1 ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "bounding_box" )
			);
	mPimpl->mViewerActions.addAction( actionEnableBoundingBox );
	mPimpl->mViewerActions.addSeparator();
	//****************************
	QAction *actionEnableCutPlane = createGeneralViewerAction(
			QString( "Cut Plane" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::enableCutPlane, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::isCutPlaneEnabled, _1 ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "cut_plane" )
			);
	mPimpl->mViewerActions.addAction( actionEnableCutPlane );
	//****************************
	QAction *actionEnableVolumeRestriction = createGeneralViewerAction(
			QString( "Volume Restriction" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::enableVolumeRestrictions, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::isVolumeRestrictionEnabled, _1 ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "volume_restrictions" )
			);
	mPimpl->mViewerActions.addAction( actionEnableVolumeRestriction );
	//****************************
	QAction *actionEnableInterpolation = createGeneralViewerAction(
			QString( "Enable Interpolation" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::enableInterpolation, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::isInterpolationEnabled, _1 ),
			M4D::Functors::PredicateAlwaysTrue(),
			true,
			app.getIcon( "interpolation" )
			);
	mPimpl->mViewerActions.addAction( actionEnableInterpolation );
	mPimpl->mViewerActions.addSeparator();
	//****************************
	QAction *actionXY = createGeneralViewerAction(
			QString( "XY" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setCurrentViewPlane, _1, XY_PLANE ),
			(boost::lambda::bind<CartesianPlanes>( &M4D::GUI::Viewer::GeneralViewer::getCurrentViewPlane, boost::lambda::_1 ) == boost::lambda::constant(XY_PLANE) ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt2DAlignedSlices) ),
			true,
			app.getIcon( "XY" )
			);
	QAction *actionYZ = createGeneralViewerAction(
			QString( "YZ" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setCurrentViewPlane, _1, YZ_PLANE ),
			(boost::lambda::bind<CartesianPlanes>( &M4D::GUI::Viewer::GeneralViewer::getCurrentViewPlane, boost::lambda::_1 ) == boost::lambda::constant(YZ_PLANE) ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt2DAlignedSlices) ),
			true,
			app.getIcon( "YZ" )
			);
	QAction *actionXZ = createGeneralViewerAction(
			QString( "XZ" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setCurrentViewPlane, _1, XZ_PLANE ),
			(boost::lambda::bind<CartesianPlanes>( &M4D::GUI::Viewer::GeneralViewer::getCurrentViewPlane, boost::lambda::_1 ) == boost::lambda::constant(XZ_PLANE) ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt2DAlignedSlices) ),
			true,
			app.getIcon( "XZ" )
			);
	QActionGroup *planeGroup = new QActionGroup( NULL );
	planeGroup->addAction( actionXY );
	planeGroup->addAction( actionYZ );
	planeGroup->addAction( actionXZ );
	mPimpl->mViewerActions.addActionGroup( planeGroup );
	mPimpl->mViewerActions.addSeparator();
	//****************************
	QAction *actionResetView = createGeneralViewerAction(
			QString( "Reset view" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::resetView, _1 ),
			M4D::Functors::PredicateAlwaysFalse(),
			false,
			app.getIcon( "reset_view" )
			);
	mPimpl->mViewerActions.addAction( actionResetView );
	//*******************************
	//QAction *colorTransformChooser = new QAction( "aaaa", NULL );
	/*QMenu *menu = new QMenu( "Color transform" );
	QAction * testAction = new QAction( "1111", menu );
	menu->addAction( testAction );
	testAction = new QAction( "2222", menu );
	menu->addAction( testAction );*/
	void (M4D::GUI::Viewer::GeneralViewer::*setColTr) ( const QString & ) = &M4D::GUI::Viewer::GeneralViewer::setColorTransformType;
	QAction *colorTransformChooser = createGeneralViewerSubmenuAction(
			QString( "Color transform" ),
			boost::bind( setColTr, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::getAvailableColorTransformationNames, _1 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::getColorTransformName, _1 ),
			M4D::Functors::PredicateAlwaysTrue()
			);
	mPimpl->mViewerActions.addAction( colorTransformChooser );
	//****************************
	void (M4D::GUI::Viewer::GeneralViewer::*setZoom) ( const QString & ) = &M4D::GUI::Viewer::GeneralViewer::setZoom;
	QAction *zoomChooser = createGeneralViewerSubmenuAction(
			QString( "Zoom" ),
			boost::bind( setZoom, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::getPredefinedZoomValueNames, _1 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::getZoomValueName, _1 ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt2DAlignedSlices) ),
			NULL,
			true
			);
	mPimpl->mViewerActions.addAction( zoomChooser );
	//****************************
	QAction *actionEnableIntegratedTF = createGeneralViewerAction(
			QString( "Integrated TF" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::enableIntegratedTransferFunction, _1, _2 ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::isIntegratedTransferFunctionEnabled, _1 ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "integrated_tf" )
			);
	mPimpl->mViewerActions.addAction( actionEnableIntegratedTF );

	QAction *actionLowQuality = createGeneralViewerAction(
			QString( "Low Quality" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setRenderingQuality, _1, M4D::GUI::Viewer::qmLow ),
			(boost::lambda::bind<M4D::GUI::Viewer::QualityMode>( &M4D::GUI::Viewer::GeneralViewer::getRenderingQuality, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::qmLow) ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "low_quality" )
			);
	QAction *actionNormalQuality = createGeneralViewerAction(
			QString( "Normal Quality" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setRenderingQuality, _1, M4D::GUI::Viewer::qmNormal ),
			(boost::lambda::bind<M4D::GUI::Viewer::QualityMode>( &M4D::GUI::Viewer::GeneralViewer::getRenderingQuality, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::qmNormal) ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "normal_quality" )
			);
	QAction *actionHighQuality = createGeneralViewerAction(
			QString( "High Quality" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setRenderingQuality, _1, M4D::GUI::Viewer::qmHigh ),
			(boost::lambda::bind<M4D::GUI::Viewer::QualityMode>( &M4D::GUI::Viewer::GeneralViewer::getRenderingQuality, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::qmHigh) ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "high_quality" )
			);
	QAction *actionFinestQuality = createGeneralViewerAction(
			QString( "Finest Quality" ),
			boost::bind( &M4D::GUI::Viewer::GeneralViewer::setRenderingQuality, _1, M4D::GUI::Viewer::qmFinest ),
			(boost::lambda::bind<M4D::GUI::Viewer::QualityMode>( &M4D::GUI::Viewer::GeneralViewer::getRenderingQuality, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::qmFinest) ),
			(boost::lambda::bind<M4D::GUI::Viewer::ViewType>( &M4D::GUI::Viewer::GeneralViewer::getViewType, boost::lambda::_1 ) == boost::lambda::constant(M4D::GUI::Viewer::vt3D) ),
			true,
			app.getIcon( "finest_quality" )
			);
	QActionGroup *qualityGroup = new QActionGroup( NULL );
	qualityGroup->addAction( actionLowQuality );
	qualityGroup->addAction( actionNormalQuality );
	qualityGroup->addAction( actionHighQuality );
	qualityGroup->addAction( actionFinestQuality );
	mPimpl->mViewerActions.addActionGroup( qualityGroup );
	mPimpl->mViewerActions.addSeparator();

	mPimpl->mViewerActionsCreated = true;
}


int
ViewerListModel::rowCount(const QModelIndex & parent) const
{
	return ViewerManager::getInstance()->mPimpl->mRegisteredViewers.size();
}

QVariant
ViewerListModel::data(const QModelIndex & index, int role) const
{
	if (role != Qt::DisplayRole) {
		return QVariant();
	}
	if (index.row() >= 0 && index.row() < rowCount()) {
		return QVariant(ViewerManager::getInstance()->mPimpl->mRegisteredViewers[index.row()]->name().c_str());
	}
	return QVariant();
}

QVariant
ViewerListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole || orientation != Qt::Horizontal) {
		return QVariant();
	}
	return QVariant();
}
