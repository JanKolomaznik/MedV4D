#include "SelectionModule/SelectionModule.hpp"



void SelectionModule::loadModule()
{
	M4D::ApplicationManager * appManager = M4D::ApplicationManager::getInstance();

	mViewerController = std::make_shared<SelectionProxyController>(*this);
	mBoxSelectionController = std::make_shared<BoxSelectionController>();

	mModeId = appManager->addNewMode(
				mViewerController/*controller*/,
				mViewerController/*renderer*/ );
	mViewerController->setModeId( mModeId );
	QObject::connect(
		mViewerController.get(),
		&SelectionProxyController::updateRequest,
		appManager,
		&M4D::ApplicationManager::updateGUIRequest);

	QToolBar *toolbar = new QToolBar( "Selection toolbar" );
	QAction *boxSelectionAction = toolbar->addAction("Box Selection");
	boxSelectionAction->setCheckable(true);
	boxSelectionAction->setChecked(false);
	QObject::connect(
		boxSelectionAction,
		&QAction::toggled,
		[this](bool aChecked) {
			if (aChecked) {
				this->startBoxSelection();
			} else {
				this->endBoxSelection();
			}
		});

	appManager->addToolBar( toolbar );

	mLoaded = true;
}


void SelectionModule::unloadModule()
{

}

void SelectionModule::startBoxSelection()
{
	mViewerController->setController(mBoxSelectionController);

	M4D::ApplicationManager * appManager = M4D::ApplicationManager::getInstance();
	appManager->activateMode( mModeId );
}

void SelectionModule::endBoxSelection()
{

}

//-----------------------------------------------------------

unsigned SelectionProxyController::getAvailableViewTypes() const
{
	return M4D::GUI::Viewer::vt3D | M4D::GUI::Viewer::vt2DAlignedSlices;
}

void SelectionProxyController::render2DAlignedSlices(int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane)
{

}

void SelectionProxyController::preRender3D()
{
	if( !mOverlay ) {
		render3D();
	}
}

void SelectionProxyController::postRender3D()
{
	if( mOverlay ) {
		render3D();
	}
}

void SelectionProxyController::render3D()
{

}
