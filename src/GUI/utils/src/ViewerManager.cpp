#include "GUI/utils/ViewerManager.h"


ViewerManager *viewerManagerInstance = NULL;
//*******************************************************************************************
ViewerManager *
ViewerManager::getInstance()
{
	ASSERT( viewerManagerInstance );
	return viewerManagerInstance;
}

ViewerManager::ViewerManager( ViewerManager *aInstance ): mSelectedViewer( NULL )
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

}

void
ViewerManager::finalize()
{
	
}

M4D::GUI::Viewer::AGLViewer*
ViewerManager::getSelectedViewer()
{
	return mSelectedViewer;
}

void
ViewerManager::deselectCurrentViewer()
{
	selectViewer( NULL );
}

void
ViewerManager::selectViewer( M4D::GUI::Viewer::AGLViewer *aViewer )
{
	if ( aViewer != NULL && mSelectedViewer == aViewer ) { // prevent infinite cycling
		return;
	}
	M4D::GUI::Viewer::AGLViewer *tmp = mSelectedViewer;
	if ( mSelectedViewer ) {
		mSelectedViewer = NULL;
		tmp->deselect();
	}
	if ( aViewer ) {
		mSelectedViewer = aViewer;
		aViewer->select();
	}
	viewerSelectionChangedHelper();
}

void
ViewerManager::deselectViewer( M4D::GUI::Viewer::AGLViewer *aViewer )
{
	if ( aViewer != NULL && mSelectedViewer == aViewer ) { 
		mSelectedViewer = NULL;
		aViewer->deselect();
	}
	viewerSelectionChangedHelper();
}

