#ifndef VIEWER_MANAGER_H
#define VIEWER_MANAGER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/GUI/widgets/AGLViewer.h"
//#include "MedV4D/GUI/utils/ViewerAction.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{
	class AGLViewer;
} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

class ViewerActionSet;

struct ViewerManagerPimpl;

class ViewerManager
{
public:
	

	static ViewerManager *
	getInstance();

	virtual void
	initialize();

	virtual void
	finalize();

	virtual 
	~ViewerManager();

	M4D::GUI::Viewer::AGLViewer*
	getSelectedViewer();

	void
	deselectCurrentViewer();

	virtual void
	selectViewer( M4D::GUI::Viewer::AGLViewer *aViewer );

	virtual void
	deselectViewer( M4D::GUI::Viewer::AGLViewer *aViewer );

	ViewerActionSet &
	getViewerActionSet();

	virtual void
	notifyAboutChangedViewerSettings()=0;
protected:
	virtual void
	viewerSelectionChangedHelper() = 0;
	
	virtual void
	createViewerActions();


	ViewerManager( ViewerManager *aInstance );

	ViewerManagerPimpl *mPimpl;

};


#endif /*VIEWER_MANAGER_H*/
