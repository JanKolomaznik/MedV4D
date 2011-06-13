#ifndef VIEWER_MANAGER_H
#define VIEWER_MANAGER_H

#include "common/Common.h"
#include "GUI/widgets/AGLViewer.h"

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
protected:
	virtual void
	viewerSelectionChangedHelper() = 0;


	ViewerManager( ViewerManager *aInstance );


	M4D::GUI::Viewer::AGLViewer* mSelectedViewer;

};


#endif /*VIEWER_MANAGER_H*/
