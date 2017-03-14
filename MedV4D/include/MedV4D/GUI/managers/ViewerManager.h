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

class ViewerListModel : public QAbstractListModel
{
public:
	friend class ViewerManager;

	ViewerListModel()
		: QAbstractListModel(nullptr)
	{}

	int
	rowCount(const QModelIndex & parent = QModelIndex()) const override;

	int
	columnCount(const QModelIndex & parent = QModelIndex()) const override
	{
		return 1;
	}

	QVariant
	data(const QModelIndex & index, int role = Qt::DisplayRole) const override;

	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

protected:
};



class ViewerManager
{
public:
	friend class ViewerListModel;

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

	virtual void
	registerViewer( M4D::GUI::Viewer::AGLViewer *aViewer );

	virtual void
	unregisterViewer( M4D::GUI::Viewer::AGLViewer *aViewer );


	ViewerActionSet &
	getViewerActionSet();

	virtual void
	notifyAboutChangedViewerSettings()=0;

	M4D::GUI::Viewer::AGLViewer *getViewer(int aIndex);

	int
	getViewerIndex(M4D::GUI::Viewer::AGLViewer *aViewer);

	ViewerListModel *
	registeredViewers() const;
protected:
	virtual void
	viewerSelectionChangedHelper() = 0;

	virtual void
	createViewerActions();


	ViewerManager( ViewerManager *aInstance );

	ViewerManagerPimpl *mPimpl;

};



#endif /*VIEWER_MANAGER_H*/
