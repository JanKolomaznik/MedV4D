#ifndef APPLICATION_MANAGER_H
#define APPLICATION_MANAGER_H


#include "GUI/utils/OpenGLManager.h"
#include "GUI/utils/ViewerManager.h"
#include <QtCore>

class QApplication;

class ApplicationManager: public QObject, public OpenGLManager, public ViewerManager
{
	Q_OBJECT
public:
	static ApplicationManager *
	getInstance();

	ApplicationManager();

	virtual void
	initialize( int argc, char** argv );

	virtual void
	finalize();

	virtual int
	exec();

	~ApplicationManager();

	void
	notifyAboutChangedViewerSettings();
signals:
	void
	viewerSelectionChanged();

	void
	selectedViewerSettingsChanged();
protected:
	void
	viewerSelectionChangedHelper();

	bool	mInitialized;
	QApplication *mApp;
};

#endif /*APPLICATION_MANAGER_H*/
