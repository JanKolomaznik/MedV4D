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
signals:
	void
	viewerSelectionChanged();
protected:
	void
	viewerSelectionChangedHelper();

	bool	mInitialized;
	QApplication *mApp;
};
