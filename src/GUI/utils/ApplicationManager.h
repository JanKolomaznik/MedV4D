#include "GUI/utils/OpenGLManager.h"

class QApplication;

class ApplicationManager: public OpenGLManager
{
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
protected:
	bool	mInitialized;
	QApplication *mApp;
};
