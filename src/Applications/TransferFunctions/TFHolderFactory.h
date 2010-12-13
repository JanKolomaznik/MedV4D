#ifndef TF_HOLDERFACTORY
#define TF_HOLDERFACTORY

#include <QtCore/QString>
#include <QtCore/QFile>
#include <QtCore/QXmlStreamReader>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <fstream>

#include <TFAction.h>
#include <TFAbstractHolder.h>

namespace M4D {
namespace GUI {

class TFHolderFactory{

public:
	
	static TFActions createMenuTFActions(QObject *parent);

	static TFAbstractHolder* createHolder(QMainWindow* mainWindow, const TFHolderType holderType);

	static TFAbstractHolder* loadHolder(QMainWindow* mainWindow);

protected:

	class TFTypeSwitcher: public QXmlStreamReader{
	public:
		TFTypeSwitcher(){}
		~TFTypeSwitcher(){}

		TFHolderType read(QIODevice* device);
	};

private:

	TFHolderFactory(){}
	~TFHolderFactory(){}
};

} // namespace GUI
} // namespace M4D

#endif	//TF_HOLDERFACTORY