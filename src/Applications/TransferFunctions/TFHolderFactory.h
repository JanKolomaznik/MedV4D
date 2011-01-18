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
#include <TFHolder.h>

namespace M4D {
namespace GUI {

class TFHolderFactory{

public:
	
	static TFActions createMenuTFActions(QObject *parent);

	static TFHolder* createHolder(QMainWindow* mainWindow, const TFHolder::Type holderType, const TFSize& domain);

	static TFHolder* loadHolder(QMainWindow* mainWindow, const TFSize& domain);

protected:

	class TFTypeSwitcher: public QXmlStreamReader{
	public:
		TFTypeSwitcher(){}
		~TFTypeSwitcher(){}

		TFHolder::Type read(QIODevice* device, bool& error);
	};

private:

	TFHolderFactory(){}
	~TFHolderFactory(){}
};

} // namespace GUI
} // namespace M4D

#endif	//TF_HOLDERFACTORY