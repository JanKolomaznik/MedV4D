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
	
	static TFActions createMenuTFActions(QWidget *parent);

	static TFAbstractHolder* createHolder(QWidget* window, const TFType holderType);

	static TFAbstractHolder* loadHolder(QWidget* window);

protected:

	class TFTypeSwitcher: public QXmlStreamReader{
	public:
		TFTypeSwitcher(){}
		~TFTypeSwitcher(){}

		TFType read(QIODevice* device);
	};

private:

	TFHolderFactory(){}
	~TFHolderFactory(){}
};

} // namespace GUI
} // namespace M4D

#endif	//TF_HOLDERFACTORY