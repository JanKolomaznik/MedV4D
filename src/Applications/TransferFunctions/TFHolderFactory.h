#ifndef TF_HOLDERFACTORY
#define TF_HOLDERFACTORY

#include <QtCore/QString>
#include <QtCore/QFile>
#include <QtCore/QXmlStreamReader>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <fstream>
#include <cassert>

#include <TFSimpleHolder.h>


class TFHolderFactory{

public:
	
	static TFActions createMenuTFActions(QWidget *owner, QMenu *menu);

	static TFAbstractHolder* create(TFType &holderType);

	static TFAbstractHolder* load(QWidget* parent);

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

#endif	//TF_HOLDERFACTORY