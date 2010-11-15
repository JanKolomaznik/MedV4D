#include<TFHolderFactory.h>

#include <TFGrayscaleHolder.h>
#include <TFGrayscaleAlphaHolder.h>
#include <TFRGBHolder.h>
#include <TFRGBaHolder.h>
#include <TFHSVHolder.h>
#include <TFHSVaHolder.h>

namespace M4D {
namespace GUI {

TFActions TFHolderFactory::createMenuTFActions(QWidget *parent){

	TFActions actions;

	//adds transferfunction types to menu
	actions.push_back(new TFAction(parent, TFHOLDER_GRAYSCALE));
	actions.push_back(new TFAction(parent, TFHOLDER_GRAYSCALE_ALPHA));
	actions.push_back(new TFAction(parent, TFHOLDER_RGB));
	actions.push_back(new TFAction(parent, TFHOLDER_RGBA));
	actions.push_back(new TFAction(parent, TFHOLDER_HSV));
	actions.push_back(new TFAction(parent, TFHOLDER_HSVA));
	//actions.push_back(new TFAction(menu, TFHOLDER_MYTYPE));

	return actions;
}

TFAbstractHolder* TFHolderFactory::createHolder(QWidget* window, const TFHolderType holderType){

	switch(holderType)
	{
		case TFHOLDER_GRAYSCALE:
		{
			return new TFGrayscaleHolder(window);
		}
		case TFHOLDER_GRAYSCALE_ALPHA:
		{
			return new TFGrayscaleAlphaHolder(window);
		}
		case TFHOLDER_RGB:
		{
			return new TFRGBHolder(window);
		}
		case TFHOLDER_RGBA:
		{
			return new TFRGBaHolder(window);
		}
		case TFHOLDER_HSV:
		{
			return new TFHSVHolder(window);
		}
		case TFHOLDER_HSVA:
		{
			return new TFHSVaHolder(window);
		}
		case TFHOLDER_UNKNOWN:
		default:
		{
			tfAssert("unknown holder");
			break;
		}
	}
	return NULL;
}

TFAbstractHolder* TFHolderFactory::loadHolder(QWidget* window){
	
	QString fileName = QFileDialog::getOpenFileName(
		window,
		QObject::tr("Open Transfer Function"),
		QDir::currentPath(),
		QObject::tr("TF Files (*.tf *.xml)"));

	if (fileName.isEmpty()) return NULL;

	QFile qFile(fileName);

	if (!qFile.open(QFile::ReadOnly | QFile::Text)) {
		QMessageBox::warning(
			window,
			QObject::tr("Transfer Functions"),
			QObject::tr("Cannot read file %1:\n%2.").arg(fileName).arg(qFile.errorString()));
		return NULL;
	}

	TFTypeSwitcher switcher;
	TFHolderType holderType = switcher.read(&qFile);
	qFile.close();

	qFile.open(QFile::ReadOnly | QFile::Text);
	TFAbstractHolder* loaded = createHolder(window, holderType);

	if(loaded)
	{
		if(!loaded->load_(qFile))
		{ 
			QMessageBox::warning(
				window,
				QObject::tr("TFXmlReader"),
				QObject::tr("Parse error in file %1").arg(fileName));
		}
	}

	qFile.close();
	return loaded;
}

TFHolderType TFHolderFactory::TFTypeSwitcher::read(QIODevice* device){

	setDevice(device);

	while(!atEnd())
	{
		readNext(); 

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TransferFunction"))
		{
			return convert<std::string, TFHolderType>(attributes().value("holderType").toString().toStdString());
		}
	}
	return TFHOLDER_UNKNOWN;
}

} // namespace GUI
} // namespace M4D
