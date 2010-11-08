#include<TFHolderFactory.h>

namespace M4D {
namespace GUI {

TFActions TFHolderFactory::createMenuTFActions(QWidget *parent){

	TFActions actions;

	//adds transferfunction types to menu
	actions.push_back(new TFAction(parent, TFTYPE_SIMPLE));
	actions.push_back(new TFAction(parent, TFTYPE_GRAYSCALE_TRANSPARENCY));
	actions.push_back(new TFAction(parent, TFTYPE_RGB));
	actions.push_back(new TFAction(parent, TFTYPE_RGBA));
	//actions.push_back(new TFAction(menu, TFTYPE_MYTYPE));

	return actions;
}

TFAbstractHolder* TFHolderFactory::createHolder(QWidget* window, const TFType holderType){

	switch(holderType)
	{
		case TFTYPE_SIMPLE:
		{
			return new TFSimpleHolder(window);
		}
		case TFTYPE_GRAYSCALE_TRANSPARENCY:
		{
			return new TFGrayscaleTransparencyHolder(window);
		}
		case TFTYPE_RGB:
		{
			return new TFRGBHolder(window);
		}
		case TFTYPE_RGBA:
		{
			return new TFRGBaHolder(window);
		}
		case TFTYPE_UNKNOWN:
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
	TFType tfType = switcher.read(&qFile);
	qFile.close();

	qFile.open(QFile::ReadOnly | QFile::Text);
	TFAbstractHolder* loaded = createHolder(window, tfType);

	if(loaded)
	{
		if(!loaded->load_(qFile))
		{ 
			QMessageBox::warning(
				window,
				QObject::tr("TFXmlSimpleReader"),
				QObject::tr("Parse error in file %1").arg(fileName));
		}
	}

	qFile.close();
	return loaded;
}

TFType TFHolderFactory::TFTypeSwitcher::read(QIODevice* device){

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
			return convert<std::string, TFType>(attributes().value("type").toString().toStdString());
		}
	}
	return TFTYPE_UNKNOWN;
}

} // namespace GUI
} // namespace M4D
