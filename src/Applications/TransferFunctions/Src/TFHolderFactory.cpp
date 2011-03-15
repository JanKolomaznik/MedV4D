#include<TFHolderFactory.h>

#include <TFHolder.h>

#include <TFRGBaFunction.h>
#include <TFHSVaFunction.h>

#include <TFGrayscaleAlphaPainter.h>
#include <TFRGBaPainter.h>
#include <TFHSVaPainter.h>

#include <TFSimpleModifier.h>
#include <TFPolygonModifier.h>

namespace M4D {
namespace GUI {

TFActions TFHolderFactory::createMenuTFActions(QObject *parent){

	TFActions actions;

	//adds transferfunction types to menu
	actions.push_back(new TFAction(parent, TFHolder::TFHolderGrayscale));
	actions.push_back(new TFAction(parent, TFHolder::TFHolderGrayscaleAlpha));
	actions.push_back(new TFAction(parent, TFHolder::TFHolderRGB));
	actions.push_back(new TFAction(parent, TFHolder::TFHolderRGBa));
	actions.push_back(new TFAction(parent, TFHolder::TFHolderHSV));
	actions.push_back(new TFAction(parent, TFHolder::TFHolderHSVa));
	actions.push_back(new TFAction(parent, TFHolder::TFHolderPolygonRGBa));
	//actions.push_back(new TFAction(menu, TFHOLDER_MYTYPE));

	return actions;
}

TFHolder* TFHolderFactory::createHolder(QMainWindow* mainWindow, const TFHolder::Type holderType, const TFSize domain){

	switch(holderType)
	{
		case TFHolder::TFHolderGrayscale:
		{
			return new TFHolder(mainWindow,
				TFAbstractFunction::Ptr(new TFRGBaFunction(domain)),
				TFAbstractModifier::Ptr(new TFSimpleModifier(TFAbstractModifier::TFModifierGrayscale, domain)),
				TFAbstractPainter::Ptr(new TFGrayscaleAlphaPainter(false)),
				TFHolder::TFHolderGrayscale);
		}
		case TFHolder::TFHolderGrayscaleAlpha:
		{
			return new TFHolder(mainWindow,
				TFAbstractFunction::Ptr(new TFRGBaFunction(domain)),
				TFAbstractModifier::Ptr(new TFSimpleModifier(TFAbstractModifier::TFModifierGrayscaleAlpha, domain)),
				TFAbstractPainter::Ptr(new TFGrayscaleAlphaPainter(true)),
				TFHolder::TFHolderGrayscaleAlpha);
		}
		case TFHolder::TFHolderRGB:
		{
			return new TFHolder(mainWindow,
				TFAbstractFunction::Ptr(new TFRGBaFunction(domain)),
				TFAbstractModifier::Ptr(new TFSimpleModifier(TFAbstractModifier::TFModifierRGB, domain)),
				TFAbstractPainter::Ptr(new TFRGBaPainter(false)),
				TFHolder::TFHolderRGB);
		}
		case TFHolder::TFHolderRGBa:
		{
			return new TFHolder(mainWindow,
				TFAbstractFunction::Ptr(new TFRGBaFunction(domain)),
				TFAbstractModifier::Ptr(new TFSimpleModifier(TFAbstractModifier::TFModifierRGBa, domain)),
				TFAbstractPainter::Ptr(new TFRGBaPainter(true)),
				TFHolder::TFHolderRGBa);
		}
		case TFHolder::TFHolderHSV:
		{
			return new TFHolder(mainWindow,
				TFAbstractFunction::Ptr(new TFHSVaFunction(domain)),
				TFAbstractModifier::Ptr(new TFSimpleModifier(TFAbstractModifier::TFModifierHSV, domain)),
				TFAbstractPainter::Ptr(new TFHSVaPainter(false)),
				TFHolder::TFHolderHSV);
		}
		case TFHolder::TFHolderHSVa:
		{
			return new TFHolder(mainWindow,
				TFAbstractFunction::Ptr(new TFHSVaFunction(domain)),
				TFAbstractModifier::Ptr(new TFSimpleModifier(TFAbstractModifier::TFModifierHSVa, domain)),
				TFAbstractPainter::Ptr(new TFHSVaPainter(true)),
				TFHolder::TFHolderHSVa);
		}
		case TFHolder::TFHolderPolygonRGBa:
		{
			return new TFHolder(mainWindow,
				TFAbstractFunction::Ptr(new TFRGBaFunction(domain)),
				TFAbstractModifier::Ptr(new TFPolygonModifier(TFAbstractModifier::TFModifierRGBa, domain)),
				TFAbstractPainter::Ptr(new TFRGBaPainter(true)),
				TFHolder::TFHolderPolygonRGBa);
		}
	}
	return NULL;
}

TFHolder* TFHolderFactory::loadHolder(QMainWindow* mainWindow, const TFSize domain){
	
	QString fileName = QFileDialog::getOpenFileName(
		(QWidget*)mainWindow,
		QObject::tr("Open Transfer Function"),
		QDir::currentPath(),
		QObject::tr("TF Files (*.tf *.xml)"));

	if (fileName.isEmpty()) return NULL;

	QFile qFile(fileName);

	if (!qFile.open(QFile::ReadOnly | QFile::Text)) {
		QMessageBox::warning(
			(QWidget*)mainWindow,
			QObject::tr("Transfer Functions"),
			QObject::tr("Cannot read file %1:\n%2.").arg(fileName).arg(qFile.errorString()));
		return NULL;
	}

	TFTypeSwitcher switcher;
	bool error = false;
	TFHolder::Type holderType = switcher.read(&qFile, error);
	qFile.close();

	if(error)
	{
		QMessageBox::warning(
			(QWidget*)mainWindow,
			QObject::tr("TFXmlReader"),
			QObject::tr("Unknown Transfer Function in file %1").arg(fileName));
	}

	qFile.open(QFile::ReadOnly | QFile::Text);
	TFHolder* loaded = createHolder(mainWindow, holderType, domain);

	if(loaded)
	{
		if(!loaded->load_(qFile))
		{ 
			QMessageBox::warning(
				(QWidget*)mainWindow,
				QObject::tr("TFXmlReader"),
				QObject::tr("Parse error in file %1").arg(fileName));
		}
	}

	qFile.close();
	return loaded;
}

TFHolder::Type TFHolderFactory::TFTypeSwitcher::read(QIODevice* device, bool& error){

	error = false;
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
			return convert<std::string, TFHolder::Type>(attributes().value("holderType").toString().toStdString());
		}
	}
	error = true;
	return TFHolder::TFHolderGrayscale;
}

} // namespace GUI
} // namespace M4D
