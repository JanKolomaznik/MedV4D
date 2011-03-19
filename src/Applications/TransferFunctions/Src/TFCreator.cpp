#include "TFCreator.h"

namespace M4D {
namespace GUI {

TFHolder* TFCreator::createTransferFunction(QMainWindow* mainWindow, const TF::Size domain){

	TFCreator dialog;
	dialog.exec();

	if(dialog.result() == QDialog::Rejected) return NULL;

	TF::Types::PredefinedStructure structure = dialog.getResult();

	TFAbstractFunction::Ptr function;
	TFAbstractPainter::Ptr painter;
	TFAbstractModifier::Ptr modifier;

	function = TF::Types::createFunction(structure.function, domain);
	painter = TF::Types::createPainter(structure.painter);
	modifier = TF::Types::createModifier(structure.modifier, TFWorkCopy::Ptr(new TFWorkCopy(function)), structure.painter);

	return new TFHolder(mainWindow,	painter, modifier, TF::convert<TF::Types::Predefined, std::string>(structure.predefined));
}
	
TFHolder* TFCreator::loadTransferFunction(QMainWindow* mainWindow, const TF::Size domain){

	QString fileName = QFileDialog::getOpenFileName(
		(QWidget*)mainWindow,
		QObject::tr("Open Transfer Function"),
		QDir::currentPath(),
		QObject::tr("TF Files (*.tf *.xml)"));

	if (fileName.isEmpty()) return NULL;

	TFHolder* loaded = new TFHolder(mainWindow);

	QFile qFile(fileName);

	if (!qFile.open(QFile::ReadOnly | QFile::Text)) {
		QMessageBox::warning(
			(QWidget*)mainWindow,
			QObject::tr("Transfer Functions"),
			QObject::tr("Cannot read file %1:\n%2.").arg(fileName).arg(qFile.errorString()));

		qFile.close();
		return NULL;
	}
	if(!loaded->load(qFile))
	{ 
		QMessageBox::warning(
			(QWidget*)mainWindow,
			QObject::tr("TFXmlReader"),
			QObject::tr("Parse error in file %1").arg(fileName));

		qFile.close();
		return NULL;
	}
	qFile.close();

	return loaded;
}

TFCreator::TFCreator(QWidget* parent):
	QDialog(parent),
	ui_(new Ui::TFCreator),
	state_(Predefined),
	otherLayout_(NULL),
	functionSet_(false),
	painterSet_(false),
	modifierSet_(false){

	ui_->setupUi(this);
	ui_->nextButton->setEnabled(false);

	//---predefined---
	predefinedLayout_ = new QVBoxLayout();
	predefinedLayout_->setContentsMargins(10,10,10,10);
	predefinedLayout_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);	
	ui_->predefinedScroll->setLayout(predefinedLayout_);

	TF::Types::PredefinedTypesPtr predefined = TF::Types::getPredefinedTypes();
	
	for(TF::Types::PredefinedTypes::iterator it = predefined->begin(); it != predefined->end(); ++it)
	{
		TFPredefinedDialogButton* type = new TFPredefinedDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Predefined, std::string>(*it)));

		bool typeButtonConnected = QObject::connect(type, SIGNAL(Activated(TF::Types::Predefined)), this, SLOT(predefinedButton_clicked(TF::Types::Predefined)));
		tfAssert(typeButtonConnected);

		predefinedLayout_->addWidget(type);
	}
	//------

	//---functions---
	functionLayout_ = new QVBoxLayout();
	functionLayout_->setContentsMargins(10,10,10,10);
	functionLayout_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);	
	ui_->functionScroll->setLayout(functionLayout_);

	TF::Types::FunctionsPtr allFunctions = TF::Types::getAllFunctions();
	
	for(TF::Types::Functions::iterator it = allFunctions->begin(); it != allFunctions->end(); ++it)
	{
		TFFunctionDialogButton* type = new TFFunctionDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Function, std::string>(*it)));

		bool typeButtonConnected = QObject::connect( type, SIGNAL(Activated(TF::Types::Function)), this, SLOT(functionButton_clicked(TF::Types::Function)));
		tfAssert(typeButtonConnected);

		functionLayout_->addWidget(type);
	}
	//------

	//---other---
	if(otherLayout_) delete otherLayout_;
	otherLayout_ = new QVBoxLayout();
	otherLayout_->setContentsMargins(10,10,10,10);
	otherLayout_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);	
	ui_->otherScroll->setLayout(otherLayout_);
	//------

	setWindowTitle(QObject::tr("Transfer Function Creator"));
	show();
}

TFCreator::~TFCreator(){}

TF::Types::PredefinedStructure TFCreator::getResult(){

	return structure_;
}

void TFCreator::on_nextButton_clicked(){

	switch(state_)
	{
		case Predefined:
		{
			if(structure_.predefined == TF::Types::PredefinedCustom) setStateFunction_();
			else accept();
			break;
		}
		case Function:
		{
			setStatePainter_();
			break;
		}
		case Painter:
		{
			setStateModifier_();
			break;
		}
		case Modifier:
		{
			accept();
			break;
		}
	}
}

void TFCreator::on_backButton_clicked(){

	switch(state_)
	{
		case Modifier:
		{
			setStatePainter_();
			break;
		}
		case Painter:
		{
			setStateFunction_();
			break;
		}
		case Function:
		{
			setStatePredefined_();
			break;
		}
		case Predefined:
		{
			reject();
			break;
		}
	}
}

void TFCreator::setStatePredefined_(){
	
	ui_->description->setText(QObject::tr("Available Editors"));

	ui_->predefinedScroll->raise();
	ui_->backButton->setText(QObject::tr("Cancel"));
	ui_->nextButton->setEnabled(predefinedSet_);

	state_ = Predefined;
}

void TFCreator::setStateFunction_(){
	
	ui_->description->setText(QObject::tr("Available Functions"));

	ui_->functionScroll->raise();
	ui_->nextButton->setEnabled(functionSet_);

	state_ = Function;
}

void TFCreator::setStatePainter_(){

	ui_->description->setText(QObject::tr("Available Painters"));

	QLayoutItem* layoutIt;
	while(!otherLayout_->isEmpty())
	{
		layoutIt = otherLayout_->itemAt(0);
		otherLayout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
	}

	TF::Types::PaintersPtr allowedPainters = TF::Types::getAllowedPainters(structure_.function);

	TFPainterDialogButton* toActivate(NULL);
	TF::Types::Painters::iterator begin = allowedPainters->begin();
	TF::Types::Painters::iterator end = allowedPainters->end();
	for(TF::Types::Painters::iterator it = begin; it != end; ++it)
	{
		TFPainterDialogButton* type = new TFPainterDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Painter, std::string>(*it)));
		if(painterSet_ && structure_.painter == *it) toActivate = type;

		bool typeButtonConnected = QObject::connect( type, SIGNAL(Activated(TF::Types::Painter)), this, SLOT(painterButton_clicked(TF::Types::Painter)));
		tfAssert(typeButtonConnected);

		otherLayout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);

	ui_->otherScroll->raise();
	ui_->backButton->setText(QObject::tr("Back"));
	ui_->nextButton->setText(QObject::tr("Next"));
	ui_->nextButton->setEnabled(painterSet_);

	state_ = Painter;
}

void TFCreator::setStateModifier_(){

	ui_->description->setText(QObject::tr("Available Modifiers"));

	QLayoutItem* layoutIt;
	while(!otherLayout_->isEmpty())
	{
		layoutIt = otherLayout_->itemAt(0);
		otherLayout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
	}

	TF::Types::ModifiersPtr allowedModifiers = TF::Types::getAllowedModifiers(structure_.painter);

	TFModifierDialogButton* toActivate(NULL);
	TF::Types::Modifiers::iterator begin = allowedModifiers->begin();
	TF::Types::Modifiers::iterator end = allowedModifiers->end();
	for(TF::Types::Modifiers::iterator it = begin; it != end; ++it)
	{
		TFModifierDialogButton* type = new TFModifierDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Modifier, std::string>(*it)));
		if(modifierSet_ && structure_.modifier == *it)  toActivate = type;

		bool typeButtonConnected = QObject::connect( type, SIGNAL(Activated(TF::Types::Modifier)), this, SLOT(modifierButton_clicked(TF::Types::Modifier)));
		tfAssert(typeButtonConnected);

		otherLayout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);

	ui_->nextButton->setText(QObject::tr("Finish"));
	ui_->nextButton->setEnabled(modifierSet_);

	state_ = Modifier;
}

void TFCreator::predefinedButton_clicked(TF::Types::Predefined predefined){

	predefinedSet_ = true;
	if(predefined != structure_.predefined)
	{
		functionSet_ = false;
		painterSet_ = false;
		modifierSet_ = false;
	}

	structure_ = TF::Types::getPredefinedStructure(predefined);

	ui_->nextButton->setEnabled(true);
	if(structure_.predefined == TF::Types::PredefinedCustom) ui_->nextButton->setText(QObject::tr("Next"));
	else ui_->nextButton->setText(QObject::tr("Finish"));
}

void TFCreator::functionButton_clicked(TF::Types::Function function){

	functionSet_ = true;	
	if(function != structure_.function)
	{
		painterSet_ = false;
		modifierSet_ = false;
	}

	ui_->nextButton->setEnabled(true);

	structure_.function = function;
}

void TFCreator::painterButton_clicked(TF::Types::Painter painter){

	painterSet_ = true;
	if(painter != structure_.painter)
	{
		modifierSet_ = false;
	}
	
	ui_->nextButton->setEnabled(true);
	
	structure_.painter = painter;
}

void TFCreator::modifierButton_clicked(TF::Types::Modifier modifier){

	modifierSet_ = true;

	ui_->nextButton->setEnabled(true);

	structure_.modifier = modifier;
}
//------

} // namespace GUI
} // namespace M4D