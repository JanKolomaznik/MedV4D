#include "TFCreator.h"

#include <TFQtXmlReader.h>

#include <TFPalette.h>

#include <TFBasicHolder.h>

#include <TFSimpleModifier.h>
#include <TFPolygonModifier.h>
#include <TFViewModifier.h>
#include <TFCompositeModifier.h>

#include <TFRGBaFunction.h>
#include <TFHSVaFunction.h>

#include <TFGrayscaleAlphaPainter.h>
#include <TFRGBaPainter.h>
#include <TFHSVaPainter.h>

namespace M4D {
namespace GUI {

template<TF::Size dim>
typename TFAbstractFunction<dim>* TFCreator::createFunction_(){

	switch(structure_[mode_].function)
	{
		case TF::Types::FunctionRGBa1D:
		{
			return new typename TFRGBaFunction<dim>(dataStructure_);
		}
		case TF::Types::FunctionHSVa1D:
		{
			return new typename TFHSVaFunction<dim>(dataStructure_);
		}
	}

	tfAssert(!"Unknown function");
	return new typename TFRGBaFunction<dim>(dataStructure_);	//default
}

TFAbstractPainter* TFCreator::createPainter_(TFBasicHolder::Attributes& attributes){

	switch(structure_[mode_].painter)
	{
		case TF::Types::PainterGrayscaleAlpha1D:
		{
			return new TFGrayscaleAlphaPainter();
		}
		case TF::Types::PainterRGBa1D:
		{
			return new TFRGBaPainter();
		}
		case TF::Types::PainterHSVa1D:
		{
			return new TFHSVaPainter();
		}
	}
	
	tfAssert(!"Unknown painter!");
	return new TFRGBaPainter();
}

TFAbstractModifier* TFCreator::createModifier_(TFBasicHolder::Attributes& attributes){
	
	switch(structure_[mode_].modifier)
	{
		case TF::Types::ModifierSimple1D:
		{
			attributes.insert(TFBasicHolder::Dimension1);
			return new TFSimpleModifier(
				TFAbstractFunction<TF_DIMENSION_1>::Ptr(createFunction_<TF_DIMENSION_1>()),
				TFSimplePainter::Ptr(dynamic_cast<TFSimplePainter*>(createPainter_(attributes)))
			);
		}
		case TF::Types::ModifierPolygon1D:
		{
			attributes.insert(TFBasicHolder::Dimension1);
			return new TFPolygonModifier(
				TFAbstractFunction<TF_DIMENSION_1>::Ptr(createFunction_<TF_DIMENSION_1>()),
				TFSimplePainter::Ptr(dynamic_cast<TFSimplePainter*>(createPainter_(attributes)))
			);
		}
		case TF::Types::ModifierComposite1D:
		{
			attributes.insert(TFBasicHolder::Dimension1);
			attributes.insert(TFBasicHolder::Composition);
			return new TFCompositeModifier(
				TFAbstractFunction<TF_DIMENSION_1>::Ptr(createFunction_<TF_DIMENSION_1>()),
				TFSimplePainter::Ptr(dynamic_cast<TFSimplePainter*>(createPainter_(attributes))),
				palette_
			);
		}
	}

	tfAssert(!"Unknown modifier!");	
	attributes.insert(TFBasicHolder::Dimension1);
	return new TFSimpleModifier(
		TFAbstractFunction<TF_DIMENSION_1>::Ptr(createFunction_<TF_DIMENSION_1>()),
		TFSimplePainter::Ptr(dynamic_cast<TFSimplePainter*>(createPainter_(attributes)))
	);	//default

}

TFBasicHolder* TFCreator::createHolder_(){

	switch(structure_[mode_].holder)
	{
		case TF::Types::HolderBasic:
		{
			TFBasicHolder::Attributes attributes;
			TFAbstractModifier::Ptr modifier(createModifier_(attributes));

			return new TFBasicHolder(modifier, structure_[mode_], attributes, name_);
		}
	}

	tfAssert(!"Unknown holder");
	return NULL;
}


TFCreator::TFCreator(QMainWindow* mainWindow, TFPalette* palette):
	QDialog(mainWindow),
	ui_(new Ui::TFCreator),
	reader_(new TF::QtXmlReader),
	mainWindow_(mainWindow),
	palette_(palette),
	dataStructure_(std::vector<TF::Size>(1,TFAbstractFunction<1>::defaultDomain)),
	layout_(new QVBoxLayout()),
	state_(ModeSelection),
	mode_(CreatePredefined),
	holderSet_(false),
	predefinedSet_(false),
	functionSet_(false),
	painterSet_(false),
	modifierSet_(false){

	structure_[CreateCustom].predefined = TF::Types::PredefinedCustom;

	ui_->setupUi(this);

	layout_->setContentsMargins(10,10,10,10);
	layout_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);	
	ui_->scrollArea->setLayout(layout_);

	predefinedRadio_ = new QRadioButton("Select predefined");	
	bool predefinedRadioConnected = QObject::connect(predefinedRadio_, SIGNAL(clicked()), this, SLOT(mode_clicked()));
	tfAssert(predefinedRadioConnected);
	predefinedRadio_->setChecked(true);

	customRadio_ = new QRadioButton("Create custom");
	bool customRadioConnected = QObject::connect(customRadio_, SIGNAL(clicked()), this, SLOT(mode_clicked()));
	tfAssert(customRadioConnected);

	loadRadio_ = new QRadioButton("Load");
	bool loadRadioConnected = QObject::connect(loadRadio_, SIGNAL(clicked()), this, SLOT(mode_clicked()));
	tfAssert(loadRadioConnected);
}

TFCreator::~TFCreator(){

	delete reader_;
	delete ui_;
}

TFBasicHolder* TFCreator::createTransferFunction(){

	clearLayout_(state_ != ModeSelection);
	setStateModeSelection_();

	exec();

	if(result() == QDialog::Rejected) return NULL;
	
	if(mode_ == CreateLoaded) return loadTransferFunction_();

	name_ = TF::convert<TF::Types::Predefined, std::string>(structure_[mode_].predefined);
	return createHolder_();	
}
	
TFBasicHolder* TFCreator::loadTransferFunction_(){

	TFBasicHolder* loaded = NULL;

	QString fileName = QFileDialog::getOpenFileName(
		(QWidget*)mainWindow_,
		"Open Transfer Function",
		QDir::currentPath(),
		QObject::tr("TF Files (*.tf *.xml)"));

	QMessageBox errorMessage(QMessageBox::Critical, "Error", "", QMessageBox::Ok);
	errorMessage.setDefaultButton(QMessageBox::Ok);

	if(!reader_->begin(fileName.toStdString()))
	{
		errorMessage.setText(QString::fromStdString(reader_->errorMessage()));
		errorMessage.exec();
		return loaded;
	}

	bool error;
	loaded = load_(reader_, error);

	if(!loaded)
	{ 
		errorMessage.setText(QObject::tr("File %1 is corrupted").arg(fileName));
		errorMessage.exec();
		return loaded;
	}
	else if(error)
	{
		QMessageBox::warning((QWidget*)mainWindow_,
			QObject::tr("Error"),
			QObject::tr("Error while reading additional data.\nSome settings are set to default."));
	}

	#ifndef TF_NDEBUG
		std::cout << "Loading finished." << std::endl;
	#endif

	return loaded;
}

TFBasicHolder* TFCreator::load_(TF::XmlReaderInterface* reader, bool& sideError){

	#ifndef TF_NDEBUG
		std::cout << "Loading editor..." << std::endl;
	#endif

	TFBasicHolder* loaded = NULL;
	std::string name;
	bool ok = false;
	if(reader->readElement("Editor"))
	{		
		name_ = reader->readAttribute("Name");

		structure_[mode_].predefined = TF::convert<std::string, TF::Types::Predefined>(
			reader->readAttribute("Predefined"));

		structure_[mode_].holder = TF::convert<std::string, TF::Types::Holder>(
			reader->readAttribute("Holder"));

		structure_[mode_].function = TF::convert<std::string, TF::Types::Function>(
			reader->readAttribute("Function"));

		structure_[mode_].painter = TF::convert<std::string, TF::Types::Painter>(
			reader->readAttribute("Painter"));

		structure_[mode_].modifier = TF::convert<std::string, TF::Types::Modifier>(
			reader->readAttribute("Modifier"));

		loaded = createHolder_();
		if(loaded) ok = loaded->loadData(reader, sideError);
	}
	if(!ok)
	{
		if(loaded) delete loaded;
		return NULL;
	}
	return loaded;
}

void TFCreator::setDataStructure(const std::vector<TF::Size>& dataStructure){

	dataStructure_ = dataStructure;
}

void TFCreator::on_nextButton_clicked(){

	switch(state_)
	{
		case ModeSelection:
		{
			switch(mode_)
			{
				case CreatePredefined:
				{
					clearLayout_(false);
					setStatePredefined_();
					break;
				}
				case CreateCustom:
				{
					clearLayout_(false);
					setStateHolder_();
					break;
				}
				case CreateLoaded:
				{
					accept();
					break;
				}
			}
			break;
		}
		case Predefined:
		{
			accept();
			break;
		}
		case Holder:
		{
			clearLayout_();
			setStateModifier_();
			break;
		}
		case Modifier:
		{
			clearLayout_();
			setStateFunction_();
			break;
		}
		case Function:
		{
			clearLayout_();
			setStatePainter_();
			break;
		}
		case Painter:
		{
			accept();
			break;
		}
	}
}

void TFCreator::on_backButton_clicked(){

	switch(state_)
	{
		case Painter:
		{
			clearLayout_();
			setStateFunction_();
			break;
		}
		case Function:
		{
			clearLayout_();
			setStateModifier_();
			break;
		}
		case Modifier:
		{
			clearLayout_();
			setStateHolder_();
			break;
		}
		case Holder:
		{
			clearLayout_();
			setStateModeSelection_();
			break;
		}
		case Predefined:
		{
			clearLayout_();
			setStateModeSelection_();
			break;
		}
		case ModeSelection:
		{
			reject();
			break;
		}
	}
}

void TFCreator::clearLayout_(bool deleteItems){

	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
		if(deleteItems) delete layoutIt;
	}
}

void TFCreator::setStateModeSelection_(){

	layout_->addWidget(predefinedRadio_);
	predefinedRadio_->show();
	predefinedRadio_->setChecked(mode_ == CreatePredefined);

	layout_->addWidget(customRadio_);
	customRadio_->show();
	customRadio_->setChecked(mode_ == CreateCustom);

	layout_->addWidget(loadRadio_);
	loadRadio_->show();
	loadRadio_->setChecked(mode_ == CreateLoaded);

	ui_->description->setText(QObject::tr("Create Editor"));
	ui_->backButton->setText(QObject::tr("Cancel"));
	ui_->nextButton->setText(QObject::tr("Next"));
	ui_->nextButton->setEnabled(true);

	state_ = ModeSelection;
}

void TFCreator::setStatePredefined_(){
	
	TF::Types::PredefinedTypes allPredefined = TF::Types::getPredefinedTypes();
	
	TFPredefinedDialogButton* toActivate = NULL;
	TF::Types::PredefinedTypes::iterator begin = allPredefined.begin();
	TF::Types::PredefinedTypes::iterator end = allPredefined.end();
	for(TF::Types::PredefinedTypes::iterator it = begin; it != end; ++it)
	{
		TFPredefinedDialogButton* type = new TFPredefinedDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Predefined, std::string>(*it)));
		if(predefinedSet_ && structure_[mode_].predefined == *it) toActivate = type;

		bool typeButtonConnected = QObject::connect(type, SIGNAL(Activated(TF::Types::Predefined)), this, SLOT(predefinedButton_clicked(TF::Types::Predefined)));
		tfAssert(typeButtonConnected);

		layout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);
	
	ui_->description->setText(QObject::tr("Predefined Editors"));
	ui_->backButton->setText(QObject::tr("Back"));
	ui_->nextButton->setText(QObject::tr("Finish"));
	ui_->nextButton->setEnabled(predefinedSet_);

	state_ = Predefined;
}

void TFCreator::setStateHolder_(){
	
	TF::Types::Holders allHolders = TF::Types::getAllHolders();
	
	TFHolderDialogButton* toActivate = NULL;
	TF::Types::Holders::iterator begin = allHolders.begin();
	TF::Types::Holders::iterator end = allHolders.end();
	for(TF::Types::Holders::iterator it = begin; it != end; ++it)
	{
		TFHolderDialogButton* type = new TFHolderDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Holder, std::string>(*it)));
		if(holderSet_ && structure_[mode_].holder == *it) toActivate = type;

		bool typeButtonConnected = QObject::connect(type, SIGNAL(Activated(TF::Types::Holder)), this, SLOT(holderButton_clicked(TF::Types::Holder)));
		tfAssert(typeButtonConnected);

		layout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);
	
	ui_->description->setText(QObject::tr("Available Holders"));
	ui_->backButton->setText(QObject::tr("Back"));
	ui_->nextButton->setEnabled(holderSet_);

	state_ = Holder;
}

void TFCreator::setStateModifier_(){

	TF::Types::Modifiers allowedModifiers = TF::Types::getAllowedModifiers(structure_[mode_].holder);

	TFModifierDialogButton* toActivate = NULL;
	TF::Types::Modifiers::iterator begin = allowedModifiers.begin();
	TF::Types::Modifiers::iterator end = allowedModifiers.end();
	for(TF::Types::Modifiers::iterator it = begin; it != end; ++it)
	{
		TFModifierDialogButton* type = new TFModifierDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Modifier, std::string>(*it)));
		if(modifierSet_ && structure_[mode_].modifier == *it)  toActivate = type;

		bool typeButtonConnected = QObject::connect( type, SIGNAL(Activated(TF::Types::Modifier)), this, SLOT(modifierButton_clicked(TF::Types::Modifier)));
		tfAssert(typeButtonConnected);

		layout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);

	ui_->description->setText(QObject::tr("Available Modifiers"));
	ui_->nextButton->setEnabled(modifierSet_);

	state_ = Modifier;
}

void TFCreator::setStateFunction_(){

	TF::Types::Functions allowedFunctions = TF::Types::getAllowedFunctions(structure_[mode_].modifier);
	
	TFFunctionDialogButton* toActivate = NULL;
	TF::Types::Functions::iterator begin = allowedFunctions.begin();
	TF::Types::Functions::iterator end = allowedFunctions.end();
	for(TF::Types::Functions::iterator it = begin; it != end; ++it)
	{
		TFFunctionDialogButton* type = new TFFunctionDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Function, std::string>(*it)));
		if(functionSet_ && structure_[mode_].function == *it) toActivate = type;

		bool typeButtonConnected = QObject::connect( type, SIGNAL(Activated(TF::Types::Function)), this, SLOT(functionButton_clicked(TF::Types::Function)));
		tfAssert(typeButtonConnected);

		layout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);

	ui_->description->setText(QObject::tr("Available Functions"));
	ui_->nextButton->setText(QObject::tr("Next"));
	ui_->nextButton->setEnabled(functionSet_);

	state_ = Function;
}

void TFCreator::setStatePainter_(){

	TF::Types::Painters allowedPainters = TF::Types::getAllowedPainters(structure_[mode_].function);

	TFPainterDialogButton* toActivate = NULL;
	TF::Types::Painters::iterator begin = allowedPainters.begin();
	TF::Types::Painters::iterator end = allowedPainters.end();
	for(TF::Types::Painters::iterator it = begin; it != end; ++it)
	{
		TFPainterDialogButton* type = new TFPainterDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Painter, std::string>(*it)));
		if(painterSet_ && structure_[mode_].painter == *it) toActivate = type;

		bool typeButtonConnected = QObject::connect( type, SIGNAL(Activated(TF::Types::Painter)), this, SLOT(painterButton_clicked(TF::Types::Painter)));
		tfAssert(typeButtonConnected);

		layout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);

	ui_->description->setText(QObject::tr("Available Painters"));
	ui_->nextButton->setText(QObject::tr("Finish"));
	ui_->nextButton->setEnabled(painterSet_);

	state_ = Painter;
}

void TFCreator::mode_clicked(){

	if(predefinedRadio_->isChecked())
	{
		mode_ = CreatePredefined;
		if(predefinedSet_)
		{
			ui_->predefinedValue->setText(QString::fromStdString(TF::convert<TF::Types::Predefined, std::string>(
				structure_[mode_].predefined
			)));
			ui_->holderValue->setText(QString::fromStdString(TF::convert<TF::Types::Holder, std::string>(
				structure_[mode_].holder
			)));
			ui_->modifierValue->setText(QString::fromStdString(TF::convert<TF::Types::Modifier, std::string>(
				structure_[mode_].modifier
			)));
			ui_->functionValue->setText(QString::fromStdString(TF::convert<TF::Types::Function, std::string>(
				structure_[mode_].function
			)));
			ui_->painterValue->setText(QString::fromStdString(TF::convert<TF::Types::Painter, std::string>(
				structure_[mode_].painter
			)));
		}
		else
		{
			ui_->predefinedValue->setText("Predefined");
			ui_->holderValue->setText("");
			ui_->modifierValue->setText("");
			ui_->functionValue->setText("");
			ui_->painterValue->setText("");
		}
		
	}
	if(customRadio_->isChecked())
	{
		mode_ = CreateCustom;
		ui_->predefinedValue->setText("Custom");

		if(holderSet_)
		{
			ui_->holderValue->setText(QString::fromStdString(TF::convert<TF::Types::Holder, std::string>(
				structure_[mode_].holder
			)));
		}
		else ui_->holderValue->setText("");

		if(modifierSet_)
		{
			ui_->modifierValue->setText(QString::fromStdString(TF::convert<TF::Types::Modifier, std::string>(
				structure_[mode_].modifier
			)));
		}
		else ui_->modifierValue->setText("");

		if(functionSet_)
		{
			ui_->functionValue->setText(QString::fromStdString(TF::convert<TF::Types::Function, std::string>(
				structure_[mode_].function
			)));
		}
		else ui_->functionValue->setText("");

		if(painterSet_)
		{
			ui_->painterValue->setText(QString::fromStdString(TF::convert<TF::Types::Painter, std::string>(
				structure_[mode_].painter
			)));
		}
		else ui_->painterValue->setText("");
	}
	if(loadRadio_->isChecked())
	{
		mode_ = CreateLoaded;
		ui_->predefinedValue->setText("Loaded");
		ui_->holderValue->setText("");
		ui_->modifierValue->setText("");
		ui_->functionValue->setText("");
		ui_->painterValue->setText("");
		ui_->nextButton->setText(QObject::tr("Finish"));
	}
	else ui_->nextButton->setText(QObject::tr("Next"));
}

void TFCreator::predefinedButton_clicked(TF::Types::Predefined predefined){

	predefinedSet_ = true;
	ui_->nextButton->setEnabled(true);

	structure_[mode_] = TF::Types::getPredefinedStructure(predefined);

	ui_->predefinedValue->setText(QString::fromStdString(TF::convert<TF::Types::Predefined, std::string>(
		structure_[mode_].predefined
	)));
	ui_->holderValue->setText(QString::fromStdString(TF::convert<TF::Types::Holder, std::string>(
		structure_[mode_].holder
	)));
	ui_->modifierValue->setText(QString::fromStdString(TF::convert<TF::Types::Modifier, std::string>(
		structure_[mode_].modifier
	)));
	ui_->functionValue->setText(QString::fromStdString(TF::convert<TF::Types::Function, std::string>(
		structure_[mode_].function
	)));
	ui_->painterValue->setText(QString::fromStdString(TF::convert<TF::Types::Painter, std::string>(
		structure_[mode_].painter
	)));
}

void TFCreator::holderButton_clicked(TF::Types::Holder holder){

	holderSet_ = true;	
	ui_->nextButton->setEnabled(true);

	if(holder != structure_[mode_].holder)
	{
		modifierSet_ = false;
		functionSet_ = false;
		painterSet_ = false;
		ui_->modifierValue->setText("");
		ui_->functionValue->setText("");
		ui_->painterValue->setText("");
	}

	structure_[mode_].holder = holder;
	ui_->holderValue->setText(QString::fromStdString(TF::convert<TF::Types::Holder, std::string>(
		structure_[mode_].holder
	)));
}

void TFCreator::modifierButton_clicked(TF::Types::Modifier modifier){

	modifierSet_ = true;
	ui_->nextButton->setEnabled(true);

	if(modifier != structure_[mode_].modifier)
	{
		functionSet_ = false;
		painterSet_ = false;
		ui_->functionValue->setText("");
		ui_->painterValue->setText("");
	}

	structure_[mode_].modifier = modifier;
	ui_->modifierValue->setText(QString::fromStdString(TF::convert<TF::Types::Modifier, std::string>(
		structure_[mode_].modifier
	)));
}

void TFCreator::functionButton_clicked(TF::Types::Function function){

	functionSet_ = true;	
	ui_->nextButton->setEnabled(true);

	if(function != structure_[mode_].function)
	{
		painterSet_ = false;
		ui_->painterValue->setText("");
	}

	structure_[mode_].function = function;
	ui_->functionValue->setText(QString::fromStdString(TF::convert<TF::Types::Function, std::string>(
		structure_[mode_].function
	)));
}

void TFCreator::painterButton_clicked(TF::Types::Painter painter){

	painterSet_ = true;	
	ui_->nextButton->setEnabled(true);
	
	structure_[mode_].painter = painter;
	ui_->painterValue->setText(QString::fromStdString(TF::convert<TF::Types::Painter, std::string>(
		structure_[mode_].painter
	)));
}

} // namespace GUI
} // namespace M4D