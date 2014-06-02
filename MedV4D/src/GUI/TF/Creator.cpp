#include "MedV4D/GUI/TF/Creator.h"

#include "MedV4D/GUI/TF/QtXmlReader.h"

#include "MedV4D/GUI/TF/Palette.h"

#include "MedV4D/GUI/TF/EditorGUI.h"

#include "MedV4D/GUI/TF/Modifier1D.h"
#include "MedV4D/GUI/TF/PolygonModifier.h"
#include "MedV4D/GUI/TF/CompositeModifier.h"

#include "MedV4D/GUI/TF/RGBaFunction.h"
#include "MedV4D/GUI/TF/HSVaFunction.h"

#include "MedV4D/GUI/TF/GrayscaleAlphaPainter1D.h"
#include "MedV4D/GUI/TF/RGBaPainter1D.h"
#include "MedV4D/GUI/TF/HSVaPainter1D.h"

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Debug.h"

namespace M4D {
namespace GUI {

//---creation-methods---

template<TF::Size dim>
AbstractFunction<dim>* Creator::createFunction_(){

	switch(structure_[mode_].function)
	{
		case TF::Types::FunctionRGBa:
		{
			return new RGBaFunction<dim>(dataStructure_);
		}
		case TF::Types::FunctionHSVa:
		{
			return new HSVaFunction<dim>(dataStructure_);
		}
	}

	tfAssert(!"Unknown function");
	return new RGBaFunction<dim>(dataStructure_);	//default
}

AbstractPainter* Creator::createPainter_(Editor::Attributes& attributes){

	switch(structure_[mode_].painter)
	{
		case TF::Types::PainterGrayscaleAlpha1D:
		{
			return new GrayscaleAlphaPainter1D();
		}
		case TF::Types::PainterRGBa1D:
		{
			return new RGBaPainter1D();
		}
		case TF::Types::PainterHSVa1D:
		{
			return new HSVaPainter1D();
		}
	}
	
	tfAssert(!"Unknown painter!");
	return new RGBaPainter1D();
}

AbstractModifier* Creator::createModifier_(Editor::Attributes& attributes)
{
	//D_BLOCK_COMMENT( TO_STRING(__FUNCTION__ << " entered"), TO_STRING(__FUNCTION__ << " leaved") );
	
	switch(structure_[mode_].modifier)
	{
		case TF::Types::ModifierSimple1D:
		{
			tfAssert(structure_[mode_].dimension == TF::Types::Dimension1);
			return new Modifier1D(
				TransferFunctionInterface::Ptr(createFunction_<TF_DIMENSION_1>()),
				Painter1D::Ptr(dynamic_cast<Painter1D*>(createPainter_(attributes)))
			);
		}
		case TF::Types::ModifierPolygon1D:
		{
			tfAssert(structure_[mode_].dimension == TF::Types::Dimension1);
			return new PolygonModifier(
				TransferFunctionInterface::Ptr(createFunction_<TF_DIMENSION_1>()),
				Painter1D::Ptr(dynamic_cast<Painter1D*>(createPainter_(attributes)))
			);
		}
		case TF::Types::ModifierComposite1D:
		{
			tfAssert(structure_[mode_].dimension == TF::Types::Dimension1);
			attributes.insert(Editor::Composition);
			TransferFunctionInterface::Ptr func = TransferFunctionInterface::Ptr(createFunction_<TF_DIMENSION_1>());
			Painter1D::Ptr painter = Painter1D::Ptr(dynamic_cast<Painter1D*>(createPainter_(attributes)));
			CompositeModifier *modifier = new CompositeModifier( func, painter, palette_ );
			return modifier; 
		}
	}

	tfAssert(!"Unknown modifier!");	
	return new Modifier1D(
		TransferFunctionInterface::Ptr(createFunction_<TF_DIMENSION_1>()),
		Painter1D::Ptr(dynamic_cast<Painter1D*>(createPainter_(attributes)))
	);	//default

}

Editor* Creator::createEditor_()
{
	//D_BLOCK_COMMENT( TO_STRING(__FUNCTION__ << " entered"), TO_STRING(__FUNCTION__ << " leaved") );
	Editor::Attributes attributes;
	AbstractModifier::Ptr modifier(createModifier_(attributes));

	return new EditorGUI(modifier, structure_[mode_], attributes, name_);
}

//---creation-dialog---

Creator::Creator(QMainWindow* mainWindow, Palette* palette, const std::vector<TF::Size>& dataStructure):
	QDialog(mainWindow),
	ui_(new Ui::Creator),
	layout_(new QVBoxLayout()),
	reader_(new TF::QtXmlReader),
	state_(ModeSelection),
	mode_(CreatePredefined),
	predefinedSet_(false),
	dimensionSet_(false),
	functionSet_(false),
	painterSet_(false),
	modifierSet_(false),
	mainWindow_(mainWindow),
	palette_(palette),
	dataStructure_(dataStructure)
{

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

Creator::~Creator(){

	delete reader_;
	delete ui_;
}

Editor* Creator::createEditor(){

	clearLayout_(state_ != ModeSelection);
	setStateModeSelection_();

	exec();

	if(result() == QDialog::Rejected) return NULL;
	
	if(mode_ == CreateLoaded) return loadEditor_();

	name_ = TF::convert<TF::Types::Predefined, std::string>(structure_[mode_].predefined);
	return createEditor_();	
}

Editor*
Creator::loadEditorFromFile( QString fileName )
{
	Editor* loaded = NULL; //TODO improve

	if(fileName.isEmpty()) return loaded;

	if(!reader_->begin(fileName.toLocal8Bit().data()))
	{
		LOG_ERR( reader_->errorMessage());
		return loaded;
	}

	bool error;
	loaded = load_(reader_, error);

	if(!loaded)
	{ 
		LOG_ERR( "File \""<< fileName.toLocal8Bit().data() << " is corrupted." );
	}else if(error)
	{
		LOG_ERR("Transfer Function Loading Error");
		LOG_ERR("Error while reading additional data.\nSome settings are set to default.");
	}
	reader_->end();
	return loaded;
}
	
Editor* Creator::loadEditor_(){

	Editor* loaded = NULL;

	QString fileName = QFileDialog::getOpenFileName(
		(QWidget*)mainWindow_,
		"Load Transfer Function Editor",
		QDir::currentPath(),
		QObject::tr("TF Editor Files (*.tfe)"));

	if(fileName.isEmpty()) return loaded;

	QMessageBox errorMessage(QMessageBox::Critical, "Transfer Function Loading Error", "", QMessageBox::Ok);
	errorMessage.setDefaultButton(QMessageBox::Ok);

	if(!reader_->begin(fileName.toLocal8Bit().data()))
	{
		errorMessage.setText(QString::fromStdString(reader_->errorMessage()));
		errorMessage.exec();
		return loaded;
	}

	bool error;
	loaded = load_(reader_, error);

	if(!loaded)
	{ 
		errorMessage.setText(QObject::tr("File \"%1\" is corrupted.").arg(fileName));
		errorMessage.exec();
	}
	else if(error)
	{
		QMessageBox::warning((QWidget*)mainWindow_,
			QObject::tr("Transfer Function Loading Error"),
			QObject::tr("Error while reading additional data.\nSome settings are set to default."));
	}

	#ifndef TF_NDEBUG
		std::cout << "Loading finished." << std::endl;
	#endif

	reader_->end();
	return loaded;
}

Editor* Creator::load_(TF::XmlReaderInterface* reader, bool& sideError){

	#ifndef TF_NDEBUG
		std::cout << "Loading editor..." << std::endl;
	#endif

	Editor* loaded = NULL;
	std::string name;
	bool ok = false;
	if(reader->readElement("Editor"))
	{		
		name_ = reader->readAttribute("Name");

		structure_[mode_].predefined = TF::convert<std::string, TF::Types::Predefined>(
			reader->readAttribute("Predefined"));

		structure_[mode_].dimension = TF::convert<std::string, TF::Types::Dimension>(
			reader->readAttribute("Dimension"));

		structure_[mode_].function = TF::convert<std::string, TF::Types::Function>(
			reader->readAttribute("Function"));

		structure_[mode_].painter = TF::convert<std::string, TF::Types::Painter>(
			reader->readAttribute("Painter"));

		structure_[mode_].modifier = TF::convert<std::string, TF::Types::Modifier>(
			reader->readAttribute("Modifier"));

		loaded = createEditor_();
		if(loaded) ok = loaded->load(reader, sideError);
	}
	if(!ok)
	{
		if(loaded) delete loaded;
		return NULL;
	}
	return loaded;
}

void Creator::setDataStructure(const std::vector<TF::Size>& dataStructure){

	dataStructure_ = dataStructure;
}

void Creator::on_nextButton_clicked(){

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
					setStateDimension_();
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
		case Dimension:
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

void Creator::on_backButton_clicked(){

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
			setStateDimension_();
			break;
		}
		case Dimension:
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

void Creator::clearLayout_(bool deleteItems){

	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
		if(deleteItems) delete layoutIt;
	}
}

void Creator::setStateModeSelection_(){

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

void Creator::setStatePredefined_(){
	
	TF::Types::PredefinedTypes allPredefined = TF::Types::getPredefinedTypes();
	
	PredefinedDialogButton* toActivate = NULL;
	TF::Types::PredefinedTypes::iterator begin = allPredefined.begin();
	TF::Types::PredefinedTypes::iterator end = allPredefined.end();
	for(TF::Types::PredefinedTypes::iterator it = begin; it != end; ++it)
	{
		PredefinedDialogButton* type = new PredefinedDialogButton(*it);
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

void Creator::setStateDimension_(){
	
	TF::Types::Dimensions allDimensions = TF::Types::getSupportedDimensions();
	
	DimensionDialogButton* toActivate = NULL;
	TF::Types::Dimensions::iterator begin = allDimensions.begin();
	TF::Types::Dimensions::iterator end = allDimensions.end();
	for(TF::Types::Dimensions::iterator it = begin; it != end; ++it)
	{
		DimensionDialogButton* type = new DimensionDialogButton(*it);
		type->setText(QString::fromStdString(TF::convert<TF::Types::Dimension, std::string>(*it)));
		if(dimensionSet_ && structure_[mode_].dimension == *it) toActivate = type;

		bool typeButtonConnected = QObject::connect(type, SIGNAL(Activated(TF::Types::Dimension)), this, SLOT(dimensionButton_clicked(TF::Types::Dimension)));
		tfAssert(typeButtonConnected);

		layout_->addWidget(type);
	}
	if(toActivate) toActivate->setChecked(true);
	
	ui_->description->setText(QObject::tr("Available Dimensions"));
	ui_->backButton->setText(QObject::tr("Back"));
	ui_->nextButton->setEnabled(dimensionSet_);

	state_ = Dimension;
}

void Creator::setStateModifier_(){

	TF::Types::Modifiers allowedModifiers = TF::Types::getAllowedModifiers(structure_[mode_].dimension);

	ModifierDialogButton* toActivate = NULL;
	TF::Types::Modifiers::iterator begin = allowedModifiers.begin();
	TF::Types::Modifiers::iterator end = allowedModifiers.end();
	for(TF::Types::Modifiers::iterator it = begin; it != end; ++it)
	{
		ModifierDialogButton* type = new ModifierDialogButton(*it);
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

void Creator::setStateFunction_(){

	TF::Types::Functions allowedFunctions = TF::Types::getAllowedFunctions(structure_[mode_].modifier);
	
	FunctionDialogButton* toActivate = NULL;
	TF::Types::Functions::iterator begin = allowedFunctions.begin();
	TF::Types::Functions::iterator end = allowedFunctions.end();
	for(TF::Types::Functions::iterator it = begin; it != end; ++it)
	{
		FunctionDialogButton* type = new FunctionDialogButton(*it);
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

void Creator::setStatePainter_(){

	TF::Types::Painters allowedPainters = TF::Types::getAllowedPainters(structure_[mode_].function);

	PainterDialogButton* toActivate = NULL;
	TF::Types::Painters::iterator begin = allowedPainters.begin();
	TF::Types::Painters::iterator end = allowedPainters.end();
	for(TF::Types::Painters::iterator it = begin; it != end; ++it)
	{
		PainterDialogButton* type = new PainterDialogButton(*it);
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

void Creator::mode_clicked(){

	if(predefinedRadio_->isChecked())
	{
		mode_ = CreatePredefined;
		if(predefinedSet_)
		{
			ui_->predefinedValue->setText(QString::fromStdString(TF::convert<TF::Types::Predefined, std::string>(
				structure_[mode_].predefined
			)));
			ui_->dimensionValue->setText(QString::fromStdString(TF::convert<TF::Types::Dimension, std::string>(
				structure_[mode_].dimension
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
			ui_->dimensionValue->setText("");
			ui_->modifierValue->setText("");
			ui_->functionValue->setText("");
			ui_->painterValue->setText("");
		}
		
	}
	if(customRadio_->isChecked())
	{
		mode_ = CreateCustom;
		ui_->predefinedValue->setText("Custom");

		if(dimensionSet_)
		{
			ui_->dimensionValue->setText(QString::fromStdString(TF::convert<TF::Types::Dimension, std::string>(
				structure_[mode_].dimension
			)));
		}
		else ui_->dimensionValue->setText("");

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
		ui_->dimensionValue->setText("");
		ui_->modifierValue->setText("");
		ui_->functionValue->setText("");
		ui_->painterValue->setText("");
		ui_->nextButton->setText(QObject::tr("Finish"));
	}
	else ui_->nextButton->setText(QObject::tr("Next"));
}

void Creator::predefinedButton_clicked(TF::Types::Predefined predefined){

	predefinedSet_ = true;
	ui_->nextButton->setEnabled(true);

	structure_[mode_] = TF::Types::getPredefinedStructure(predefined);

	ui_->predefinedValue->setText(QString::fromStdString(TF::convert<TF::Types::Predefined, std::string>(
		structure_[mode_].predefined
	)));
	ui_->dimensionValue->setText(QString::fromStdString(TF::convert<TF::Types::Dimension, std::string>(
		structure_[mode_].dimension
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

void Creator::dimensionButton_clicked(TF::Types::Dimension dimension){

	dimensionSet_ = true;	
	ui_->nextButton->setEnabled(true);

	if(dimension != structure_[mode_].dimension)
	{
		modifierSet_ = false;
		functionSet_ = false;
		painterSet_ = false;
		ui_->modifierValue->setText("");
		ui_->functionValue->setText("");
		ui_->painterValue->setText("");
	}

	structure_[mode_].dimension = dimension;
	ui_->dimensionValue->setText(QString::fromStdString(TF::convert<TF::Types::Dimension, std::string>(
		structure_[mode_].dimension
	)));
}

void Creator::modifierButton_clicked(TF::Types::Modifier modifier){

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

void Creator::functionButton_clicked(TF::Types::Function function){

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

void Creator::painterButton_clicked(TF::Types::Painter painter){

	painterSet_ = true;	
	ui_->nextButton->setEnabled(true);
	
	structure_[mode_].painter = painter;
	ui_->painterValue->setText(QString::fromStdString(TF::convert<TF::Types::Painter, std::string>(
		structure_[mode_].painter
	)));
}

} // namespace GUI
} // namespace M4D
