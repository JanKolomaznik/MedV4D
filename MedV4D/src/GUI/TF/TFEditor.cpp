#include "MedV4D/GUI/TF/TFEditor.h"

#include "MedV4D/GUI/TF/TFQtXmlWriter.h"
#include "MedV4D/GUI/TF/TFQtXmlReader.h"

#include "MedV4D/GUI/TF/TFCreator.h"

namespace M4D{
namespace GUI{

TFEditor::TFEditor(TFAbstractModifier::Ptr modifier,
				   TF::Types::Structure structure,
				   Attributes attributes,
				   std::string name):
	editorDock_(NULL),
	toolsDock_(NULL),
	writer_(new TF::QtXmlWriter),
	modifier_(modifier),
	index_(0),
	name_(name),
	structure_(structure),
	attributes_(attributes)
{
}

TFEditor::~TFEditor()
{

	if(toolsDock_) toolsDock_->close();
	delete writer_;
}
	
bool 
TFEditor::save()
{
	
	if(fileName_.isEmpty())
	{
		fileName_ = QDir::currentPath().append(QString::fromStdString("/" + name_));

		fileName_ = QFileDialog::getSaveFileName(this,
			QObject::tr("Save Transfer Function Editor"),
			fileName_,
			QObject::tr("TF Editor Files (*.tfe)"));

		if (fileName_.isEmpty()) return false;
	}
	
	if (!writer_->begin(fileName_.toLocal8Bit().data()))
	{
		QMessageBox errorMessage(QMessageBox::Critical,
			"Error",
			QString::fromStdString(writer_->errorMessage()),
			QMessageBox::Ok
		);
		errorMessage.setDefaultButton(QMessageBox::Ok);
		errorMessage.exec();
		return false;
	}

	writer_->writeDTD("TransferFunctionEditorFile");

	writer_->writeStartElement("Editor");

		writer_->writeAttribute("Name", name_);
		writer_->writeAttribute("Predefined",
			TF::convert<TF::Types::Predefined, std::string>(structure_.predefined));
		writer_->writeAttribute("Dimension",
			TF::convert<TF::Types::Dimension, std::string>(structure_.dimension));
		writer_->writeAttribute("Function",
			TF::convert<TF::Types::Function, std::string>(structure_.function));
		writer_->writeAttribute("Painter",
			TF::convert<TF::Types::Painter, std::string>(structure_.painter));
		writer_->writeAttribute("Modifier",
			TF::convert<TF::Types::Modifier, std::string>(structure_.modifier));

	saveSettings_(writer_);

	modifier_->save(writer_);

	writer_->end();

	lastSave_ = modifier_->getTimeStamp();

	return true;
}
	
bool TFEditor::saveFunction(){
	
	if(fileNameFunction_.isEmpty())
	{
		fileNameFunction_ = QDir::currentPath().append(QString::fromStdString("/" + name_));

		fileNameFunction_ = QFileDialog::getSaveFileName(this,
			QObject::tr("Save Transfer Function"),
			fileNameFunction_,
			QObject::tr("TF Files (*.tf)"));

		if (fileNameFunction_.isEmpty()) return false;
	}
	
	if (!writer_->begin(fileNameFunction_.toLocal8Bit().data()))
	{
		QMessageBox errorMessage(QMessageBox::Critical,
			"Error",
			QString::fromStdString(writer_->errorMessage()),
			QMessageBox::Ok
		);
		errorMessage.setDefaultButton(QMessageBox::Ok);
		errorMessage.exec();
		return false;
	}

	writer_->writeDTD("TransferFunctionFile");

	writer_->writeStartElement("FunctionInfo");

		writer_->writeAttribute("Name", name_);
		writer_->writeAttribute("Dimension",
			TF::convert<TF::Types::Dimension, std::string>(structure_.dimension));
		writer_->writeAttribute("Function",
			TF::convert<TF::Types::Function, std::string>(structure_.function));

	modifier_->saveFunction(writer_);

	writer_->end();

	lastSave_ = modifier_->getTimeStamp();

	return true;
}

bool TFEditor::load(TF::XmlReaderInterface* reader, bool& sideError){	

	#ifndef TF_NDEBUG
		std::cout << "Loading editor..." << std::endl;
	#endif

	bool ok = modifier_->load(reader, sideError);
	
	bool error = !loadSettings_(reader);
	sideError = sideError || error;

	if(ok)
	{
		fileName_ = QString::fromStdString(reader->fileName());
		lastSave_ = modifier_->getTimeStamp();
	}

	return ok;
}

bool TFEditor::loadFunction(TF::XmlReaderInterface* reader){

	QMessageBox errorMessage(QMessageBox::Critical, "Transfer Function Loading Error", "", QMessageBox::Ok);
	errorMessage.setDefaultButton(QMessageBox::Ok);	

	std::string newName;
	if(!reader->readElement("FunctionInfo"))
	{
		errorMessage.setText(("File \"" + reader->fileName() + "\" is corrupted.").c_str());
		errorMessage.exec();
		return false;
	}

	newName = reader->readAttribute("Name");

	TF::Types::Dimension loadedDimension = TF::convert<std::string, TF::Types::Dimension>(
		reader->readAttribute("Dimension")
	);
	if(loadedDimension != structure_.dimension)
	{
		errorMessage.setText("Dimension does not match.");
		errorMessage.exec();
		return false;
	}

	TF::Types::Function loadedFunction = TF::convert<std::string, TF::Types::Function>(
		reader->readAttribute("Function")
	);
	if(loadedFunction != structure_.function)
	{
		errorMessage.setText("Function type does not match.");
		errorMessage.exec();
		return false;
	}

	bool ok = modifier_->loadFunction(reader);

	if(ok)
	{
		fileNameFunction_ = QString::fromStdString(reader->fileName());
		name_ = newName;
		lastSave_ = modifier_->getTimeStamp();
		++lastChange_;
		update();
	}
	else
	{
		errorMessage.setText(("File \"" + reader->fileName() + "\" is corrupted.").c_str());
		errorMessage.exec();
	}

	return ok;
}

void TFEditor::saveSettings_(TF::XmlWriterInterface* writer){}

bool TFEditor::loadSettings_(TF::XmlReaderInterface* reader){
	
	return true;	
}

bool TFEditor::close(){

	if(lastSave_ != modifier_->getTimeStamp())
	{
		QMessageBox msgBox;
		msgBox.setIcon(QMessageBox::Warning);
		msgBox.setText(QString::fromStdString(name_ + " has been modified."));
		msgBox.setInformativeText("Do you want to save your changes?");
		msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Save);
		int ret = msgBox.exec();

		if(ret == QMessageBox::Cancel) return false;
		if(ret == QMessageBox::Save) save();
	}
	emit Close(index_);
	QMainWindow::close();
	return true;
}

TF::Size TFEditor::getIndex(){

	return index_;
}

std::string TFEditor::getName(){

	return name_;
}

TFEditor::Attributes TFEditor::getAttributes(){

	return attributes_;
}

bool TFEditor::hasAttribute(const Attribute attribute){

	return (attributes_.find(attribute) != attributes_.end());
}

TF::Size TFEditor::getDimension(){

	return modifier_->getDimension();
}

TFFunctionInterface::Const TFEditor::getFunction(){

	return TFFunctionInterface::Const(modifier_->getFunction());
}

QDockWidget* TFEditor::getDockWidget() const{

	return editorDock_;
}

Common::TimeStamp TFEditor::lastChange(){

	if(modifier_->changed()) ++lastChange_;
	return lastChange_;
}

void TFEditor::setHistogram(TF::HistogramInterface::Ptr histogram){

	if(!histogram) return;

	modifier_->setHistogram(histogram);

	update();
}

void TFEditor::setDataStructure(const std::vector<TF::Size>& dataStructure){

	modifier_->setDataStructure(dataStructure);
	update();
}

void TFEditor::setActive(const bool active){

	active_ = active;
}

} // namespace GUI
} // namespace M4D
