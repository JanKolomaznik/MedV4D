#include <TFAbstractHolder.h>

namespace M4D{
namespace GUI{

TFAbstractHolder::TFAbstractHolder():
	holderDock_(NULL),
	holderMain_(NULL),
	ui_(new Ui::TFHolderUI),
	index_(0),
	name_("Default"),
	attributes_(),
	saved_(false),
	painterLeftTopMargin_(10, 50),
	painterRightBottomMargin_(10, 10){
}
	
void TFAbstractHolder::save(){
	
	if(fileName_.isEmpty()) fileName_ = QDir::currentPath().append(QString::fromStdString("/" + name_));

	fileName_ = QFileDialog::getSaveFileName(this,
		QObject::tr("Save Transfer Function"),
		fileName_,
		QObject::tr("TF Files (*.tf)"));

	if (fileName_.isEmpty()) return;

	QFile file(fileName_);
	if (!file.open(QFile::WriteOnly | QFile::Text))
	{
		QMessageBox::warning(this,
			QObject::tr("Transfer Functions"),
			QObject::tr("Cannot write file %1:\n%2.")
			.arg(fileName_)
			.arg(file.errorString()));
		return;
	}
	
	TFXmlWriter::Ptr writer(new TFXmlWriter(&file));

	writer->writeDTD("TransferFunctionsFile");

	writer->writeStartElement("Editor");

		writer->writeAttribute("Predefined",
			TF::convert<TF::Types::Predefined, std::string>(structure_.predefined));
		writer->writeAttribute("Holder",
			TF::convert<TF::Types::Holder, std::string>(structure_.holder));
		writer->writeAttribute("Function",
			TF::convert<TF::Types::Function, std::string>(structure_.function));
		writer->writeAttribute("Painter",
			TF::convert<TF::Types::Painter, std::string>(structure_.painter));
		writer->writeAttribute("Modifier",
			TF::convert<TF::Types::Modifier, std::string>(structure_.modifier));

		saveData_(writer);

	writer->finalizeDocument();

	file.close();

	saved_ = true;
}

bool TFAbstractHolder::close(){

	if(!saved_)
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
	return true;
}

TF::Size TFAbstractHolder::getIndex(){

	return index_;
}

std::string TFAbstractHolder::getName(){

	return name_;
}

TFAbstractHolder::Attributes TFAbstractHolder::getAttributes(){

	return attributes_;
}

QDockWidget* TFAbstractHolder::getDockWidget() const{

	return holderDock_;
}

void TFAbstractHolder::setup(QMainWindow* mainWindow, const int index){

	if(index >= 0) index_ = index;

	holderMain_ = new QMainWindow(mainWindow);

	ui_->setupUi(this);
	setMouseTracking(true);
	holderMain_->setCentralWidget(this);

	holderDock_ = new QDockWidget(QString::fromStdString(name_), mainWindow);
	holderDock_->setWidget(holderMain_);

	show();
	if(index == -1) ui_->activateButton->hide();
	
	bool closeConnected = QObject::connect(ui_->closeButton, SIGNAL(clicked()), this, SLOT(close()));
	tfAssert(closeConnected);
	bool saveConnected = QObject::connect(ui_->saveButton, SIGNAL(clicked()), this, SLOT(save()));
	tfAssert(saveConnected);
	bool activateConnected = QObject::connect(ui_->activateButton, SIGNAL(clicked()), this, SLOT(activate_clicked()));
	tfAssert(activateConnected);
}

void TFAbstractHolder::refresh_view(){

	saved_ = false;
	update();
}

void TFAbstractHolder::activate(){

	ui_->activateButton->setChecked(true);
	active_ = true;
}

void TFAbstractHolder::deactivate(){

	ui_->activateButton->setChecked(false);
	active_ = false;
}

void TFAbstractHolder::activate_clicked(){

	if(!active_) emit Activate(index_);
	else ui_->activateButton->setChecked(true);
}

} // namespace GUI
} // namespace M4D