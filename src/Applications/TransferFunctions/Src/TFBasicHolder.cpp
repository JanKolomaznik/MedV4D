#include <TFBasicHolder.h>

namespace M4D{
namespace GUI{

TFBasicHolder::TFBasicHolder(TFAbstractModifier::Ptr modifier,
							 TF::Types::Structure structure,
							 Attributes attributes,
							 std::string name):
	holderDock_(NULL),
	holderMain_(NULL),
	ui_(new Ui::TFHolderUI),
	modifier_(modifier),
	index_(0),
	name_(name),
	attributes_(attributes){
}

TFBasicHolder::~TFBasicHolder(){

	if(toolsDock_) toolsDock_->close();
}

void TFBasicHolder::setup(QMainWindow* mainWindow, const int index){

	if(index >= 0) index_ = index;

	ui_->setupUi(this);
	ui_->nameEdit->setText(QString::fromStdString(name_));

	holderMain_ = new QMainWindow(mainWindow);
	holderMain_->setCentralWidget(this);

	holderDock_ = new QDockWidget(QString::fromStdString(name_), mainWindow);
	holderDock_->setWidget(holderMain_);
	
	bool closeConnected = QObject::connect(ui_->closeButton, SIGNAL(clicked()), this, SLOT(close()));
	tfAssert(closeConnected);
	bool saveConnected = QObject::connect(ui_->saveButton, SIGNAL(clicked()), this, SLOT(save()));
	tfAssert(saveConnected);
	bool activateConnected = QObject::connect(ui_->activateButton, SIGNAL(clicked()), this, SLOT(activate_clicked()));
	tfAssert(activateConnected);
	
	ui_->holderLayout->addWidget(&(*modifier_));
	QWidget* tools = modifier_->getTools();
	if(tools)
	{
		toolsDock_ = new QDockWidget(this);	
		toolsDock_->setWidget(tools);
		toolsDock_->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);	
		toolsDock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);	
		toolsDock_->setWindowTitle(QString::fromStdString(name_ + " Tools"));
		holderMain_->addDockWidget(Qt::LeftDockWidgetArea, toolsDock_);	
	}

	show();
	if(index == -1) ui_->activateButton->hide();
}

void TFBasicHolder::setAvailable(const bool available){

	ui_->activateButton->setEnabled(available);
}
	
void TFBasicHolder::save(){
	
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

		writer->writeAttribute("Name", name_);
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

	lastSave_ = modifier_->getTimeStamp();
}
	
void TFBasicHolder::saveData_(TFXmlWriter::Ptr writer){
		
	saveSettings_(writer);

	modifier_->save(writer);
}

bool TFBasicHolder::loadData(TFXmlReader::Ptr reader, bool& sideError){	

	#ifndef TF_NDEBUG
		std::cout << "Loading data:" << std::endl;
	#endif
	
	sideError = loadSettings_(reader);

	bool error;
	bool ok = modifier_->load(reader, error);
	sideError = sideError || error;

	fileName_ = reader->fileName();
	lastSave_ = modifier_->getTimeStamp();
	return ok;
}

void TFBasicHolder::saveSettings_(TFXmlWriter::Ptr writer){}

bool TFBasicHolder::loadSettings_(TFXmlReader::Ptr reader){
	
	return true;	
}

bool TFBasicHolder::close(){

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
	return true;
}

TF::Size TFBasicHolder::getIndex(){

	return index_;
}

std::string TFBasicHolder::getName(){

	return name_;
}

TFBasicHolder::Attributes TFBasicHolder::getAttributes(){

	return attributes_;
}

bool TFBasicHolder::hasAttribute(const Attribute attribute){

	return (attributes_.find(attribute) != attributes_.end());
}

TF::Size TFBasicHolder::getDimension(){

	return modifier_->getDimension();
}

TFFunctionInterface::Const TFBasicHolder::getFunction(){

	return TFFunctionInterface::Const(modifier_->getFunction());
}

QDockWidget* TFBasicHolder::getDockWidget() const{

	return holderDock_;
}

bool TFBasicHolder::changed(){

	return modifier_->changed();
}

void TFBasicHolder::setHistogram(TF::Histogram::Ptr histogram){

	if(!histogram) return;

	modifier_->setHistogram(histogram);

	update();
}

void TFBasicHolder::setDataStructure(const std::vector<TF::Size>& dataStructure){

	modifier_->setDataStructure(dataStructure);
	update();
}

void TFBasicHolder::resizeEvent(QResizeEvent *e){

	ui_->holderWidget->setGeometry(rect());
}

void TFBasicHolder::setActive(const bool active){

	ui_->activateButton->setChecked(active);
	active_ = active;
}

void TFBasicHolder::activate_clicked(){

	if(!active_) emit Activate(index_);
	else ui_->activateButton->setChecked(true);
}

void TFBasicHolder::on_nameEdit_editingFinished(){

	name_ = ui_->nameEdit->text().toStdString();
	holderDock_->setWindowTitle(QString::fromStdString(name_));
	toolsDock_->setWindowTitle(QString::fromStdString(name_ + " Tools"));
}

} // namespace GUI
} // namespace M4D