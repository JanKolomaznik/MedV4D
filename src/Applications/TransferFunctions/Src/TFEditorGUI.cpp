#include <TFEditorGUI.h>

#include <TFQtXmlReader.h>

namespace M4D{
namespace GUI{

TFEditorGUI::TFEditorGUI(TFAbstractModifier::Ptr modifier,
						 TF::Types::Structure structure,
						 Attributes attributes,
						 std::string name):
	TFEditor(modifier, structure, attributes, name),
	ui_(new Ui::TFEditorGUI){
}

TFEditorGUI::~TFEditorGUI(){

	delete ui_;
}

void TFEditorGUI::setup(QMainWindow* mainWindow, const int index){

	if(index >= 0) index_ = index;

	ui_->setupUi(this);
	ui_->nameEdit->setText(QString::fromStdString(name_));
	if(hasAttribute(Composition))
	{
		ui_->menuEditor->setEnabled(false);
		ui_->actionFunctionLoad->setEnabled(false);
	}

	ui_->actionClose->setShortcut(QKeySequence::Close);
	ui_->actionEditorSave->setShortcut(QKeySequence::Save);

	ui_->editorWidget->setLayout(ui_->editorLayout);

	editorDock_ = new QDockWidget(QString::fromStdString(name_), mainWindow);
	editorDock_->setWidget(this);
	
	ui_->editorLayout->addWidget(&(*modifier_));
	QWidget* tools = modifier_->getTools();
	if(tools)
	{
		toolsDock_ = new QDockWidget(this);
		toolsDock_->setWidget(tools);
		toolsDock_->setFeatures(QDockWidget::AllDockWidgetFeatures);	
		toolsDock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);	
		toolsDock_->setWindowTitle(QString::fromStdString(name_ + ": Tools"));
		addDockWidget(Qt::LeftDockWidgetArea, toolsDock_);	

		int minHeight = tools->minimumHeight() + ui_->menubar->height() + ui_->statusBar->height();
		if(minimumHeight() > minHeight) minHeight = minimumHeight();
		editorDock_->resize(minimumWidth() + tools->width(), minHeight);
	}
	else
	{
		editorDock_->resize(minimumSize());
	}

	show();
	if(index == -1) ui_->activateButton->hide();
}

void TFEditorGUI::setActive(const bool active){

	ui_->activateButton->setChecked(active);
	active_ = active;
}

void TFEditorGUI::setAvailable(const bool available){

	ui_->activateButton->setEnabled(available);
}

void TFEditorGUI::on_actionEditorSave_triggered(){

	ui_->statusBar->showMessage("Saving Transfer Function Editor...");

	save();

	ui_->statusBar->clearMessage();
}

void TFEditorGUI::on_actionEditorSaveAs_triggered(){

	ui_->statusBar->showMessage("Saving Transfer Function Editor...");

	QString fileNameMem = fileName_;
	fileName_ = "";
	
	bool saveSuccess = save();

	if(!saveSuccess) fileName_ = fileNameMem;

	ui_->statusBar->clearMessage();
}

void TFEditorGUI::on_actionFunctionSave_triggered(){

	ui_->statusBar->showMessage("Saving Transfer Function...");

	saveFunction();

	ui_->statusBar->clearMessage();
}

void TFEditorGUI::on_actionFunctionSaveAs_triggered(){

	ui_->statusBar->showMessage("Saving Transfer Function...");

	QString fileNameMem = fileNameFunction_;
	fileNameFunction_ = "";
	
	bool saveSuccess = saveFunction();

	if(!saveSuccess) fileNameFunction_ = fileNameMem;

	ui_->statusBar->clearMessage();
}

void TFEditorGUI::on_actionFunctionLoad_triggered(){

	ui_->statusBar->showMessage("Loading Transfer Function...");

	QString fileName = QFileDialog::getOpenFileName(
		this,
		"Load Transfer Function",
		QDir::currentPath(),
		QObject::tr("TF Files (*.tf)"));

	if(fileName.isEmpty()) return;

	QMessageBox errorMessage(QMessageBox::Critical, "Transfer Function Loading Error", "", QMessageBox::Ok);
	errorMessage.setDefaultButton(QMessageBox::Ok);

	TF::QtXmlReader reader;
	if(!reader.begin(fileName.toLocal8Bit().data()))
	{
		errorMessage.setText(QString::fromStdString(reader.errorMessage()));
		errorMessage.exec();
		return;
	}

	bool loadSuccess = loadFunction(&reader);

	if(loadSuccess)
	{
		ui_->nameEdit->setText(name_.c_str());
		on_nameEdit_editingFinished();
	}

	ui_->statusBar->clearMessage();
}

void TFEditorGUI::on_actionClose_triggered(){

	close();
}

void TFEditorGUI::on_activateButton_clicked(){

	if(!active_) emit Activate(index_);
	else ui_->activateButton->setChecked(true);
}

void TFEditorGUI::on_nameEdit_editingFinished(){

	name_ = ui_->nameEdit->text().toLocal8Bit().data();
	editorDock_->setWindowTitle(ui_->nameEdit->text());
	toolsDock_->setWindowTitle(ui_->nameEdit->text().append(": Tools"));
}

} // namespace GUI
} // namespace M4D