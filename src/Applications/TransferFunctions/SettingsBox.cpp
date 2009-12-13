#include "SettingsBox.h"
#include "ui_SettingsBox.h"

#include <cassert>

/*
 * constructor, destructor
 */

SettingsBox::SettingsBox()
    : ui(new Ui::SettingsBox){

    ui->setupUi(this);

    ui->progressBar->reset();
    ui->progressLabel->hide();
    ui->progressBar->hide();

    savedFunctions = new TFScheme();
	currentFunction = savedFunctions->getFirstFunction();

	int marginH = 10;
	int marginV = 10;
	painter = new TFPaintingWidget(marginH, marginV);
	painter->setParent(this);
    painter->setObjectName(QString::fromUtf8("painter"));
	painter->setGeometry(QRect(220, 80, FUNCTION_RANGE + 2*marginH, COLOUR_RANGE + 2*marginV));
	painter->setView(&currentFunction);

	setFocus();
}

SettingsBox::~SettingsBox(){

	delete savedFunctions;	//includes delete of currentFunction and currentPoint
	delete painter;
    delete ui;
}

/*
 * function methods
 */

void SettingsBox::on_functionSave_clicked(){

	if(ui->functionName->text() == "unsaved_function")
	{
		//cannot save
		//error dialog shows
		//TODO
		return;
	}

	QString newName = ui->functionName->text();

	savedFunctions->removeFunction(currentFunction->name);
	currentFunction->name = newName.toStdString();
	savedFunctions->addFunction(currentFunction, false);

	ui->functionBox->setItemText( ui->functionBox->currentIndex(), newName );
	
	painter->repaint();
}

void SettingsBox::on_functionDelete_clicked(){

    int removeIndex = ui->functionBox->currentIndex();
    int count = ui->functionBox->count( );
	int newIndex = 0;

    if(removeIndex != 0)
    {      
        if(removeIndex < (count - 2))
        {
            newIndex = removeIndex;
        }
        else
        {
            newIndex = removeIndex - 1;
        }
    }

	savedFunctions->removeFunction(currentFunction->name);

	delete currentFunction;
	currentFunction = savedFunctions->getFunction(ui->functionBox->itemText(newIndex).toStdString());

	ui->functionBox->setCurrentIndex(newIndex);
    ui->functionBox->removeItem(removeIndex);
	
	painter->repaint();
}

void SettingsBox::on_functionBox_currentIndexChanged(int index){

    int count = ui->functionBox->count();
	QString currentText = ui->functionBox->currentText();

	int unsaved = ui->functionBox->findText(QString::fromStdString("unsaved_function"));
	if(0 <= unsaved)
	{
		if(index == count - 1)
		{
			ui->functionBox->removeItem(count - 2);
			return;
		}
		ui->functionBox->removeItem(unsaved);
		--count;
	}

    if(index == count-1)
    {
        ui->functionBox->addItem(QString::fromStdString("add function"));
        
        ui->functionBox->setItemText(count-1, QString::fromStdString("unsaved_function"));

        ui->functionBox->setCurrentIndex(count-1);

		ui->functionName->setText(QString::fromStdString("unsaved_function"));
		ui->functionName->setFocus();

		delete currentFunction;
		currentFunction = new TFFunction("unsaved_function");
    }	
	else
	{
		delete currentFunction;
		currentFunction = savedFunctions->getFunction(currentText.toStdString());

		ui->functionName->setText( QString::fromStdString(currentFunction->name) );
	}
	
	painter->repaint();
}

/*
 * scheme methods
 */

void SettingsBox::on_schemeUse_clicked(){

    ui->progressLabel->show();
    ui->progressBar->show();

	//TODO

    //---
        for(int i = 0; i <= 100; ++i)
        {
            ui->progressBar->setValue(i);
        }
    //---

	emit UseTransferFunction(savedFunctions);
}

void SettingsBox::on_actionExit_triggered(){

    close();
}

void SettingsBox::on_saveScheme_triggered(){

	savedFunctions->name = ui->schemeName->text().toStdString();
	on_functionSave_clicked();
	//savedFunctions->save();

     QString fileName =
         QFileDialog::getSaveFileName(this, tr("Save Transfer Function"),
                                      QDir::currentPath(),
                                      tr("TF Files (*.tf *.xml)"));
     if (fileName.isEmpty())
         return;

     QFile file(fileName);
     if (!file.open(QFile::WriteOnly | QFile::Text)) {
         QMessageBox::warning(this, tr("QXmlStream TransferFunctions"),
                              tr("Cannot write file %1:\n%2.")
                              .arg(fileName)
                              .arg(file.errorString()));
         return;
     }

	 TFXmlWriter writer;
     writer.write(&file, &savedFunctions);

         //statusBar()->showMessage(tr("File saved"), 2000);
}

void SettingsBox::on_loadScheme_triggered(){

	QString fileName =
		QFileDialog::getOpenFileName(this, QObject::tr("Open Transfer Function"),
								  QDir::currentPath(),
								  QObject::tr("TF Files (*.tf *.xml)"));
	if (fileName.isEmpty())
	 return;

	QFile file(fileName);
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
	 QMessageBox::warning(this, QObject::tr("QXmlStream TransferFunctions"),
						  QObject::tr("Cannot read file %1:\n%2.")
						  .arg(fileName)
						  .arg(file.errorString()));
	 return;
	}

	TFScheme* loaded = NULL;
	TFXmlReader reader;
	if (!reader.read(&file, &loaded)) {
		QMessageBox::warning(this, QObject::tr("QXmlStream TransferFunctions"),
						  QObject::tr("Parse error in file %1 at line %2, column %3:\n%4")
						  .arg(fileName)
						  .arg(reader.lineNumber())
						  .arg(reader.columnNumber())
						  .arg(reader.errorString()));
	}
	else
	{
		delete savedFunctions;
		savedFunctions = loaded;
		currentFunction = savedFunctions->getFirstFunction();

		int toRemove = ui->functionBox->count()-1;

		for(int i = 0; i < toRemove; ++i)
		{
			ui->functionBox->removeItem(ui->functionBox->count()-2);
		}

		vector<TFName> points = savedFunctions->getFunctionNames();
		vector<TFName>::iterator first = points.begin();
		vector<TFName>::iterator end = points.end();
		vector<TFName>::iterator it = first;

		if(it != end)
		{
			ui->functionBox->setItemText(0, QString::fromStdString(*it));
			++it;
		}
		for(it; it != end; ++it)
		{
			ui->functionBox->insertItem(ui->functionBox->count()-1, QString::fromStdString(*it));
		}
		ui->schemeName->setText(QString::fromStdString(savedFunctions->name));
		on_functionBox_currentIndexChanged(0);
	}
}
