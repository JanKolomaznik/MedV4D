#include "SettingsBox.h"
#include "ui_SettingsBox.h"

#include <TF/TFScheme.h>

#include <cassert>

/*
 * constructor, destructor
 */

SettingsBox::SettingsBox(): ui(new Ui::SettingsBox){

    ui->setupUi(this);

    ui->progressBar->reset();
    ui->progressLabel->hide();
    ui->progressBar->hide();
}

SettingsBox::~SettingsBox(){

	delete tf;	//includes delete of currentFunction and currentPoint
    delete ui;
}

void SettingsBox::build(){

	tf = createDefaultTransferFunction();
	setupToolsAndPainter();
}

void SettingsBox::setupToolsAndPainter(){	

	toolsWidget = tf->getTools();
    toolsWidget->setObjectName(QString::fromUtf8("toolsWidget"));
    toolsWidget->setGeometry(QRect(10, 70, 220, 290));
	toolsWidget->setParent(this);

	painterWidget = tf->getPainter();
	painterWidget->setObjectName(QString::fromUtf8("painterWidget"));
	painterWidget->setGeometry(QRect(240, 70, 330, 290));
	painterWidget->setParent(this);
}

TFAFunction* SettingsBox::createDefaultTransferFunction(){
	return new TFScheme();
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

	emit UseTransferFunction(tf);
}

void SettingsBox::on_actionExit_triggered(){

    close();
}

void SettingsBox::on_saveScheme_triggered(){

	tf->name = ui->schemeName->text().toStdString();
	//tf->save(); TODO

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

	 TFScheme* toWrite = (TFScheme*)tf;
	 TFXmlWriter writer;
     writer.write(&file, &toWrite);

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
		delete tf;
		tf = loaded;
		tf->load();
		setupToolsAndPainter();
		ui->schemeName->setText(QString::fromStdString(tf->name));
	}
}
