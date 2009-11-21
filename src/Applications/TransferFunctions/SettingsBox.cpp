#include "SettingsBox.h"
#include "uic_SettingsBox.h"

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

    savedFunctions = new TFScheme("Default Scheme");
	currentFunction = new TFFunction("default_function");

	savedFunctions->addFunction(currentFunction, false);

	painter = new PaintingWidget(this);
    painter->setObjectName(QString::fromUtf8("painter"));
    painter->setGeometry(QRect(220, 80, 350, 280));

	painter->setView(&currentFunction);
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
}

void SettingsBox::on_actionExit_triggered(){

    close();
}
