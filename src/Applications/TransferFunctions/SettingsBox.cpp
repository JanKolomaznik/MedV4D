#include "SettingsBox.h"
#include "uic_SettingsBox.h"

#include <cassert>

/*
 * support methods
 */

QString makeQStringFromInt(int value, string prefix = ""){

    QString name = QString::fromStdString(prefix);

     name.append( QString::fromStdString( convert<int,string>(value) ) );

    return name;
}

/*
 * constructor, destructor
 */

SettingsBox::SettingsBox()
    : ui(new Ui::SettingsBox){

    ui->setupUi(this);

    ui->progressBar->reset();
    ui->progressLabel->hide();
    ui->progressBar->hide();

    currentScheme = new TFScheme("Default Scheme");
	currentFunction = new TFFunction("default_function");
	currentPoint = new TFPoint(0,0);

	currentFunction->addPoint(currentPoint);

	currentScheme->addFunction(currentFunction);
}

SettingsBox::~SettingsBox(){

	delete currentScheme;	//includes delete of currentFunction and currentPoint
    delete ui;
}

/*
 * function methods
 */

void SettingsBox::on_functionSave_clicked(){

	QString newName = ui->functionName->text();
	int currentIndex = ui->functionBox->currentIndex();

	currentFunction = currentScheme->getFunction(ui->functionBox->currentText().toStdString());
	if(currentFunction == NULL)
	{
		if(ui->functionName->text() == "unsaved_function")
		{
			//cannot save
			//error dialog shows
			//TODO
			return;
		}
		currentFunction = new TFFunction( newName.toStdString() );
		currentScheme->addFunction(currentFunction);

		ui->functionBox->setItemText(currentIndex, newName);
		on_functionBox_currentIndexChanged(currentIndex);
	}
	else
	{
		currentScheme->changeFunctionName(currentFunction->name, newName.toStdString());
		//currentFunction->colourRGB = ui->functionColour->

		ui->functionBox->setItemText( currentIndex, newName );
	}
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

	currentScheme->removeFunction(currentFunction->name);
	currentFunction = currentScheme->getFunction(ui->functionBox->itemText(newIndex).toStdString());

	ui->functionBox->setCurrentIndex(newIndex);
    ui->functionBox->removeItem(removeIndex);
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

	int pointCount = ui->pointBox->count();
	for(int i = (pointCount-1); i > 0; --i)
	{
		ui->pointBox->removeItem(i);
	}

    if(index == count-1)
    {
        ui->functionBox->addItem(QString::fromStdString("add function"));
        
        ui->functionBox->setItemText(count-1, QString::fromStdString("unsaved_function"));

        ui->functionBox->setCurrentIndex(count-1);

		ui->functionName->setText(QString::fromStdString("unsaved_function"));
		ui->functionName->setFocus();

		ui->pointBox->setItemText(0,QString::fromStdString("unsaved_point"));
		ui->pointBox->addItem(QString::fromStdString("add point"));
		
		ui->pointXValue->setText( QString::fromStdString("-1") );
		ui->pointYValue->setText( QString::fromStdString("-1") );

		return;
    }	

	currentFunction = currentScheme->getFunction(ui->functionBox->currentText().toStdString());
	ui->functionName->setText( QString::fromStdString(currentFunction->name) );

	vector<TFName> currentFunctionPoints = currentFunction->getPointNames();

	if(currentFunctionPoints.size() > 0)
	{
		vector<TFName>::iterator first = currentFunctionPoints.begin();
		vector<TFName>::iterator end = currentFunctionPoints.end();
		vector<TFName>::iterator it = first;

		ui->pointBox->setItemText(0, QString::fromStdString(*it++));
		for(it; it != end; ++it)
		{
			ui->pointBox->addItem( QString::fromStdString(*it) );
		}

		ui->pointBox->addItem(QString::fromStdString("add point"));
	}
	else
	{
		ui->pointBox->setItemText(0, QString::fromStdString("add point"));
	}

	on_pointBox_currentIndexChanged(0);
}

/*
 * point methods
 */

void SettingsBox::on_pointSave_clicked(){

	if( currentScheme->getFunction(ui->functionBox->currentText().toStdString()) == NULL)
	{
		//cannot save point in unsaved function
		//TODO
		return;
	}

	int valueX = convert<string,int>(ui->pointXValue->text().toStdString());

	int valueY = convert<string,int>(ui->pointYValue->text().toStdString());

	TFName pointName = ui->pointBox->currentText().toStdString();
	currentPoint = currentFunction->getPoint(pointName);
	if(currentPoint == NULL)
	{
		if(valueX < 0 || valueY < 0)
		{
			//cannot save
			//error dialog shows
			//TODO
			return;
		}
		currentPoint = new TFPoint( valueX, valueY );
		currentFunction->addPoint(currentPoint);
	}
	else
	{
		currentPoint->x = valueX;
		currentPoint->y = valueY;
	}

	ui->pointBox->setItemText(
		ui->pointBox->currentIndex(),
		QString::fromStdString( TFPoint::makePointName(currentPoint) ) );
}

void SettingsBox::on_pointDelete_clicked(){

    int removeIndex = ui->pointBox->currentIndex();
    int count = ui->pointBox->count( );
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

	currentFunction->removePoint(ui->pointBox->currentText().toStdString());
	currentPoint = currentFunction->getPoint(ui->pointBox->itemText(newIndex).toStdString());

    ui->pointBox->removeItem(removeIndex);
	ui->pointBox->setCurrentIndex(newIndex);
}

void SettingsBox::on_pointBox_currentIndexChanged(int index){

    int count = ui->pointBox->count();
	QString currentText = ui->pointBox->currentText();

	int unsaved = ui->pointBox->findText(QString::fromStdString("unsaved_point"));
	if(0 <= unsaved)
	{
		if(index == count - 1)
		{
			ui->pointBox->removeItem(count - 2);
			return;
		}
		ui->pointBox->removeItem(unsaved);
		--count;
	}

	if(index == count -1)
    {
        ui->pointBox->addItem(QString::fromStdString("add point"));
        
        ui->pointBox->setItemText(count-1, QString::fromStdString("unsaved_point"));;

		ui->pointXValue->setText( QString::fromStdString("-1") );
		ui->pointYValue->setText( QString::fromStdString("-1") );
        ui->pointBox->setCurrentIndex(count-1);

		return;
    }

	currentPoint = currentFunction->getPoint(currentText.toStdString());

	assert(currentPoint != NULL);

	ui->pointXValue->setText( QString::fromStdString( convert<int,string>(currentPoint->x) ) );
	ui->pointYValue->setText( QString::fromStdString( convert<int,string>(currentPoint->y) ) );

	ui->pointBox->setCurrentIndex(ui->pointBox->findText(currentText));
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
