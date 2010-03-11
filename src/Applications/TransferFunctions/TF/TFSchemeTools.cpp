
#include "TFSchemeTools.h"
#include "ui_TFSchemeTools.h"

TFSchemeTools::TFSchemeTools(): tools(new Ui::TFSchemeTools){

	tools->setupUi(this);
}

TFSchemeTools::~TFSchemeTools(){
	delete tools;
}

void TFSchemeTools::setScheme(TFScheme* scheme){
	currentScheme = scheme;
}

void TFSchemeTools::save(){
	on_functionSave_clicked();
}

void TFSchemeTools::load(){

	int toRemove = tools->functionBox->count()-1;

	for(int i = 0; i < toRemove; ++i)
	{
		tools->functionBox->removeItem(tools->functionBox->count()-2);
	}

	vector<TFName> points = currentScheme->getFunctionNames();
	vector<TFName>::iterator first = points.begin();
	vector<TFName>::iterator end = points.end();
	vector<TFName>::iterator it = first;

	if(it != end)
	{
		tools->functionBox->setItemText(0, QString::fromStdString(*it));
		++it;
	}
	for(it; it != end; ++it)
	{
		tools->functionBox->insertItem(tools->functionBox->count()-1, QString::fromStdString(*it));
	}
	on_functionBox_currentIndexChanged(0);
}

/*
 * function methods
 */

void TFSchemeTools::on_functionSave_clicked(){

	if(tools->functionName->text() == "unsaved_function")
	{
		//cannot save
		//error dialog shows
		//TODO
		return;
	}

	QString newName = tools->functionName->text();

	currentScheme->removeFunction(currentScheme->currentFunction->name);
	currentScheme->currentFunction->name = newName.toStdString();
	currentScheme->addFunction(currentScheme->currentFunction, false);

	tools->functionBox->setItemText( tools->functionBox->currentIndex(), newName );
	
	emit CurrentFunctionChanged();
}

void TFSchemeTools::on_functionDelete_clicked(){

    int removeIndex = tools->functionBox->currentIndex();
    int count = tools->functionBox->count( );
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
	
	if(tools->functionBox->currentText().toStdString() != "unsaved_function")
	{
		currentScheme->removeFunction(currentScheme->currentFunction->name);

		delete currentScheme->currentFunction;
		currentScheme->currentFunction = currentScheme->getFunction(tools->functionBox->itemText(newIndex).toStdString());

		tools->functionBox->setCurrentIndex(newIndex);
		tools->functionBox->removeItem(removeIndex);
	}
	else
	{
		tools->functionBox->setCurrentIndex(newIndex);
	}
	
	emit CurrentFunctionChanged();
}

void TFSchemeTools::on_functionBox_currentIndexChanged(int index){

    int count = tools->functionBox->count();
	QString currentText = tools->functionBox->currentText();

	int unsaved = tools->functionBox->findText(QString::fromStdString("unsaved_function"));
	if(0 <= unsaved)
	{
		if(index == count - 1)
		{
			tools->functionBox->removeItem(count - 2);
			return;
		}
		tools->functionBox->removeItem(unsaved);
		--count;
	}

    if(index == count-1)
    {
        tools->functionBox->addItem(QString::fromStdString("add function"));
        
        tools->functionBox->setItemText(count-1, QString::fromStdString("unsaved_function"));

        tools->functionBox->setCurrentIndex(count-1);

		tools->functionName->setText(QString::fromStdString("unsaved_function"));
		tools->functionName->setFocus();

		delete currentScheme->currentFunction;
		currentScheme->currentFunction = new TFSchemeFunction("unsaved_function");
    }	
	else
	{
		delete currentScheme->currentFunction;
		currentScheme->currentFunction = currentScheme->getFunction(currentText.toStdString());

		tools->functionName->setText( QString::fromStdString(currentScheme->currentFunction->name) );
	}
	
	emit CurrentFunctionChanged();
}