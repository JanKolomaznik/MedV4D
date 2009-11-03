#include "SettingsBox.h"
#include "uic_SettingsBox.h"

/*
 * support methods
 */

QString makeQStringFromInt(int value, string prefix = "")
{
    QString name = QString::fromStdString(prefix);

     name.append( QString::fromStdString( convert<int,string>(value) ) );

    return name;
}

/*
 * constructor, destructor
 */

SettingsBox::SettingsBox()
    : ui(new Ui::SettingsBox)
{
    ui->setupUi(this);

    ui->progressBar->reset();
    ui->progressLabel->hide();
    ui->progressBar->hide();

    //---
        points.insert( make_pair( ui->pointBox->currentText(), make_pair(0,0) ) );
        maxPointNumber = 1;
    //---
}

SettingsBox::~SettingsBox()
{
    delete ui;
}

/*
 * function methods
 */

void SettingsBox::on_functionName_textChanged(QString text)
{
    ui->functionBox->setItemText(ui->functionBox->currentIndex(),text);
}

void SettingsBox::on_functionNew_clicked()
{
    int count = ui->functionBox->count();

    QString name = makeQStringFromInt(count+1, "function_");

    ui->functionBox->addItem(name);

    ui->functionBox->setCurrentIndex(count);
}

void SettingsBox::on_functionDelete_clicked()
{
    int removeIndex = ui->functionBox->currentIndex();
    ui->functionBox->removeItem(removeIndex);

    if(removeIndex == 0)
    {
        if(ui->functionBox->count() == 0)
        {
           on_functionNew_clicked();
        }
        else
        {
            ui->functionBox->setCurrentIndex(0);
        }
    }
    else
    {
        if(removeIndex < ui->functionBox->count())
        {
            ui->functionBox->setCurrentIndex(removeIndex);
        }
        else
        {
            ui->functionBox->setCurrentIndex(removeIndex-1);
        }
    }
}

void SettingsBox::on_functionBox_currentIndexChanged(int index)
{
    ui->functionName->setText(ui->functionBox->currentText());
}

/*
 * point methods
 */

void SettingsBox::on_pointXValue_textChanged(QString value)
{
    //ui->pointBox->setItemText(ui->pointBox->currentIndex(),value);
    int getValue = convert<string,int>(value.toStdString());
    (points.find(ui->pointBox->currentText()))->second.first = getValue;
}

void SettingsBox::on_pointYValue_textChanged(QString value)
{
    //ui->pointBox->setItemText(ui->pointBox->currentIndex(),value);
    int getValue = convert<string,int>(value.toStdString());
    (points.find(ui->pointBox->currentText()))->second.second = getValue;
}

void SettingsBox::on_pointNew_clicked()
{
    ++maxPointNumber;
    QString name = makeQStringFromInt(maxPointNumber, "point_");

    ui->pointBox->addItem(name);
    points.insert( make_pair( name , make_pair(0,0) ) );

    ui->pointBox->setCurrentIndex(ui->pointBox->count()-1);
}

void SettingsBox::on_pointDelete_clicked()
{
    int removeIndex = ui->pointBox->currentIndex();
    points.erase(points.find(ui->pointBox->currentText()));
    ui->pointBox->removeItem(removeIndex);

    if(removeIndex == 0)
    {
        if(ui->pointBox->count() == 0)
        {
           on_pointNew_clicked();
        }
        else
        {
            ui->pointBox->setCurrentIndex(0);
        }
    }
    else
    {
        if(removeIndex < ui->pointBox->count())
        {
            ui->pointBox->setCurrentIndex(removeIndex);
        }
        else
        {
            ui->pointBox->setCurrentIndex(removeIndex-1);
        }
    }
}

void SettingsBox::on_pointBox_currentIndexChanged(int index)
{
    //ui->pointXValue->setText(ui->pointBox->currentText());
    ui->pointXValue->setText( makeQStringFromInt( points.find(ui->pointBox->currentText())->second.first ) );

    //ui->pointYValue->setText(ui->pointBox->currentText());
    ui->pointYValue->setText( makeQStringFromInt( points.find(ui->pointBox->currentText())->second.second ) );
}

/*
 * scheme methods
 */

void SettingsBox::on_schemeUse_clicked()
{
    ui->progressLabel->show();
    ui->progressBar->show();

    //---
        for(int i = 0; i <= 100; ++i)
        {
            ui->progressBar->setValue(i);
        }
    //---
}

void SettingsBox::on_actionExit_triggered()
{
    close();
}
