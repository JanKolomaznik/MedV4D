#include "SettingsBox.h"
#include "ui_SettingsBox.h"
#include "m4dMySliceViewerWidget.h"

#include <cassert>

/*
 * constructor, destructor
 */


SettingsBox::SettingsBox(M4D::GUI::m4dGUIMainWindow * parent)
    : ui(new Ui::SettingsBox), _parent( parent ){

    ui->setupUi(this);
}

SettingsBox::~SettingsBox(){
    delete ui;
}

void SettingsBox::build(){
//	M4D::Viewer::m4dGUIAViewerWidget *selectedViewer = _parent->currentViewerDesktop->setSelectedViewerLeftTool();
	int a = 32;
}


void SettingsBox::slotSetSphereCenter(double x, double y, double z)
{
	char string[50]; 

	sprintf( string, "%f", x ); 
	ui->lineEdit_x->setText(string);
	sprintf( string, "%f", y ); 
	ui->lineEdit_y->setText(string);
	sprintf( string, "%f", z+1 ); 
	ui->lineEdit_z->setText(string);
}

void SettingsBox::slotSetSphereRadius(int amountA, int amountB, double zoomRate)
{
	char string[100]; 

	sprintf( string, "AmountA = %d, AmountB = %d, zoomRate = %f", amountA, amountB, zoomRate); 
	ui->lineEdit_r->setText(string);
}
