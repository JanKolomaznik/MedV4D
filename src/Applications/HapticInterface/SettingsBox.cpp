#include "SettingsBox.h"
#include "ui_SettingsBox.h"

#include <cassert>

SettingsBox::SettingsBox(ViewerWindow* parent)
{
	this->parent = parent;
	ui = new Ui::SettingsBox();
	ui->setupUi(this);
}

SettingsBox::~SettingsBox()
{
	delete(ui);
}

void SettingsBox::slotChangeScale()
{
	emit scaleChanged(100.0);
}