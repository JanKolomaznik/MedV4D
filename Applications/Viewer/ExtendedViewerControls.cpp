#include "ExtendedViewerControls.hpp"
#include "ui_ExtendedViewerControls.h"

ExtendedViewerControls::ExtendedViewerControls(QWidget *parent) :
QWidget(parent),
ui(new Ui::ExtendedViewerControls)
{
	ui->setupUi(this);
}

ExtendedViewerControls::~ExtendedViewerControls()
{
	delete ui;
}

ViewerControls &
ExtendedViewerControls::viewerControls() const
{
	return *(ui->mViewerControls);
}

void ExtendedViewerControls::updateControls()
{
	viewerControls().updateControls();
}
