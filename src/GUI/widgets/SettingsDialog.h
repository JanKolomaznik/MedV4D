#ifndef SETTINGS_DIALOG_H
#define SETTINGS_DIALOG_H

#include "ui_SettingsDialog.h"
#include "common/Common.h"
#include "GUI/utils/Settings.h"


class SettingsDialog: public QDialog, public Ui::SettingsDialog
{
	Q_OBJECT;
public:
	SettingsDialog( QWidget * parent = 0 ): QDialog( parent )
	{
		setupUi( this );
	}

	int
	showDialog( Settings &aSettings ) 
	{
		mSettings = &aSettings;
		settingsView->setModel( mSettings );
		//settingsView->setSortingEnabled( true );
		settingsView->resizeColumnsToContents();
		return exec();
	}
protected:
	Settings *mSettings;
};

#endif /*SETTINGS_DIALOG_H*/
