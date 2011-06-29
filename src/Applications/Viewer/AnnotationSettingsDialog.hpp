#ifndef ANNOTATION_SETTINGS_DIALOG_HPP
#define ANNOTATION_SETTINGS_DIALOG_HPP

#include "ui_AnnotationSettingsDialog.h"
#include "AnnotationEditorController.hpp"

class AnnotationSettingsDialog: public QDialog, public Ui::AnnotationSettingsDialog
{
	Q_OBJECT;
public:
	AnnotationSettingsDialog()
	{
		setupUi( this );
		QObject::connect( applyButton, SIGNAL( clicked() ), this, SIGNAL( applied() ) );
	}
	
	int
	showDialog( const AnnotationEditorController::AnnotationSettings &aSettings ) 
	{
		mCurrentSettings = aSettings;
		return exec();
	}
	const AnnotationEditorController::AnnotationSettings &
	getSettings()const
	{
		return mCurrentSettings;
	}
signals:
	void
	applied();
protected:
	void
	synchronizeDialogAndSettings( AnnotationEditorController::AnnotationSettings &aSettings, bool aFrom )
	{
		
	}

	AnnotationEditorController::AnnotationSettings mCurrentSettings;
};

#endif /*ANNOTATION_SETTINGS_DIALOG_HPP*/
