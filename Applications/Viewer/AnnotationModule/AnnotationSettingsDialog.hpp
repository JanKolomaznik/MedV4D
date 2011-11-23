#ifndef ANNOTATION_SETTINGS_DIALOG_HPP
#define ANNOTATION_SETTINGS_DIALOG_HPP

#include "ui_AnnotationSettingsDialog.h"
#include "AnnotationModule/AnnotationEditorController.hpp"

class AnnotationSettingsDialog: public QDialog, public Ui::AnnotationSettingsDialog
{
	Q_OBJECT;
public:
	AnnotationSettingsDialog()
	{
		setupUi( this );
		contourColorChooser->enableAlpha( false );
		fillColorChooser->enableAlpha( true );
		QObject::connect( applyButton, SIGNAL( clicked() ), this, SIGNAL( applied() ) );
	}
	
	int
	showDialog( const AnnotationEditorController::AnnotationSettings &aSettings ) 
	{
		mCurrentSettings = aSettings;
		synchronizeDialogAndSettings( mCurrentSettings, false );
		return exec();
	}
	const AnnotationEditorController::AnnotationSettings &
	getSettings()
	{
		synchronizeDialogAndSettings( mCurrentSettings, true );
		return mCurrentSettings;
	}
signals:
	void
	applied();
protected:
	void
	synchronizeDialogAndSettings( AnnotationEditorController::AnnotationSettings &aSettings, bool aFrom )
	{
		if ( aFrom ) {
			mCurrentSettings.sphereContourColor2D = contourColorChooser->color();
			mCurrentSettings.sphereFillColor2D = fillColorChooser->color();
			mCurrentSettings.sphereColor3D = sphereColorChooser->color();
		} else {
			contourColorChooser->setColor( mCurrentSettings.sphereContourColor2D );
			fillColorChooser->setColor( mCurrentSettings.sphereFillColor2D );
			sphereColorChooser->setColor( mCurrentSettings.sphereColor3D );
		}
	}

	AnnotationEditorController::AnnotationSettings mCurrentSettings;
};

#endif /*ANNOTATION_SETTINGS_DIALOG_HPP*/
