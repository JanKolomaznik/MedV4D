#include "MedV4D/GUI/widgets/FilePathComboBox.h"

FilePathComboBox::FilePathComboBox( QWidget * parent, bool aOpenDialog ): QWidget( parent )
{
	setupUi( this );
	mFileChooseDialog = new QFileDialog( this );
	mFileChooseDialog->setAcceptMode(aOpenDialog ? QFileDialog::AcceptOpen : QFileDialog::AcceptSave);
	mFileChooseDialog->setViewMode(QFileDialog::Detail);
}

void
FilePathComboBox::showFileChooseDialog()
{
	QStringList fileNames;
	if( mFileChooseDialog->exec() ) {
		fileNames = mFileChooseDialog->selectedFiles();
		mComboBox->insertItem( 0, fileNames[0] );
		mComboBox->setCurrentIndex(0);
	}
}