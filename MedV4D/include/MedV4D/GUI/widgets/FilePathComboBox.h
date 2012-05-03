#ifndef FILE_PATH_COMBO_BOX_H
#define FILE_PATH_COMBO_BOX_H

#include "MedV4D/generated/ui_FilePathComboBox.h"
#include "MedV4D/Common/Common.h"
#include "MedV4D/GUI/utils/Settings.h"
#include "QtGui/QFileDialog"


class FilePathComboBox: public QWidget, public Ui::FilePathComboBox
{
	Q_OBJECT;
public:
	FilePathComboBox( QWidget * parent = 0, bool aOpenDialog = false );

	int
	count()const
	{ return mComboBox->count(); }

	int
	currentIndex()
	{ return mComboBox->currentIndex(); }
	
	void
	setCurrentIndex( int aValue )
	{ mComboBox->setCurrentIndex( aValue ); }

	const QString
	currentText()const 
	{ return mComboBox->currentText(); }
	
	void
	addItem( const QString & text, const QVariant & userData = QVariant() )
	{ mComboBox->addItem( text, userData ); }
	
public slots:
	void
	showFileChooseDialog();
protected:
	QFileDialog *mFileChooseDialog;
};

#endif //FILE_PATH_COMBO_BOX_H