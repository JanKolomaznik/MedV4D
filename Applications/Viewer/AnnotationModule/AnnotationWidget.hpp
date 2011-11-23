#ifndef ANNOTATION_WIDGET_HPP
#define ANNOTATION_WIDGET_HPP

#include "MedV4D/Common/Common.h"
#include "ui_AnnotationWidget.h"
#include "AnnotationModule/AnnotationEditorController.hpp"
#include <QtGui>
#include <iostream>
#include <fstream>


template< typename TStream >
void
saveModelAsCSVToStream( TStream &aStream, QAbstractItemModel &aModel )
{
	char sep = ';';
	int rows = aModel.rowCount();
	int cols = aModel.columnCount();

	QModelIndex index;
	aStream << sep;
	for (int j = 0; j < cols; ++j ) {
		aStream << qPrintable( aModel.headerData( j, Qt::Horizontal, Qt::DisplayRole ).toString() ) << sep ;
	}
	aStream << std::endl;
	for (int i = 0; i < rows; ++i ) {
		aStream << qPrintable( aModel.headerData( i, Qt::Vertical, Qt::DisplayRole ).toString() ) << sep ;
		for (int j = 0; j < cols; ++j ) {
			index = aModel.index ( i, j );
			aStream << qPrintable( aModel.data( index, Qt::DisplayRole ).toString() ) << sep ;
		}
		aStream << std::endl;
	}
}


class AnnotationWidget: public QWidget, public Ui::AnnotationWidget
{
	Q_OBJECT;
public:
	AnnotationWidget( AVectorItemModel *aPoints, AVectorItemModel *aSpheres, AVectorItemModel *aLines )
		: mPoints( aPoints ), mSpheres( aSpheres ), mLines( aLines )
	{
		setupUi( this );
		ASSERT( mPoints );

		pointsTableView->setModel( mPoints );
		spheresTableView->setModel( mSpheres );
		linesTableView->setModel( mLines );
	}
public slots:
	void
	exportCSV()
	{
		try {
		QString fileName = QFileDialog::getSaveFileName(this, tr("Export to file") );
		if ( !fileName.isEmpty() ) {
			std::ofstream outfile;
			outfile.open( qPrintable(fileName) );

			if ( outfile.good() ) {
				
				outfile << "POINTS" << std::endl;
				saveModelAsCSVToStream( outfile, *mPoints );

				outfile << std::endl << "SPHERES" << std::endl;
				saveModelAsCSVToStream( outfile, *mSpheres );

				outfile << std::endl << "LINES" << std::endl;
				saveModelAsCSVToStream( outfile, *mLines );

				outfile.close();
			} else {
				QMessageBox::critical ( NULL, "Export", "Cannot open file for writing" );
			}
		}
		} catch(...) {
			QMessageBox::critical ( NULL, "Erro", "Problem with file saving" );
		}
	}

	void
	clearAllAnnotations()
	{
		mPoints->clear();
		mSpheres->clear();
		mLines->clear();
		emit annotationsCleared();
	}
	
signals:
	void
	annotationsCleared();

protected:
	AVectorItemModel *mPoints;
	AVectorItemModel *mSpheres;
	AVectorItemModel *mLines;
};

#endif /*ANNOTATION_WIDGET_HPP*/
