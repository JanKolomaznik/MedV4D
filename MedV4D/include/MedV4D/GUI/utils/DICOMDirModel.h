#ifndef DICOM_DIR_MODEL_H
#define DICOM_DIR_MODEL_H

#include <QtWidgets>
#include "MedV4D/DICOMInterface/DcmProvider.h"



class DICOMDirModel: public QAbstractItemModel
{
public:
	DICOMDirModel( DcmDicomDirPtr aDicomDir, QObject *parent = 0): mDicomDir( aDicomDir ), QAbstractItemModel(parent)
	{}

	int
	columnCount( const QModelIndex & parent = QModelIndex() ) const
	{
		return 1;
	}

	QVariant
	data( const QModelIndex & index, int role = Qt::DisplayRole ) const
	{
		return QVariant();
	}

	QModelIndex
	index( int row, int column, const QModelIndex & parent = QModelIndex() ) const
	{
		if ( parent.isValid() ) {
			DcmDirectoryRecord* rec = static_cast< DcmDirectoryRecord* >( parent.internalPointer() );
			if ( rec && row < rec->card() ) {
				return createIndex( row, column, rec->getSub( row ) );
			}
		} else {
			if ( row == 0 ) {
				return createIndex( row, column, &(mDicomDir->getRootRecord()) );
			}
		}
		return QModelIndex();
	}

	QModelIndex
	parent( const QModelIndex & index ) const;

	int
	rowCount( const QModelIndex & parent = QModelIndex() ) const
	{
		if ( parent.isValid() ) {
			DcmDirectoryRecord* rec = static_cast< DcmDirectoryRecord* >( parent.internalPointer() );
			if ( rec ) {
				return rec->card();
			}
		}
		return 1;
	}
protected:
	DcmDicomDirPtr mDicomDir;
};


#endif /*DICOM_DIR_MODEL_H*/
