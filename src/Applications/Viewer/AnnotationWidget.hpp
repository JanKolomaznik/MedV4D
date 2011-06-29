#ifndef ANNOTATION_WIDGET_HPP
#define ANNOTATION_WIDGET_HPP

#include "ui_AnnotationWidget.h"
#include "AnnotationEditorController.hpp"
#include <QtGui>

class AnnotationWidget: public QWidget, public Ui::AnnotationWidget
{
	Q_OBJECT;
public:
	AnnotationWidget( PointSet *aPoints, SphereSet *aSpheres ): mPoints( aPoints ), mSpheres( aSpheres )
	{
		setupUi( this );
		ASSERT( mPoints );

		pointsTableView->setModel( mPoints );
		spheresTableView->setModel( mSpheres );
	}
	
signals:

protected:
	PointSet *mPoints;
	SphereSet *mSpheres;
};

#endif /*ANNOTATION_WIDGET_HPP*/
