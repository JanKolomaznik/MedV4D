#include "GUI/m4dGUIStudyManagerWidget.h"

#include <QtGui>


m4dGUIStudyManagerWidget::m4dGUIStudyManagerWidget ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                                                     QWidget *parent )
  : QWidget( parent )
{
  Q_INIT_RESOURCE( m4dGUIStudyManagerWidget );

  // studyListGroupBox must go before filterGroupBox (it needs the studyList)
  createStudyListGroupBox( vtkRenderWindowWidget );
  createFilterGroupBox();
  createHangingProtocolsGroupBox();

  QGridLayout *mainLayout = new QGridLayout;
  mainLayout->addWidget( filterGroupBox, 0, 0 );
  mainLayout->addWidget( hangingProtocolsGroupBox, 0, 1 );
  mainLayout->addWidget( studyListGroupBox, 1, 0, 1, 2 );
  setLayout( mainLayout );

  setWindowTitle( tr( "Study Manager" ) );
}


void m4dGUIStudyManagerWidget::createFilterGroupBox ()
{
  filterGroupBox = new QGroupBox( tr( "Filter" ) );

  // Filter GroupBox - no resize - during resizing the main widget
  filterGroupBox->setSizePolicy( QSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed ) );

  QVBoxLayout *filterGroupBoxLayout = new QVBoxLayout;
  filterGroupBoxLayout->setMargin( 0 );
  filterComponent = new StManagerFilterComp( studyListComponent );
  filterGroupBoxLayout->addWidget( filterComponent );

  filterGroupBox->setLayout( filterGroupBoxLayout );
}


void m4dGUIStudyManagerWidget::createHangingProtocolsGroupBox ()
{
  hangingProtocolsGroupBox = new QGroupBox( tr( "Hanging protocols" ) );
}


void m4dGUIStudyManagerWidget::createStudyListGroupBox ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget )
{
  studyListGroupBox = new QGroupBox( tr( "Study List" ) );

  QVBoxLayout *studyListGroupBoxLayout = new QVBoxLayout;
  studyListGroupBoxLayout->setMargin( 0 );
  studyListComponent = new StManagerStudyListComp( vtkRenderWindowWidget );
  studyListGroupBoxLayout->addWidget( studyListComponent );

  studyListGroupBox->setLayout( studyListGroupBoxLayout ); 
}