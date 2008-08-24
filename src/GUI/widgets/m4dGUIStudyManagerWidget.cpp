#include "GUI/m4dGUIStudyManagerWidget.h"

#include <QtGui>


// Q_INIT_RESOURCE macro cannot be used in a namespace
inline void initStudyManagerWidgetResource () { Q_INIT_RESOURCE( m4dGUIStudyManagerWidget ); }

namespace M4D {
namespace GUI {

m4dGUIStudyManagerWidget::m4dGUIStudyManagerWidget ( QDialog *studyManagerDialog, QWidget *parent )
  : QWidget( parent )
{
  initStudyManagerWidgetResource();

  // studyListGroupBox must go before filterGroupBox (it needs the studyList)
  QGroupBox *studyListGroupBox = createStudyListGroupBox( studyManagerDialog );
  QGroupBox *filterGroupBox = createFilterGroupBox();

  QGridLayout *mainLayout = new QGridLayout;
  mainLayout->addWidget( filterGroupBox, 0, 0 );
  mainLayout->addWidget( studyListGroupBox, 1, 0 );
  setLayout( mainLayout );
}


QGroupBox *m4dGUIStudyManagerWidget::createFilterGroupBox ()
{
  QGroupBox *filterGroupBox = new QGroupBox( tr( "Filter" ) );

  // Filter GroupBox - no resize - during resizing the main widget
  filterGroupBox->setSizePolicy( QSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed ) );

  QVBoxLayout *filterGroupBoxLayout = new QVBoxLayout;
  filterGroupBoxLayout->setMargin( 0 );
  filterComponent = new StManagerFilterComp( studyListComponent );
  filterGroupBoxLayout->addWidget( filterComponent );

  filterGroupBox->setLayout( filterGroupBoxLayout );

  return filterGroupBox;
}


QGroupBox *m4dGUIStudyManagerWidget::createStudyListGroupBox ( QDialog *studyManagerDialog )
{
  QGroupBox *studyListGroupBox = new QGroupBox( tr( "Study List" ) );

  QVBoxLayout *studyListGroupBoxLayout = new QVBoxLayout;
  studyListGroupBoxLayout->setMargin( 0 );
  studyListComponent = new StManagerStudyListComp( studyManagerDialog );
  studyListGroupBoxLayout->addWidget( studyListComponent );

  studyListGroupBox->setLayout( studyListGroupBoxLayout ); 

  return studyListGroupBox;
}

} // namespace GUI
} // namespace M4D

