#include "GUI/m4dGUIToolBarCustomizerWidget.h"

#include <QtGui>


// Q_INIT_RESOURCE macro cannot be used in a namespace
inline void initToolBarCustomizerWidgetResource () { Q_INIT_RESOURCE( m4dGUIToolBarCustomizerWidget ); }

namespace M4D {
namespace GUI {

m4dGUIToolBarCustomizerWidget::m4dGUIToolBarCustomizerWidget ( QWidget *parent )
  : QWidget( parent )
{
  initToolBarCustomizerWidgetResource();

  toolBarButtonsTable = createToolBarButtonsTable();
  QGroupBox *mouseButtonGroupBox = createMouseButtonGroupBox();
  QGroupBox *shortcutGroupBox    = createShortcutGroupBox();

  QGridLayout *mainLayout = new QGridLayout;

  mainLayout->addWidget( toolBarButtonsTable, 0, 0, 2, 1 );
  mainLayout->addWidget( mouseButtonGroupBox, 0, 1 );
  mainLayout->addWidget( shortcutGroupBox, 1, 1 );

  setLayout( mainLayout );
}


QTableWidget *m4dGUIToolBarCustomizerWidget::createToolBarButtonsTable ()
{
/*
  // Filter GroupBox - no resize - during resizing the main widget
  filterGroupBox->setSizePolicy( QSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed ) );

  QVBoxLayout *filterGroupBoxLayout = new QVBoxLayout;
  filterGroupBoxLayout->setMargin( 0 );
  filterComponent = new StManagerFilterComp( studyListComponent );
  filterGroupBoxLayout->addWidget( filterComponent );

  filterGroupBox->setLayout( filterGroupBoxLayout );
*/
  return new QTableWidget;
}


QGroupBox *m4dGUIToolBarCustomizerWidget::createMouseButtonGroupBox ()
{
  QGroupBox *mouseButtonGroupBox = new QGroupBox( tr( "Mouse Button" ) );
/*
  // Filter GroupBox - no resize - during resizing the main widget
  filterGroupBox->setSizePolicy( QSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed ) );

  QVBoxLayout *filterGroupBoxLayout = new QVBoxLayout;
  filterGroupBoxLayout->setMargin( 0 );
  filterComponent = new StManagerFilterComp( studyListComponent );
  filterGroupBoxLayout->addWidget( filterComponent );

  filterGroupBox->setLayout( filterGroupBoxLayout );
*/
  return mouseButtonGroupBox;
}


QGroupBox *m4dGUIToolBarCustomizerWidget::createShortcutGroupBox ()
{
  QGroupBox *shortcutGroupBox = new QGroupBox( tr( "Shortcut" ) );
/*
  QVBoxLayout *studyListGroupBoxLayout = new QVBoxLayout;
  studyListGroupBoxLayout->setMargin( 0 );
  studyListComponent = new StManagerStudyListComp( studyManagerDialog );
  studyListGroupBoxLayout->addWidget( studyListComponent );

  studyListGroupBox->setLayout( studyListGroupBoxLayout ); 
*/
  return shortcutGroupBox;
}

} // namespace GUI
} // namespace M4D

