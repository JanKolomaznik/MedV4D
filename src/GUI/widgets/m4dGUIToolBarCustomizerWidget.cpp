#include "GUI/m4dGUIToolBarCustomizerWidget.h"

#include <QtGui>


// Q_INIT_RESOURCE macro cannot be used in a namespace
inline void initToolBarCustomizerWidgetResource () { Q_INIT_RESOURCE( m4dGUIToolBarCustomizerWidget ); }

namespace M4D {
namespace GUI {

/// Prefixes for actions visibility (in the hidden column of the actions table)
#define SHOWN_ACTION_PREFIX          "Shown"
#define HIDDEN_ACTION_PREFIX         "Hidden"

m4dGUIToolBarCustomizerWidget::m4dGUIToolBarCustomizerWidget ( QAction **actions, unsigned actionsNum, 
                                                               QWidget *parent )
  : actions( actions ), actionsNum( actionsNum ), QWidget( parent )
{
  initToolBarCustomizerWidgetResource();

  // loads previously saved actions settings (from QSettings)
  loadActions();

  toolBarButtonsTable = createToolBarButtonsTable();

  connect( toolBarButtonsTable, SIGNAL(cellClicked( int, int )), 
           this, SLOT(changeVisibility( int, int )) );
  connect( toolBarButtonsTable, SIGNAL(currentItemChanged( QTableWidgetItem *, QTableWidgetItem * ) ), 
           this, SLOT(recordAction( QTableWidgetItem * )) );
  connect( toolBarButtonsTable, SIGNAL(itemChanged( QTableWidgetItem * )), 
           this, SLOT(validateAction( QTableWidgetItem * )) );
  
  QPushButton *okButton     = new QPushButton( tr( "&OK" ), this );
  QPushButton *cancelButton = new QPushButton( tr( "&Cancel" ), this );

  connect( okButton,     SIGNAL(clicked()), this, SLOT(accept()) );
  connect( cancelButton, SIGNAL(clicked()), this, SLOT(reject()) );

  QHBoxLayout *buttonLayout = new QHBoxLayout;
  buttonLayout->setSpacing( 8 );
  buttonLayout->addStretch( 1 );
  buttonLayout->addWidget( okButton );
  buttonLayout->addWidget( cancelButton );

  QVBoxLayout *mainLayout = new QVBoxLayout( this );
  mainLayout->setMargin( 8 );
  mainLayout->setSpacing( 8 );
  mainLayout->addWidget( toolBarButtonsTable );
  mainLayout->addLayout( buttonLayout );
}


void m4dGUIToolBarCustomizerWidget::changeVisibility ( int row, int column )
{
  if ( column != 1 ) {
    return;
  }

  QTableWidgetItem *visibilityItem = toolBarButtonsTable->item( row, 1 );
  QString visibilityStr = visibilityItem->toolTip();

  if  ( visibilityStr == SHOWN_ACTION_PREFIX )
  {
    visibilityItem->setIcon( QIcon( ":/icons/hidden.png" ) );
    visibilityItem->setToolTip( HIDDEN_ACTION_PREFIX );
  }
  else if ( visibilityStr == HIDDEN_ACTION_PREFIX )
  {
    visibilityItem->setIcon( QIcon( ":/icons/shown.png" ) );
    visibilityItem->setToolTip( SHOWN_ACTION_PREFIX );
  }
  else
  {
    // default
    visibilityItem->setIcon( QIcon( ":/icons/shown.png" ) );
    visibilityItem->setToolTip( SHOWN_ACTION_PREFIX );
  }
}


void m4dGUIToolBarCustomizerWidget::recordAction ( QTableWidgetItem *item )
{
  oldAccelText = item->text();
}


void m4dGUIToolBarCustomizerWidget::validateAction ( QTableWidgetItem *item )
{
  QString accelText = QString( QKeySequence( item->text() ) );

  if ( accelText.isEmpty() && !item->text().isEmpty() ) {
    item->setText( oldAccelText );
  }
  else {
    item->setText( accelText );
  }
}


void m4dGUIToolBarCustomizerWidget::accept ()
{
  for ( unsigned row = 1; row < actionsNum; row++ ) {
    actions[row]->setShortcut( QKeySequence( toolBarButtonsTable->item( row - 1, 1 )->text() ) );
  }

  saveActions();

  emit ready();
}


void m4dGUIToolBarCustomizerWidget::reject ()
{
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    QTableWidgetItem *shortcutItem = new QTableWidgetItem( QString( actions[row]->shortcut() ) );
    toolBarButtonsTable->setItem( row - 1, 1, shortcutItem );
  }

  emit cancel();
}


QTableWidget *m4dGUIToolBarCustomizerWidget::createToolBarButtonsTable ()
{
  if ( !actions ) {
    return new QTableWidget;
  }

  QTableWidget *actionsTable = new QTableWidget( actionsNum - 1, 3, this );

  QStringList labels;
  labels << tr( "Description" ) << tr( "" ) << tr( "Shortcut" );
  actionsTable->setHorizontalHeaderLabels( labels );
  actionsTable->horizontalHeader()->setResizeMode( 0, QHeaderView::ResizeToContents );
  actionsTable->horizontalHeader()->setResizeMode( 1, QHeaderView::ResizeToContents );
  actionsTable->horizontalHeader()->setStretchLastSection( true );
  actionsTable->verticalHeader()->hide();

  // from second - first is the empty action for all unplugged slots (it's not in toolBar)
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    QTableWidgetItem *actionItem = new QTableWidgetItem( actions[row]->text() );
    actionItem->setFlags( actionItem->flags() & ~Qt::ItemIsEditable );
    actionItem->setIcon( actions[row]->icon() );
    actionsTable->setItem( row - 1, 0, actionItem );

    QTableWidgetItem *visibilityItem = new QTableWidgetItem;
    if ( actions[row]->isVisible() ) 
    {
      visibilityItem->setIcon( QIcon( ":/icons/shown.png" ) );
      visibilityItem->setToolTip( SHOWN_ACTION_PREFIX );
    }
    else
    {
      visibilityItem->setIcon( QIcon( ":/icons/hidden.png" ) );   
      visibilityItem->setToolTip( HIDDEN_ACTION_PREFIX );
    }
    visibilityItem->setFlags( visibilityItem->flags() & ~Qt::ItemIsEditable );
    actionsTable->setItem( row - 1, 1, visibilityItem );
    
    QTableWidgetItem *shortcutItem = new QTableWidgetItem( QString( actions[row]->shortcut() ) );
    actionsTable->setItem( row - 1, 2, shortcutItem );
  }

  return actionsTable;
}


void m4dGUIToolBarCustomizerWidget::loadActions ()
{
  QSettings settings;
  settings.beginGroup( "Actions" );
    
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    QString accelText = settings.value( actions[row]->text() ).toString();

    if ( !accelText.isEmpty() ) {
      actions[row]->setShortcut( QKeySequence( accelText ) );
    }
  }

  settings.endGroup();
}


void m4dGUIToolBarCustomizerWidget::saveActions ()
{
  QSettings settings;
  settings.beginGroup( "Actions" );
     
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    QString accelText = QString( actions[row]->shortcut() );
    settings.setValue( actions[row]->text(), accelText );
  }

  settings.endGroup();
}

} // namespace GUI
} // namespace M4D

