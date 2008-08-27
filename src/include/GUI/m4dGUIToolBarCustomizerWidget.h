/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUIToolBarCustomizerWidget.h 
 * @{ 
 **/

#ifndef M4D_GUI_TOOLBAR_CUSTOMIZER_WIDGET_H
#define M4D_GUI_TOOLBAR_CUSTOMIZER_WIDGET_H

#include <QWidget>
#include <QList>


class QTableWidgetItem;
class QTableWidget;

namespace M4D {
namespace GUI {

/**
 * Class representing ToolBar Customizer Widget - allows user interfaces to be customized by letting the
 * user assign new keyboard shortcuts, mouse buttons, and fine-tune the appearance of toolbars.
 * The user is able to edit actions behavior in the table before either accepting the changes with
 * the OK button or rejecting them with the Cancel button. Since it's allowed any text to be entered 
 * in each shortcut field, it's needed a way to check that it can be used to specify a valid shortcut.
 */
class m4dGUIToolBarCustomizerWidget: public QWidget
{
  Q_OBJECT

  public:

    /**
     * Constructor of ToolBar Customizer Widget.
     *
     * @param actions pointer to the array of pointers to actions of the viewer - tools - to be 
     * customized (shortcuts, buttons, add to/remove from the toolBar)
     * @param actionsNum number of actions (size of the array of pointers)
     * @param parent pointer to the parent of this widget - default is 0
     */
    m4dGUIToolBarCustomizerWidget ( QAction **actions, unsigned actionsNum, QWidget *parent = 0 );

  private slots:

    /**
     * Slot changing the visibility of the tool (just in actions table) - clicking the visibility icon ->
     * switching between shown and hidden state (icon and toolTip is changing).
     *
     * @param row the row of the clicked icon - in the actions table
     * @param column the column of the clicked icon - in the actions table
     */
    void changeVisibility ( int row, int column );

    /**
     * Slot for recording current shortcut value - when the user starts to edit a cell item, 
     * this slot is called. Before the user gets a chance to modify the contents, the
     * cell's current text is recorded. Later, if the replacement text is not suitable, 
     * it can be reseted to this value.
     *
     * @param item current item of our table
     */
    void recordActionShortcut ( QTableWidgetItem *item );

    /**
     * Slot for validating the shortcut value - when the user has finished editing the cell item, 
     * this slot is called giving the chance to ensure that the text in the cell is suitable
     * for use as a shortcut.
     *
     * @param item current item of our table, which was changed and need to be validated
     */
    void validateActionShortcut ( QTableWidgetItem *item );

    /**
     * Slot for accepting the dialog - it updates all the known actions with new settings.
     */
    void accept ();

    /**
     * Slot for rejecting the dialog - it's not needed to perform any actions (just set back current
     * values to the table), the changes to the settings will be lost.
     */
    void reject ();

  signals:

    /**
     * Signal for indicating wheather everything is ready - OK button pushed and actions
     * modified to the new behavior - time to close the dialog.
     */
    void ready ();

    /**
     * Signal for indicating wheather the user closes the dialog with the Cancel button - 
     * close the dialog - connected to its reject slot.
     */
    void cancel ();

  private:

    /**
     * Creates customizer table - allows editing actions (through editable items)
     * Rows ~ Actions
     *
     * @return pointer to the created and set up tableWidget
     */
    QTableWidget *createToolBarButtonsTable ();

    /**
     * Modifies the behavior of each QAction whose name matches an entry in the settings.
     */
    void loadActions ();

    /**
     * Saves new settings of actions - to remember them, to survive a restart.
     */
    void saveActions ();


    /// Table of actions to be customized.
    QTableWidget *toolBarButtonsTable;
    /// String holding the old shortcut value of the action - replacement text not suitable -> reset to this value.
    QString oldAccelText;
    /// Pointer to the array of pointers to actions of the viewer - tools (comming from the mainWindow).
    QAction **actions;
    /// Number of actions (size of the array of pointers).
    unsigned actionsNum;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_TOOLBAR_CUSTOMIZER_WIDGET_H


/** @} */

