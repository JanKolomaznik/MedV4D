#ifndef M4D_GUI_TOOLBAR_CUSTOMIZER_WIDGET_H
#define M4D_GUI_TOOLBAR_CUSTOMIZER_WIDGET_H

#include <QWidget>


class QTableWidget;
class QGroupBox;

namespace M4D {
namespace GUI {

class m4dGUIToolBarCustomizerWidget: public QWidget
{
  Q_OBJECT

  public:

    /**
     * Constructor of ToolBar Customizer Widget.
     *
     * @ param parent pointer to the parent of this widget - default is 0
     */
    m4dGUIToolBarCustomizerWidget ( QWidget *parent = 0 );

  private:

    QTableWidget *createToolBarButtonsTable ();
    QGroupBox *createMouseButtonGroupBox ();
    QGroupBox *createShortcutGroupBox ();

    QTableWidget *toolBarButtonsTable;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_TOOLBAR_CUSTOMIZER_WIDGET_H

