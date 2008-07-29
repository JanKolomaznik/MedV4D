#ifndef M4D_GUI_SCREEN_LAYOUT_WIDGET_H
#define M4D_GUI_SCREEN_LAYOUT_WIDGET_H

#include <QWidget>

#include "GUI/m4dGUIMainViewerDesktopWidget.h"


class QToolButton;
class QGroupBox;
class QSpinBox;

namespace M4D {
namespace GUI {

class m4dGUIScreenLayoutWidget: public QWidget
{
  Q_OBJECT

  public:
    /**
     * Constructor of m4dGUIScreenLayoutWidget.
     *
     * @param screenLayoutDialog pointer to the Screen Layout Dialog - to close it after 
     * clicking Ok
     * @ param parent parent of this widget - default is 0
     */
    m4dGUIScreenLayoutWidget ( QDialog *screenLayoutDialog, QWidget *parent = 0 );

  private slots:
    void seriesLayoutChanged ();
    void imageLayoutChanged ();
    void seriesApply ();
    void imageApply ();
    void accept ();

  signals:
    void seriesLayout ( const unsigned rows, const unsigned columns );
    void imageLayout ( const unsigned columns );


  private:
    /**
     * Creates Layout GroupBox - with buttons for various layouts and custom settings.
     * ...
     */
    QGroupBox *createLayoutGroupBox ( const QString &title, QToolButton ***toolButtons,
                                      QSpinBox **rowSpinBox, QSpinBox **columnSpinBox,
                                      const unsigned dimensionsIdx, const char *layoutChangedMember,
                                      const char *applyMember );

    QToolButton *createToolButton ( const QIcon &icon, const char *member );
    QSpinBox    *createSpinBox ( const int value );

    /// Pointer to the Screen Layout Dialog - to close it after clicking Ok.
    QDialog *screenLayoutDialog;

    static const char *layoutIconNames[];
    static const unsigned layoutDimensions[][2];

    // layout buttons
    QToolButton **seriesLayoutToolButtons;
    QToolButton **imageLayoutToolButtons;
    // custom layouts - spinBoxes
    QSpinBox *seriesRowSpinBox;
    QSpinBox *seriesColumnSpinBox;
    QSpinBox *imageRowSpinBox;
    QSpinBox *imageColumnSpinBox;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_SCREEN_LAYOUT_WIDGET_H

