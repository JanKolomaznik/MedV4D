#ifndef M4D_GUI_SCREEN_LAYOUT_WIDGET_H
#define M4D_GUI_SCREEN_LAYOUT_WIDGET_H

#include <QWidget>

#include "GUI/m4dGUIVtkRenderWindowWidget.h"


class QToolButton;
class QGroupBox;
class QSpinBox;

class m4dGUIScreenLayoutWidget: public QWidget
{
  Q_OBJECT

  public:
    /**
     * Constructor of m4dGUIScreenLayoutWidget.
     *
     * @param vtkRenderWindowWidget pointer to the VTK Render Window Widget
     * - where to manage layout
     * @param screenLayoutDialog pointer to the Screen Layout Dialog - to close it after 
     * clicking Ok
     * @ param parent parent of this widget - default is 0
     */
    m4dGUIScreenLayoutWidget ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                               QDialog *screenLayoutDialog, QWidget *parent = 0 );

  private slots:
    void seriesLayoutChanged ();
    void imageLayoutChanged ();
    void seriesApply ();
    void imageApply ();
    void accept ();

  private:
    /**
     * Creates Layout GroupBox - with buttons for various layouts and custom settings.
     * ...
     */
    QGroupBox *createLayoutGroupBox ( const QString &title, QToolButton ***toolButtons,
                                      QSpinBox **rowSpinBox, QSpinBox **columnSpinBox,
                                      const int rowValue, const int columnValue,
                                      const char *layoutChangedMember, const char *applyMember );

    QToolButton *createToolButton ( const QIcon &icon, const char *member );
    QSpinBox    *createSpinBox ( const int value );

    /// pointer to the Screen Layout Dialog - to close it after clicking Ok
    QDialog *screenLayoutDialog;

    static const char *layoutIconNames[];
    static const int layoutDimensions[][2];

    // layout buttons
    QToolButton **seriesLayoutToolButtons;
    QToolButton **imageLayoutToolButtons;
    // custom layouts - spinBoxes
    QSpinBox *seriesRowSpinBox;
    QSpinBox *seriesColumnSpinBox;
    QSpinBox *imageRowSpinBox;
    QSpinBox *imageColumnSpinBox;
};

#endif // M4D_GUI_SCREEN_LAYOUT_WIDGET_H

