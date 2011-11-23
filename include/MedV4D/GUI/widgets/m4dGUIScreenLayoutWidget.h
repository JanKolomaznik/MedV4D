/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUIScreenLayoutWidget.h 
 * @{ 
 **/

#ifndef M4D_GUI_SCREEN_LAYOUT_WIDGET_H
#define M4D_GUI_SCREEN_LAYOUT_WIDGET_H

#include <QWidget>

#include "GUI/widgets/m4dGUIMainViewerDesktopWidget.h"


class QToolButton;
class QGroupBox;
class QSpinBox;

namespace M4D {
namespace GUI {

/**
 * Class representing Screen Layout Widget - images that appear on the screen are laid out in 
 * a side-by-side (1x2) grid configuration by default. This configuration can be adjusted by this widget.
 * The Series layout determines the format of the panes in the window. Each pane can contain one series. 
 * The Image layout determines the format of the images within the active series.
 * It's communicating with the Main Viewer Desktop, which is executing the wanted layout changes.
 */
class m4dGUIScreenLayoutWidget: public QWidget
{
  Q_OBJECT

  public:

    /**
     * Screen Layout Widget constructor.
     *
     * @ param parent parent of this widget - default is 0
     */
    m4dGUIScreenLayoutWidget ( QWidget *parent = 0 );

  private slots:

    /**
     * Slot for managing predefined Series layout buttons behavior.
     */
    void seriesLayoutChanged ();

    /**
     * Slot for managing predefined Image layout buttons behavior.
     */
    void imageLayoutChanged ();

    /**
     * Slot for managing Series Apply button - custom layout change.
     */
    void seriesApply ();

    /**
     * Slot for managing Image Apply button - custom layout change.
     */
    void imageApply ();

    /**
     * Slot for Ok button handling.
     */
    void accept ();

  signals:

    /**
     * Signal emitted after changing series layout (via custom settings or predefined format buttons).
     * Should be connected to Main Viewer Desktop - to change the layout of viewers.
     */
    void seriesLayout ( const unsigned rows, const unsigned columns );

    /**
     * Signal emitted after changing image layout (via custom settings or predefined format buttons).
     * Should be connected to active viewer.
     */
    void imageLayout ( const unsigned slicesPerRow, const unsigned slicesPerColumn );
    
    /**
     * Signal for indicating situation when to go back to one slice mode (not more slice with 1x1).
     * Should be connected to active viewer.
     */
    void imageLayout ();

    /**
     * Signal for indicating wheather the Ok button was pushed - to close the dialog.
     * (connected to this widget's dialog close slot)
     */
    void ready ();

  private:

    /**
     * Creates Layout GroupBox - with buttons for various layouts (predefined format) and custom settings.
     *
     * @param title reference to the string containing the title
     * @param toolButtons pointer to the array of toolButton pointers - for predefined format buttons 
     * @param rowSpinBox pointer to the row spinBox - for custom (series/image) row numbers
     * @param columnSpinBox pointer to the column spinBox - for custom (series/image) column numbers
     * @param dimensionsIdx index to layoutDimensions - which dimensions to set as default (push button, set spinBoxes)
     * @param layoutChangedMember member - slot where to connect toolButtons - which will manage (series/image) layout changes
     * @param applyMember member - slot where to connect the Apply Button - which will manage custom (series/image) layout settings
     * @return pointer to the created and customized groupBox
     */
    QGroupBox *createLayoutGroupBox ( const QString &title, QToolButton ***toolButtons,
                                      QSpinBox **rowSpinBox, QSpinBox **columnSpinBox,
                                      const unsigned dimensionsIdx, const char *layoutChangedMember,
                                      const char *applyMember );

    /** 
     * Creates a toolButton, connects and configures it.
     *
     * @param icon reference to icon of the button
     * @param member other side of the connection
     * @return pointer to the created and configured toolButton
     */
    QToolButton *createToolButton ( const QIcon &icon, const char *member );

    /** 
     * Creates a spinBox and configures it.
     *
     * @param value default value of the spinBox
     * @return pointer to the created and configured spinBox
     */
    QSpinBox    *createSpinBox ( const int value );


    /// Icon names for tool buttons - for predefined format buttons.
    static const char *layoutIconNames[];
    /// Dimensions for predefined format buttons
    static const unsigned layoutDimensions[][2];

    /// Predefined layout buttons - series/image.
    QToolButton **seriesLayoutToolButtons;
    QToolButton **imageLayoutToolButtons;
    /// Custom layouts (series/image) - spinBoxes for dimensions.
    QSpinBox *seriesRowSpinBox;
    QSpinBox *seriesColumnSpinBox;
    QSpinBox *imageRowSpinBox;
    QSpinBox *imageColumnSpinBox;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_SCREEN_LAYOUT_WIDGET_H


/** @} */

