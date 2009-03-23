/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUIProgressBarWidget.h
 * @{ 
 **/

#ifndef M4D_GUI_PROGRESS_BAR_WIDGET_H
#define M4D_GUI_PROGRESS_BAR_WIDGET_H

#include <QWidget>


class QProgressBar;
class QTimeLine;

namespace M4D {
namespace GUI {

/**
 * Class representing Progress Bar - indicating e.g. loading process of data sets 
 * (should be in a modal dialog).
 */
class m4dGUIProgressBarWidget: public QWidget
{
  Q_OBJECT

  public:

    /**
     * Progress Bar Widget constructor.
     *
     * @ param label description of the process - label for progress bar
     * @ param parent parent of this widget - default is 0
     */
    m4dGUIProgressBarWidget ( const QString &label, QWidget *parent = 0 );

    /**
     * Starts the progress bar.
     */
    void start ();

    /**
     * Stops the progress bar - emitting ready signal (should be connected
     * to this widget's dialog close slot).
     */
    void stop ();

  private slots:

    /**
     * Slot for pausing the progress bar - needed e.g. when Esc was pressed and the widget's 
     * dialog was closed (should be connected to this widget's dialog rejected signal).
     */  
    void pause ();

  signals:

    /**
     * Signal for indicating wheather the progress bar was stopped - to close the dialog.
     * (connected to this widget's dialog close slot)
     */
    void ready ();

  private:

    /// The progress bar.
    QProgressBar *progressBar;
    /// Timeline for animating the progress bar.
    QTimeLine *timeLine;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_PROGRESS_BAR_WIDGET_H


/** @} */

