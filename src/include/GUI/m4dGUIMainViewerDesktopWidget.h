#ifndef M4D_GUI_MAIN_VIEWER_DESKTOP_H
#define M4D_GUI_MAIN_VIEWER_DESKTOP_H

#include <QtGui>

#include "GUI/m4dGUIVtkRenderWindowWidget.h"


/**
 * @class m4dGUIMainViewerDesktopWidget m4dGUIMainViewerDesktopWidget.h
 * Class representing the Main Viewer Desktop - containing abstract viewers.
 * It can manages its own layout, with selection of specific viewer - it causes
 * changes in toolBars and controls depending on the type of the viewer.
 */
class m4dGUIMainViewerDesktopWidget: public QWidget
{
  // Q_OBJECT

  public:

    /** 
     * Main Viewer Desktop constructor.
     *
     * @param parent pointer to the parent widget - default is 0
     */
    m4dGUIMainViewerDesktopWidget ( QWidget *parent = 0 );

    /** 
     * Changes the Desktop's layout.
     *
     * @param rows number of rows in the new layout
     * @param columns number of columns in the new layout
     */
    void setDesktopLayout( const int rows, const int columns );

    m4dGUIVtkRenderWindowWidget *getVtkRenderWindowWidget() { return vtkRenderWindowWidget; }

  private:
    m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;
};

#endif // M4D_GUI_MAIN_VIEWER_DESKTOP_H

