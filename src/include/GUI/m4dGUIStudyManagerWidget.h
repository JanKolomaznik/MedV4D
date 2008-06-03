#ifndef M4D_GUI_STUDY_MANAGER_WIDGET_H
#define M4D_GUI_STUDY_MANAGER_WIDGET_H

#include <QWidget>
#include <QMainWindow>

#include "GUI/StManagerFilterComp.h"
#include "GUI/StManagerStudyListComp.h"


class QGroupBox;

class m4dGUIStudyManagerWidget: public QWidget
{
  Q_OBJECT

  public:
    /**
     * Constructor of m4dGUIStudyManagerWidget.
     *
     * @param vtkRenderWindowWidget pointer to the VTK Render Window Widget
     * - where to render image after clicking View
     * @param studyManagerDialog pointer to the Study Manager Dialog - to close it after 
     * clicking View
     * @ param parent parent of this widget - default is 0
     */
    m4dGUIStudyManagerWidget ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                               QDialog *studyManagerDialog, QWidget *parent = 0 );

  private:
    void createFilterGroupBox ();
    void createHangingProtocolsGroupBox ();
    void createStudyListGroupBox ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                                   QDialog *studyManagerDialog );

    QGroupBox           *filterGroupBox;
    StManagerFilterComp *filterComponent;

    QGroupBox   *hangingProtocolsGroupBox;

    QGroupBox              *studyListGroupBox;
    StManagerStudyListComp *studyListComponent;
};

#endif // M4D_GUI_STUDY_MANAGER_WIDGET_H

