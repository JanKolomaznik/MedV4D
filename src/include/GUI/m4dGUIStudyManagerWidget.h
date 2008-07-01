#ifndef M4D_GUI_STUDY_MANAGER_WIDGET_H
#define M4D_GUI_STUDY_MANAGER_WIDGET_H

#include <QWidget>

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
     * @ param parent pointer to the parent of this widget - default is 0
     */
    m4dGUIStudyManagerWidget ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                               QDialog *studyManagerDialog, QWidget *parent = 0 );

  private:
    QGroupBox *createFilterGroupBox ();
    QGroupBox *createHangingProtocolsGroupBox ();
    QGroupBox *createStudyListGroupBox ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                                         QDialog *studyManagerDialog );

    StManagerFilterComp *filterComponent;

    StManagerStudyListComp *studyListComponent;
};

#endif // M4D_GUI_STUDY_MANAGER_WIDGET_H

