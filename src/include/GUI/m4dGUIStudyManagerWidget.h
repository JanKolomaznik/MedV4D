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
    m4dGUIStudyManagerWidget ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                               QWidget *parent = 0 );

  private:
    void createFilterGroupBox ();
    void createHangingProtocolsGroupBox ();
    void createStudyListGroupBox ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget );

    QGroupBox           *filterGroupBox;
    StManagerFilterComp *filterComponent;

    QGroupBox   *hangingProtocolsGroupBox;

    QGroupBox              *studyListGroupBox;
    StManagerStudyListComp *studyListComponent;
};

#endif // M4D_GUI_STUDY_MANAGER_WIDGET_H

