#ifndef M4D_GUI_STUDY_MANAGER_WIDGET_H
#define M4D_GUI_STUDY_MANAGER_WIDGET_H

#include <QWidget>

#include "GUI/StManagerFilterComp.h"
#include "GUI/StManagerStudyListComp.h"


class QGroupBox;

namespace M4D {
namespace GUI {

class m4dGUIStudyManagerWidget: public QWidget
{
  Q_OBJECT

  public:

    /**
     * Constructor of m4dGUIStudyManagerWidget.
     *
     * @param studyManagerDialog pointer to the Study Manager Dialog - to close it after 
     * clicking View
     * @ param parent pointer to the parent of this widget - default is 0
     */
    m4dGUIStudyManagerWidget ( QDialog *studyManagerDialog, QWidget *parent = 0 );

    StManagerStudyListComp *getStudyListComponent () { return studyListComponent; }

  private:
    QGroupBox *createFilterGroupBox ();
    QGroupBox *createHangingProtocolsGroupBox ();
    QGroupBox *createStudyListGroupBox ( QDialog *studyManagerDialog );

    StManagerFilterComp *filterComponent;

    StManagerStudyListComp *studyListComponent;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_STUDY_MANAGER_WIDGET_H

