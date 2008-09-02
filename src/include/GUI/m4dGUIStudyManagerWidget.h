/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUIStudyManagerWidget.h 
 * @{ 
 **/

#ifndef M4D_GUI_STUDY_MANAGER_WIDGET_H
#define M4D_GUI_STUDY_MANAGER_WIDGET_H

#include <QWidget>

#include "GUI/StManagerFilterComp.h"
#include "GUI/StManagerStudyListComp.h"


class QGroupBox;

namespace M4D {
namespace GUI {

/**
 * Class representing the Study Manager Widget - allows the user to search for a study to view
 * and manipulate with it.
 * It has two main components:
 * 1) Study Manager Filter Component constructor: It provides filtering functionality - depending on 
 * searching mode - Recent Exams (remote and DICOMDIR), Remote Exams, DICOMDIR modes. Filters by
 * various attributes, predefined patterns, clear option, etc.
 * 2) Study Manager Study List Component: it manages the uniform search result
 * viewing and manipulation - Recent Exams (remote and DICOMDIR), Remote Exams, DICOMDIR modes. 
 */
class m4dGUIStudyManagerWidget: public QWidget
{
  Q_OBJECT

  public:

    /**
     * Constructor of Study Manager Widget.
     *
     * @param studyManagerDialog pointer to the Study Manager Dialog - to change its title
     * @param parent pointer to the parent of this widget - default is 0
     */
    m4dGUIStudyManagerWidget ( QDialog *studyManagerDialog, QWidget *parent = 0 );

    /**
     * Getter to Study Manager Study List Component.
     *
     * @return pointer to the Study Manager Study List Component;
     */
    StManagerStudyListComp *getStudyListComponent () { return studyListComponent; }

  private:

    /**
     * Creates Filter GroupBox and fills it with the Study Manager Filter Component.
     *
     * @return pointer to the created groupBox (with Study Manager Filter Component)
     */
    QGroupBox *createFilterGroupBox ();

    /**
     * Creates Study List GroupBox and fills it with the Study Manager Study List Component.
     *
     * @param studyManagerDialog pointer to the Study Manager Dialog - to change its title
     * @return pointer to the created groupBox (with Study Manager Study List Component)
     */
    QGroupBox *createStudyListGroupBox ( QDialog *studyManagerDialog );


    /// Pointer to Study Manager Filter Component.
    StManagerFilterComp *filterComponent;
    /// Pointer to Study Manager Study List Component.
    StManagerStudyListComp *studyListComponent;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_STUDY_MANAGER_WIDGET_H


/** @} */

