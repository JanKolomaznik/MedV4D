#ifndef S_MANAGER_FILTER_COMP_H
#define S_MANAGER_FILTER_COMP_H

#include <QWidget>

#include "GUI/StManagerStudyListComp.h"


class QPushButton;
class QComboBox;
class QCheckBox;
class QDateEdit;
class QGroupBox;

namespace M4D {
namespace GUI {

class StManagerFilterComp: public QWidget
{
  Q_OBJECT

  public:
    StManagerFilterComp ( StManagerStudyListComp *studyListComponent, QWidget *parent = 0 );

  private slots:
    void search ();
    void today ();
    void yesterday ();
    void clear ();

    void from ();
    void to ();
    void all ();
    void modality ();
 
  private:
    QPushButton *createButton ( const QString &text, const char *member );
    QComboBox   *createComboBox ( const QString &text = QString() );
    QCheckBox   *createCheckBox ( const QString &text, bool value, const char *member );

    /// Pointer to the Study List - there will appear filtered results (in it's tables). 
    StManagerStudyListComp *studyListComponent;

    static const char *modalities[];

    // button  column
    QPushButton *searchButton;
    QPushButton *todayButton;
    QPushButton *yesterdayButton;
    QPushButton *clearFilterButton;
    QPushButton *optionsButton;
    // input column
    QComboBox   *patientIDComboBox;
    QComboBox   *lastNameComboBox;
    QComboBox   *firstNameComboBox;
    QCheckBox   *fromDateCheckBox;
    QCheckBox   *toDateCheckBox;
    QDateEdit   *fromDateDateEdit;
    QDateEdit   *toDateDateEdit;
    QComboBox   *accesionComboBox;
    QComboBox   *studyDescComboBox;
    QComboBox   *referringMDComboBox;
    // modalities column
    QGroupBox   *modalitiesGroupBox;
    QCheckBox   *allCheckBox;
    QCheckBox  **modalityCheckBoxes;
};

} // namespace GUI
} // namespace M4D

#endif // S_MANAGER_FILTER_COMP_H

