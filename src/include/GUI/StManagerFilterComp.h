#ifndef S_MANAGER_FILTER_COMP_H
#define S_MANAGER_FILTER_COMP_H

#include <QWidget>

#include "StManagerStudyListComp.h"


class QPushButton;
class QComboBox;
class QCheckBox;
class QDateEdit;

class StManagerFilterComp: public QWidget
{
  Q_OBJECT

  public:
    StManagerFilterComp ( StManagerStudyListComp *studyListComponent, QWidget *parent = 0 );

  private slots:
    void search ();
    void today ();
    void yesterday ();

    void from ();
    void to ();
 
  private:
    QPushButton *createButton ( const QString &text, const char *member );
    QComboBox   *createComboBox ( const QString &text = QString() );
    QCheckBox   *createCheckBox ( const QString &text, const char *member );

    /// Pointer to the Study List - there will appear filtered results (in it's tables). 
    StManagerStudyListComp *studyListComponent;

    QPushButton *searchButton;
    QPushButton *todayButton;
    QPushButton *yesterdayButton;
    QPushButton *clearFilterButton;
    QPushButton *optionsButton;
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
};

#endif // S_MANAGER_FILTER_COMP_H