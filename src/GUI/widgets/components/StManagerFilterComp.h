#ifndef S_MANAGER_FILTER_COMP_H
#define S_MANAGER_FILTER_COMP_H

#include <QWidget>


class QPushButton;
class QComboBox;
class QCheckBox;
class QDateEdit;

class StManagerFilterComp: public QWidget
{
  Q_OBJECT

  public:
    StManagerFilterComp ( QWidget *parent = 0 );

  private slots:
    void search ();

    void fromCheck ();
    void toCheck ();

  private:
    QPushButton *createButton ( const QString &text, const char *member );
    QComboBox   *createComboBox ( const QString &text = QString() );
    QCheckBox   *createCheckBox ( const QString &text, const char *member );

    QPushButton *searchButton;
    QPushButton *todayButton;
    QPushButton *yesterdayButton;
    QPushButton *clearFilterButton;
    QPushButton *optionsButton;
    QComboBox   *patientIDComboBox;
    QComboBox   *lastNameComboBox;
    QComboBox   *firstNameComboBox;
    QDateEdit   *fromDateDateEdit;
    QDateEdit   *toDateDateEdit;
    QComboBox   *accesionComboBox;
    QComboBox   *studyDescComboBox;
    QComboBox   *referringMDComboBox;
};

#endif // S_MANAGER_FILTER_COMP_H