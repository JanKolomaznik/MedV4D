#ifndef M4D_GUI_STUDY_MANAGER_WIDGET_H
#define M4D_GUI_STUDY_MANAGER_WIDGET_H

#include <QWidget>
#include <QMainWindow>


class QPushButton;
class QComboBox;
class QGroupBox;
class QTabWidget;
class QTableWidget;

class m4dGUIStudyManagerWidget: public QWidget
{
  Q_OBJECT

  public:
    m4dGUIStudyManagerWidget ( QMainWindow *parent = 0 );

  private slots:
    void search ();

  private:
    void createFilterGroupBox ();
    void createHangingProtocolsGroupBox ();
    void createStudyListGroupBox ();

    QPushButton *createButton ( const QString &text, const char *member );
    QComboBox *createComboBox ( const QString &text = QString() );

    QGroupBox   *filterGroupBox;
    QPushButton *searchButton;
    QPushButton *todayButton;
    QPushButton *yesterdayButton;
    QPushButton *clearFilterButton;
    QPushButton *optionsButton;
    QComboBox   *patientIDComboBox;
    QComboBox   *lastNameComboBox;
    QComboBox   *firstNameComboBox;

    QGroupBox    *hangingProtocolsGroupBox;

    QGroupBox    *studyListGroupBox;
    QPushButton  *viewButton;
    QPushButton  *deleteButton;
    QPushButton  *sendButton;
    QPushButton  *queueFilterButton;
    QPushButton  *burnToMediaButton;
    QTabWidget   *studyListTab;
    QTableWidget *localExamsTable;
    QTableWidget *remoteExamsTable;
};

#endif // M4D_GUI_STUDY_MANAGER_WIDGET_H