#ifndef S_MANAGER_STUDY_LIST_COMP_H
#define S_MANAGER_STUDY_LIST_COMP_H

#include <QWidget>


class QTabWidget;
class QTableWidget;
class QPushButton;

class StManagerStudyListComp: public QWidget
{
  Q_OBJECT

  public:
    StManagerStudyListComp ( QWidget *parent = 0 );

  private:
    QTableWidget *createStudyTable ();
    QPushButton  *createButton ( const QString &text, const char *member );

    QPushButton  *viewButton;
    QPushButton  *deleteButton;
    QPushButton  *sendButton;
    QPushButton  *queueFilterButton;
    QPushButton  *burnToMediaButton;
    QTabWidget   *studyListTab;
    QTableWidget *localExamsTable;
    QTableWidget *remoteExamsTable;
};

#endif // S_MANAGER_STUDY_LIST_COMP_H