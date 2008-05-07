#ifndef S_MANAGER_STUDY_LIST_COMP_H
#define S_MANAGER_STUDY_LIST_COMP_H

#include <QWidget>

// DICOM includes:
#include "Common.h"          // M4DDICOMServiceProvider.h needs it (FIXME?)
#include "M4DDICOMServiceProvider.h"


class QTabWidget;
class QTableWidget;
class QPushButton;

class StManagerStudyListComp: public QWidget
{
  Q_OBJECT

  public:
    StManagerStudyListComp ( QWidget *parent = 0 );

    void addResultSetToStudyTable ( const M4D::Dicom::DcmProvider::ResultSet *resultSet );

  private:
    QTableWidget *createStudyTable ();
    void          addRowToStudyTable ( const M4D::Dicom::DcmProvider::TableRow *row );
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
