#ifndef S_MANAGER_STUDY_LIST_COMP_H
#define S_MANAGER_STUDY_LIST_COMP_H

#include <QWidget>

#include "m4dGUIVtkRenderWindowWidget.h"

// DICOM includes:
#include "Common.h"          // M4DDICOMServiceProvider.h needs it (FIXME?)
#include "dicomConn/DICOMServiceProvider.h"


class QTabWidget;
class QTableWidget;
class QPushButton;

class StManagerStudyListComp: public QWidget
{
  Q_OBJECT

  public:
    StManagerStudyListComp ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                             QWidget *parent = 0 );
    ~StManagerStudyListComp ();

    void find ( const QString &patientName, const QString &patientID, 
                const QString &fromDate, const QString &toDate );

  private slots:
    void view ();

  private:
    QTableWidget *createStudyTable ();
    QPushButton  *createButton ( const QString &text, const char *member );

    void          addResultSetToStudyTable ( QTableWidget *table );
    void          addRowToStudyTable ( const M4D::Dicom::DcmProvider::TableRow *row,
                                       QTableWidget *table );

    /// Pointer to the VTK Render Window Widget - where to render image after clicking View. 
    m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;

    QPushButton  *viewButton;
    QPushButton  *deleteButton;
    QPushButton  *sendButton;
    QPushButton  *queueFilterButton;
    QPushButton  *burnToMediaButton;
    QTabWidget   *studyListTab;
    QTableWidget *localExamsTable;
    QTableWidget *remoteExamsTable;

    /// The provider object.
    M4D::Dicom::DcmProvider *dcmProvider;

    /// ResultSet - vector of TableRows - result of Find operation.
    M4D::Dicom::DcmProvider::ResultSet *resultSet;
};

#endif // S_MANAGER_STUDY_LIST_COMP_H
