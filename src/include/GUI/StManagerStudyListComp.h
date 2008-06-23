#ifndef S_MANAGER_STUDY_LIST_COMP_H
#define S_MANAGER_STUDY_LIST_COMP_H

#include <QWidget>

#include "GUI/m4dGUIVtkRenderWindowWidget.h"

// DICOM includes:
#include "Common.h"          // M4DDICOMServiceProvider.h needs it (FIXME?)
#include "dicomConn/DICOMServiceProvider.h"


class QPushButton;
class QTabWidget;
class QTableWidget;
class QTreeView;

class StManagerStudyListComp: public QWidget
{
  Q_OBJECT

  public:
    StManagerStudyListComp ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                             QDialog *studyManagerDialog, QWidget *parent = 0 );
    ~StManagerStudyListComp ();

    void find ( const std::string &firstName, const std::string &lastName, 
                const std::string &patientID, 
                const std::string &fromDate, const std::string &toDate,
                const M4D::Dicom::DcmProvider::StringVector &modalitiesVect );

  private slots:
    void view ();
    void setEnabledView ();
    void activeTabChanged ();

  private:
    void addResultSetToStudyTable ( QTableWidget *table );
    void addRowToStudyTable ( const M4D::Dicom::DcmProvider::TableRow *row,
                              QTableWidget *table );

    QTableWidget *createStudyTable ();
    QPushButton  *createButton ( const QString &text, const char *member );

    /// Pointer to the VTK Render Window Widget - where to render image after clicking View. 
    m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;
    /// Pointer to the Study Manager Dialog - to close it after clicking View.
    QDialog *studyManagerDialog;

    QPushButton  *viewButton;
    QPushButton  *deleteButton;
    QPushButton  *sendButton;
    QPushButton  *queueFilterButton;
    QTabWidget   *studyListTab;
    QTableWidget *recentExamsTable;
    QTableWidget *remoteExamsTable;
    QTableWidget *DICOMDIRTable;
    QTableWidget *activeExamTable;
    QTreeView    *directoryTree;

    /// The provider object.
    M4D::Dicom::DcmProvider *dcmProvider;

    /// Pointer to vector of TableRows - result of the Find operation in Recent Exams mode.
    M4D::Dicom::DcmProvider::ResultSet *recentResultSet;
    /// Pointer to vector of TableRows - result of the Find operation in Remote Exams mode.
    M4D::Dicom::DcmProvider::ResultSet *remoteResultSet;
    /// Pointer to vector of TableRows - result of the Find operation in DICOMDIR mode.
    M4D::Dicom::DcmProvider::ResultSet *DICOMDIRResultSet;
    /// Pointer to vector of TableRows - pointing to active ResultSet.
    M4D::Dicom::DcmProvider::ResultSet *activeResultSet;
};

#endif // S_MANAGER_STUDY_LIST_COMP_H
