#ifndef S_MANAGER_STUDY_LIST_COMP_H
#define S_MANAGER_STUDY_LIST_COMP_H

#include <QWidget>

#include "GUI/m4dGUIVtkRenderWindowWidget.h"

// DICOM includes:
#include "Common.h"          // M4DDICOMServiceProvider.h needs it (FIXME?)
#include "dicomConn/DICOMServiceProvider.h"


class QPushButton;
class QToolButton;
class QTabWidget;
class QTableWidget;
class QSettings;
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
    void path ();

  private:
    void addResultSetToStudyTable ( const M4D::Dicom::DcmProvider::ResultSet *resultSet, 
                                    QTableWidget *table );
    void addRowToStudyTable ( const M4D::Dicom::DcmProvider::TableRow *row,
                              QTableWidget *table );
    void updateRecentExams ( const M4D::Dicom::DcmProvider::TableRow *row, const QString &prefix );
    void loadRecentExams ( M4D::Dicom::DcmProvider::ResultSet &resultSet, const QString &prefix );
    void updateRecentRow ( const M4D::Dicom::DcmProvider::TableRow *row, QSettings &settings );
    void loadRecentRow ( M4D::Dicom::DcmProvider::TableRow &row, const QSettings &settings );

    QTableWidget *createStudyTable ();
    QTreeView    *createDirectoryTreeView ();
    QPushButton  *createButton ( const QString &text, const char *member );
    QToolButton  *createToolButton ( const QIcon &icon );

    /// Pointer to the VTK Render Window Widget - where to render image after clicking View. 
    m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;
    /// Pointer to the Study Manager Dialog - to close it after clicking View.
    QDialog *studyManagerDialog;

    /// Names of the exam/image attributes
    static const char *attributeNames[];

    QPushButton  *viewButton;
    QPushButton  *pathButton;
    QToolButton  *recentRemoteButton;
    QToolButton  *recentDICOMDIRButton;
    QTabWidget   *studyListTab;
    QTableWidget *recentExamsTable;
    QTableWidget *remoteExamsTable;
    QTableWidget *DICOMDIRTable;
    QTableWidget *activeExamTable;
    QTreeView    *directoryTree;

    /// The provider object.
    M4D::Dicom::DcmProvider *dcmProvider;

    /// Pointer to vector of TableRows - result of the Find operation in Recent Exams (remote) mode.
    M4D::Dicom::DcmProvider::ResultSet *recentResultSet;
    /// Pointer to vector of TableRows - result of the Find operation in Remote Exams mode.
    M4D::Dicom::DcmProvider::ResultSet *remoteResultSet;
    /// Pointer to vector of TableRows - result of the Find operation in DICOMDIR mode.
    M4D::Dicom::DcmProvider::ResultSet *DICOMDIRResultSet;
    /// Pointer to vector of TableRows - pointing to active ResultSet.
    M4D::Dicom::DcmProvider::ResultSet *activeResultSet;
};

#endif // S_MANAGER_STUDY_LIST_COMP_H
