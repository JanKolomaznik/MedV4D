#ifndef ST_MANAGER_STUDY_LIST_COMP_H
#define ST_MANAGER_STUDY_LIST_COMP_H

#include <QWidget>

// DICOM includes:
#include "Common.h"
#include "dicomConn/DICOMServiceProvider.h"


class QPushButton;
class QToolButton;
class QTabWidget;
class QTableWidget;
class QSettings;
class QTreeView;

namespace M4D {
namespace GUI {

/**
 * Class representing one of the base components of Study Manager Widget.
 * It manages the uniform search result viewing and manipulation - Recent Exams (remote and DICOMDIR), 
 * Remote Exams, DICOMDIR. 
 */
class StManagerStudyListComp: public QWidget
{
  Q_OBJECT

  public:

    /** 
     * Study Manager Study List Component constructor.
     *
     * @param studyManagerDialog pointer to the Study Manager Dialog - to change its title
     * @param parent pointer to the parent widget - default is 0
     */
    StManagerStudyListComp ( QDialog *studyManagerDialog, QWidget *parent = 0 );
    
    /** 
     * Study Manager Study List Component destructor.
     */
    ~StManagerStudyListComp ();

    /** 
     * Find called from Filter Component - depending on mode it's performing different searching actions.
     * 
     * @param firstName reference to string containing first name of the patient
     * @param lastName reference to string containing last name of the patient
     * @param patientID patient ID search mask
     * @param fromDate reference to string containing date (from) in yyyyMMdd format
     * @param toDate reference to string containing date (to) in yyyyMMdd format 
     * @param modalitiesVect reference to vector of strings containing set of wanted modalities
     * @param referringMD reference to string containing referring MD
     * @param description reference to string containing description of the study
     * @throws ExceptionBase, exception -> TOWRITE
     */
    void find ( const std::string &firstName, const std::string &lastName, 
                const std::string &patientID, 
                const std::string &fromDate, const std::string &toDate,
                const M4D::Dicom::DcmProvider::StringVector &modalitiesVect,
                const std::string &referringMD, const std::string &description );

    /** 
     * Sets pointer to DicomObjSet - result of the Study Manager appears there.
     * 
     * @param dcmObjSet DicomObjSet to fill
     */
    void setDicomObjectSet ( M4D::Dicom::DcmProvider::DicomObjSet *dcmObjSet )
    {
      dicomObjectSet = dcmObjSet;  
    }
 
  private slots:

    /**
     * Slot for View button - view the selected exam (depending on mode - recent, remote, DICOMDIR).
     */
    void view ();

    /**
     * Slot for managing View button - when to enable it.
     */
    void setEnabledView ();

    /**
     * Slot for managing tabchange - to switch some buttons and variables.
     */
    void activeTabChanged ();

    /**
     * Slot for managing Recent Exams mode change - to clear tables and variables.
     */
    void recentChanged ();

    /**
     * Slot for directory tree behavior - to hide or show it.
     */
    void path ();

  signals:

    /**
     * Signal for indicating wheather the view button was pushed - to clese the dialog.
     */
    void ready ();

  private:

    /** 
     * Loads Recent Exams to ResultSet - it's using Settings mechanism of Qt -
     * depending on type of the exam - remote, DICOMDIR.
     * 
     * @param resultSet reference to ResultSet - the result of the load
     * @param prefix prefix for group of settings (to identify them) - same for one type of exam
     */
    void loadRecentExams ( M4D::Dicom::DcmProvider::ResultSet &resultSet, const QString &prefix );
    
    /** 
     * Loads a TableRow from specific QSettings
     * 
     * @param row reference to TableRow - the result of the load
     * @param settings reference to settings - where is the wanted row
     */
    void loadRecentRow ( M4D::Dicom::DcmProvider::TableRow &row, const QSettings &settings );

    /** 
     * Adds result of Find - ResultSet - to Study Table - depending on searching mode.
     * 
     * @param resultSet pointer to ResultSet to be added
     * @param table pointer to table where to display results
     */
    void addResultSetToStudyTable ( const M4D::Dicom::DcmProvider::ResultSet *resultSet, 
                                    QTableWidget *table );

    /** 
     * Adds result (one row) of Find - TableRow - to Study Table - depending on searching mode.
     * 
     * @param row pointer to TableRow to be added
     * @param table pointer to table where to display results
     */
    void addRowToStudyTable ( const M4D::Dicom::DcmProvider::TableRow *row,
                              QTableWidget *table );

    /** 
     * Gets index of selected series - creates modal dialog according to SerieInfoVector
     * to choose from if there are more than one series in the study.
     * 
     * @param info SerieInfoVector from which is the dialog created (descriptions of studies)
     * @return the index of selected series 
     */
    unsigned getSeriesIndex( const M4D::Dicom::DcmProvider::SerieInfoVector info );

    /** 
     * Updates Recent Exams by currently viewed one - saves it using Settings mechanism of Qt -
     * depending on type of the exam - remote, DICOMDIR.
     * 
     * @param row pointer to TableRow to be added - currently viewed
     * @param prefix prefix for group of settings (to identify them) - same for one type of exam
     */
    void updateRecentExams ( const M4D::Dicom::DcmProvider::TableRow *row, const QString &prefix );

    /** 
     * Saves a given TableRow to specific QSettings.
     * 
     * @param row pointer to TableRow to be saved
     * @param settings reference to settings - where to save the row
     */
    void updateRecentRow ( const M4D::Dicom::DcmProvider::TableRow *row, QSettings &settings );


    /** 
     * Creates a StudyTable and configures it.
     */
    QTableWidget *createStudyTable ();

    /** 
     * Creates a DirectoryTreeView and configures it.
     */
    QTreeView    *createDirectoryTreeView ();

    QTableWidget *createSeriesSelectionTable ();

    /** 
     * Creates a Button and connects it with given member.
     *
     * @param text reference to caption string
     * @param member other side of the connection
     */
    QPushButton  *createButton ( const QString &text, const char *member );

    /** 
     * Creates a ToolButton and and configures it.
     *
     * @param icon reference to icon of the button
     */
    QToolButton  *createToolButton ( const QIcon &icon, const char *member );


    /// Pointer to the Study Manager Dialog - to change its title.
    QDialog *studyManagerDialog;
    /// Title of the Study Manager Dialog - depending on searching mode and find results.
    QString studyManagerDialogTitle;

    /// Names of the exam/image attributes.
    static const char *attributeNames[];

    /// Buttons for viewing, switching between modes, hiding config. parts.
    QPushButton  *viewButton;
    QPushButton  *pathButton;
    QToolButton  *recentRemoteButton;
    QToolButton  *recentDICOMDIRButton;
    /// Tabs for different searching modes.
    QTabWidget   *studyListTab;
    /// Result tables for different searching modes.
    QTableWidget *recentExamsTable;
    QTableWidget *remoteExamsTable;
    QTableWidget *DICOMDIRTable;
    QTableWidget *activeExamTable;
    /// Directory tree for browsing in DICOMDIR mode.
    QTreeView    *directoryTree;

    /// The provider object - communication with DICOM layer.
    M4D::Dicom::DcmProvider *dcmProvider;

    /// Pointer to vector of TableRows - result of the Find operation in Recent Exams (remote) mode.
    M4D::Dicom::DcmProvider::ResultSet *recentResultSet;
    /// Pointer to vector of TableRows - result of the Find operation in Remote Exams mode.
    M4D::Dicom::DcmProvider::ResultSet *remoteResultSet;
    /// Pointer to vector of TableRows - result of the Find operation in DICOMDIR mode.
    M4D::Dicom::DcmProvider::ResultSet *DICOMDIRResultSet;
    /// Pointer to vector of TableRows - pointing to active ResultSet.
    M4D::Dicom::DcmProvider::ResultSet *activeResultSet;

    M4D::Dicom::DcmProvider::DicomObjSet *dicomObjectSet;	
};

} // namespace GUI
} // namespace M4D

#endif // ST_MANAGER_STUDY_LIST_COMP_H
