/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file StManagerFilterComp.h 
 * @{ 
 **/

#ifndef ST_MANAGER_FILTER_COMP_H
#define ST_MANAGER_FILTER_COMP_H

#include <QWidget>

#include "GUI/StManagerStudyListComp.h"


class QPushButton;
class QComboBox;
class QCheckBox;
class QDateEdit;
class QGroupBox;

namespace M4D {
namespace GUI {

/**
 * Class representing one of the base components of Study Manager Widget.
 * It provides filtering functionality - depending on searching mode - Recent Exams (remote and DICOMDIR), 
 * Remote Exams, DICOMDIR modes. Filters by various attributes, predefined patterns, clear option, etc.
 */
class StManagerFilterComp: public QWidget
{
  Q_OBJECT

  public:

    /** 
     * Study Manager Filter Component constructor.
     *
     * @param studyListComponent pointer to the Study List (other component) - there will appear
     * filtered results (in its tables).
     * @param parent pointer to the parent widget - default is 0
     */
    StManagerFilterComp ( StManagerStudyListComp *studyListComponent, QWidget *parent = 0 );

  private slots:

    /**
     * Search slot - for starting search - depending on mode and filter settings. It's calling
     * Study List Component's find method.
     */
    void search ();

    /**
     * Today slot - for enabling (if they are disabled) and setting dateCombos to actual date by
     * clicking on 'Today' button.
     */
    void today ();

    /**
     * Yesterday slot - for enabling (if they are disabled) and setting dateCombos to yesterday 
     * date by clicking on 'Yesterday' button.
     */
    void yesterday ();

    /**
     * Clear slot - for clearing inputs, checkBoxes, dateCombos - filtering settings.
     */
    void clear ();


    /**
     * From slot - for enabling/disabling From Date Edit. Disabled -> no restriction.
     */
    void from ();

    /**
     * To slot - for enabling/disabling To Date Edit. Disabled -> no restriction.
     */
    void to ();

    /**
     * All slot - for All modalities checkBox behavior - checking/unchecking all modalities.
     */
    void all ();

    /**
     * Modality slot - for All modalities checkBox behavior.
     */
    void modality ();
 
  private:

    /** 
     * Creates a Button and connects it with given member.
     *
     * @param text reference to caption string
     * @param member other side of the connection
     * @return pointer to the created and configured Button
     */
    QPushButton *createButton ( const QString &text, const char *member );

    /** 
     * Creates a ComboBox and configures it - with history and auto-completion features.
     *
     * @param text reference to string with the value of init. item - shown on ComboBox at the beginning -
     * before adding anything else - default is empty string
     * @return pointer to the created and configured Directory ComboBox
     */
    QComboBox   *createComboBox ( const QString &text = QString() );

    /** 
     * Creates a CheckBox and configures it - label, value, connection.
     *
     * @param text reference to string with the value of label
     * @param value init. value of the CheckBox
     * @param member other side of the connection
     * @return pointer to the created and configured CheckBox
     */
    QCheckBox   *createCheckBox ( const QString &text, bool value, const char *member );

    /// Pointer to the Study List - there will appear filtered results (in its tables). 
    StManagerStudyListComp *studyListComponent;

    /// Values and labels of possible modalities.
    static const char *modalities[];

    /// Button column GUI items
    QPushButton *searchButton;
    QPushButton *todayButton;
    QPushButton *yesterdayButton;
    QPushButton *clearFilterButton;
    QPushButton *optionsButton;
    /// Input column GUI items
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
    /// Modalities column GUI items
    QGroupBox   *modalitiesGroupBox;
    QCheckBox   *allCheckBox;
    QCheckBox  **modalityCheckBoxes;
};

} // namespace GUI
} // namespace M4D

#endif // ST_MANAGER_FILTER_COMP_H


/** @} */

