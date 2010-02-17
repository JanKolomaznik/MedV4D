#ifndef SETTINGS_BOX_H
#define SETTINGS_BOX_H

#include <QtGui>

#include "MultiscanRegistrationFilter.h"


class SettingsBox: public QWidget
{
	Q_OBJECT

  public:

	  static const unsigned MINIMUM_WIDTH    = 200;
	  static const unsigned SPACING          = 10;
    static const unsigned TOOL_BUTTON_SIZE = 40;
  
	  SettingsBox ( M4D::Imaging::APipeFilter *registrationFilter,
                  M4D::Imaging::APipeFilter *segmentationFilter,
                  M4D::Imaging::APipeFilter *analysisFilter,
                  QWidget *parent );
  	
	  void SetEnabledExecButton ( bool val ) { execButton->setEnabled( val ); }

  protected slots:

    void InterpolationTypeChanged ( int val );

    void BackgroundValueChanged ( int val );

    void SliceNumberValueChanged ( int val );

    void BoneDensityBottomValueChanged ( int val );

    void BoneDensityTopValueChanged ( int val );

    void MaximumTypeSet ();

    void SubtractionTypeSet ();

    void ParameterTypeSet ();

    void DefaultSubtractionChanged ( bool checked );

    void LowIndexChanged ( int val );

    void HighIndexChanged ( int val );

    void TTPParameterSet ();

    void ATParameterSet ();

    void CBVParameterSet ();

    void MTTParameterSet ();

    void CBFParameterSet ();

    void MaxValuePercentageChanged ( int value );

	  void ExecuteFilter ();

    void CancelFilter ();

	  void EndOfExecution ();

  signals:

    void VisualizationDone ();

    void SimpleSelected ();

    void ParamaterMapsSelected ();

  protected:

	  void CreateWidgets ();

    /** 
     * Creates a ToolButton, connects and configures it.
     *
     * @param icon reference to icon of the button
     * @param member other side of the connection
     * @return pointer to the created and configured ToolButton
     */
    QToolButton *CreateToolButton ( const QIcon &icon, const char *member );

    /** 
     * Creates a CheckBox and configures it - label, value, connection.
     *
     * @param text reference to string with the value of label
     * @param checked init. value of the CheckBox
     * @param member other side of the connection
     * @return pointer to the created and configured CheckBox
     */
    QCheckBox *CreateCheckBox ( const QString &text, bool checked, const char *member );

    QRadioButton *CreateRadioButton ( const QString &text, bool checked, const char *member );

	  M4D::Imaging::APipeFilter *registrationFilter, *segmentationFilter, *analysisFilter;

    QWidget *parent;

    QCheckBox *defaultSubtractionCheckBox;
    
    QSpinBox *lowIndexSpin, *highIndexSpin;

    QRadioButton *TTPRadio, *ATRadio, *CBVRadio, *MTTRadio, *CBFRadio;

    QSlider *maxValuePercentageSlider;

	  QPushButton *execButton, *cancelButton;

  private:

    bool onceFinished;
};


#endif // SETTINGS_BOX_H


