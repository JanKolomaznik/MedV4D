#ifndef SETTINGS_BOX_H
#define SETTINGS_BOX_H

#include <QtGui>

#include "MultiscanRegistrationFilter.h"


/**
 * Perfusion Studies settings widget.
 */
class SettingsBox: public QWidget
{
	Q_OBJECT

  public:

	  static const unsigned MINIMUM_WIDTH          = 200;
	  static const unsigned SPACING                = 10;
    static const unsigned PARAM_TYPE_BUTTON_SIZE = 40;
    static const unsigned TOOL_BUTTON_SIZE       = 24;

    static const unsigned PARAMETER_TYPE_SIZE = 8;
  
    /**
     * Settings widget constructor.
     *
     * @param registrationFilter pointer to the registration filter of the pipeline
     * @param segmentationFilter pointer to the segmentation filter of the pipeline
     * @param analysisFilter analysis pointer to the filter of the pipeline
     * @param parent pointer to the parent widget
     */
	  SettingsBox ( M4D::Imaging::APipeFilter *registrationFilter,
                  M4D::Imaging::APipeFilter *segmentationFilter,
                  M4D::Imaging::APipeFilter *analysisFilter,
                  QWidget *parent );
  	
    /**
     * Sets whether the execute button is enabled.
     *
     * @param val true if the execute button should be enabled, false otherwise
     */
	  void SetEnabledExecButton ( bool val ) { execButton->setEnabled( val ); }

  protected slots:

    /// Slots of the widget - managing settings, updates of various controls of the widget.

    void InterpolationTypeChanged ( int val );

    void BackgroundValueChanged ( int val );

    void MedianUsageChanged ( bool checked );

    void MedianValueChanged ( int val );

    void SliceNumberValueChanged ( int val );

    void RegistrationNeededChanged ( bool checked );

    void BoneDensityBottomValueChanged ( int val );

    void MaximumTypeSet ();

    void SubtractionTypeSet ();

    void ParameterTypeSet ();

    void DefaultSubtractionChanged ( bool checked );

    void LowIndexChanged ( int val );

    void HighIndexChanged ( int val );

    void PEParameterSet ();

    void TTPParameterSet ();

    void ATParameterSet ();

    void CBVParameterSet ();

    void MTTParameterSet ();

    void CBFParameterSet ();

    void USParameterSet ();

    void DSParameterSet ();

    void MaxValuePercentageChanged ( int value );

    void CutToolSet ( bool checked );

    void CurveToolSet ( bool checked );

	  void ExecuteFilter ();

    void CancelFilter ();

	  void EndOfExecution ();

  signals:

    /// Signals emitted by this widget.

    void VisualizationDone ();

    void SimpleSelected ();

    void ParamaterMapsSelected ();

    void CurveToolSelected ( bool checked );

    void CutToolSelected ( bool checked );

  protected:

    /**
     * Creates all the widgets - called from the constructor.
     */
	  void CreateWidgets ();

    /** 
     * Creates a ParameterTypeButton, connects and configures it.
     *
     * @param icon reference to icon of the button
     * @param member other side of the connection
     * @return pointer to the created and configured ToolButton
     */
    QToolButton *CreateParameterTypeButton ( const QIcon &icon, const char *member );

    /** 
     * Creates a CheckBox and configures it - label, value, connection.
     *
     * @param text reference to string with the value of label
     * @param checked init. value of the CheckBox
     * @param member other side of the connection
     * @return pointer to the created and configured CheckBox
     */
    QCheckBox *CreateCheckBox ( const QString &text, bool checked, const char *member );

    /** 
     * Creates a RadioButton and configures it - label, value, connection.
     *
     * @param text reference to string with the value of label
     * @param checked init. value of the RadioButton
     * @param member other side of the connection
     * @return pointer to the created and configured RadioButton
     */
    QRadioButton *CreateRadioButton ( const QString &text, bool checked, const char *member );

    /** 
     * Creates a ToolButton, connects and configures it.
     *
     * @param icon reference to icon of the button
     * @param member other side of the connection
     * @return pointer to the created and configured ToolButton
     */
    QToolButton *CreateToolButton ( const QIcon &icon, const char *member );

	  /// Pointers to the registration/segmentation/analysis filter of the pipeline.
    M4D::Imaging::APipeFilter *registrationFilter, *segmentationFilter, *analysisFilter;

    /// Pointer to the parent widget.
    QWidget *parent;

    /// Handlers for various controls of the widget.
    
    QSpinBox *medianSpin;

    QCheckBox *defaultSubtractionCheckBox;
    
    QSpinBox *lowIndexSpin, *highIndexSpin;

    QRadioButton *PERadio, *TTPRadio, *ATRadio, *CBVRadio, *MTTRadio, *CBFRadio, *USRadio, *DSRadio;

    QRadioButton *parameterTypes[ PARAMETER_TYPE_SIZE ];

    QSlider *maxValuePercentageSlider;

    QToolButton *cutButton, *curveButton;

	  QPushButton *execButton, *cancelButton;

  private:

    /// Flag indicating whether the pipeline has finished once.
    bool onceFinished;
};


#endif // SETTINGS_BOX_H


