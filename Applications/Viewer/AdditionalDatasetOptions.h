#ifndef ADDITIONAL_DATASET_OPTIONS_H_
#define ADDITIONAL_DATASET_OPTIONS_H_

#include <vector>

#include "FilteringDialogBase.h"
#include "ui_AdditionalDatasetOptions.h"

typedef double EigenvalueType;

class AdditionalDatasetOptions : public FilteringDialogBase<EigenvalueType>
{
public:
  AdditionalDatasetOptions();

  virtual ~AdditionalDatasetOptions();

  virtual std::vector<EigenvalueType> GetValues() override
  {
    EigenvalueType sigma = this->dialog->sigmaEditor->value();
    EigenvalueType alpha = this->dialog->alphaEditor->value();
    EigenvalueType beta = this->dialog->betaEditor->value();
    EigenvalueType gamma = this->dialog->gammaEditor->value();

    return std::vector < EigenvalueType > {sigma, alpha, beta, gamma};
  }

private:
  Ui::AdditionalDatasetOptions* dialog;
};

#endif //ADDITIONAL_DATASET_OPTIONS_H_