#ifndef LINEAR_COMBINATION_OPTIONS_H_
#define LINEAR_COMBINATION_OPTIONS_H_

#include <qdialog.h>

#include "FilteringDialogBase.h"
#include "ui_LinearCombinationOptions.h"

typedef float EigenvalueType;

class LinearCombinationOptions : public FilteringDialogBase<EigenvalueType>
{
public:
  LinearCombinationOptions();

  virtual ~LinearCombinationOptions();

  virtual std::vector<EigenvalueType> GetValues() override
  {
    EigenvalueType sigma = this->dialog->sigmaEditor->value();
    EigenvalueType alpha = this->dialog->alphaEditor->value();
    EigenvalueType beta = this->dialog->betaEditor->value();
    EigenvalueType gamma = this->dialog->gammaEditor->value();

    return std::vector < EigenvalueType > {sigma, alpha, beta, gamma};
  }

private:
  Ui::LinearCombinationOptions* dialog;
};

#endif // LINEAR_COMBINATION_OPTIONS_H_