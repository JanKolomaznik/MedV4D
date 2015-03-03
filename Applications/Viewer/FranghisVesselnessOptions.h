#ifndef FRANGHIS_VESSELNESS_OPTIONS_H_
#define FRANGHIS_VESSELNESS_OPTIONS_H_

#include <vector>

#include "FilteringDialogBase.h"
#include "ui_FranghisVesselnessOptions.h"

typedef float EigenvalueType;

class FranghisVesselnessOptions : public FilteringDialogBase<EigenvalueType>
{
public:
  FranghisVesselnessOptions();

  virtual ~FranghisVesselnessOptions();

  virtual std::vector<EigenvalueType> GetValues() override
  {
    EigenvalueType sigma = this->dialog->sigmaEditor->value();
    EigenvalueType alpha = this->dialog->alphaEditor->value();
    EigenvalueType beta = this->dialog->betaEditor->value();
    EigenvalueType gamma = this->dialog->gammaEditor->value();

    return std::vector < EigenvalueType > {sigma, alpha, beta, gamma};
  }

private:
  Ui::FranghisVesselnessOptions* dialog;
};

#endif //FRANGHIS_VESSELNESS_OPTIONS_H_