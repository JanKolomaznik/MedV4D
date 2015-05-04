#include "AdditionalDatasetOptions.h"

AdditionalDatasetOptions::AdditionalDatasetOptions() : dialog(new Ui::AdditionalDatasetOptions())
{
  this->dialog->setupUi(this);
}

AdditionalDatasetOptions::~AdditionalDatasetOptions()
{
  delete this->dialog;
}