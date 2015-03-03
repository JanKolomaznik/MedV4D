#include "LinearCombinationOptions.h"

LinearCombinationOptions::LinearCombinationOptions() : dialog(new Ui::LinearCombinationOptions())
{
  this->dialog->setupUi(this);
}

LinearCombinationOptions::~LinearCombinationOptions()
{
  delete this->dialog;
}