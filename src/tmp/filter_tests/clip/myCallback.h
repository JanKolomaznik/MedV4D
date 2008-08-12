class myCallback : public vtkCommand
{
public:
  vtkImageViewer *IV;
  vtkImageClip* i2;

  static myCallback *New() { return new myCallback; }

  void Execute(vtkObject *caller,
               unsigned long event,
               void *vtkNotUsed(callData))
    {
      if (this->IV->GetInput() == NULL) { return; }

      // Reset

      if (event == vtkCommand::ResetWindowLevelEvent)
        {
        this->IV->GetInput()->UpdateInformation();
        this->IV->GetInput()->SetUpdateExtent(this->IV->GetInput()->GetWholeExtent());
        this->IV->GetInput()->Update();

		this->IV->Render();
        return;
        }

      // Start

      if (event == vtkCommand::StartWindowLevelEvent)
        {
          return;
        }

      // Adjust the window level here

      vtkInteractorStyleImage *isi = static_cast<vtkInteractorStyleImage *>(caller);

      int *size = this->IV->GetRenderWindow()->GetSize();

      // Compute normalized delta

      double dx = (isi->GetWindowLevelCurrentPosition()[0] - isi->GetWindowLevelStartPosition()[0]);
      double dy = (isi->GetWindowLevelStartPosition()[1] - isi->GetWindowLevelCurrentPosition()[1]);

      this->i2->SetOutputWholeExtent(0, dx, 0, dy, 0,1);

      this->IV->Render();
    }
};

