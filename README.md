# SST-
A Transformer-based Method for Correcting Daily SST Numerical Forecasting products

## Project Description:

This project proposes a Transformer-based method for revising daily sea surface temperature (SST) numerical forecast products. Traditional SST forecasting methods usually face problems such as model errors and insufficient observational data, etc. This project adopts the advanced Transformer structure and combines spatio-temporal features for data revision and accuracy improvement. With this method, it aims to improve the accuracy and reliability of numerical forecast products, especially in the fields of marine environment monitoring, climate prediction and navigation safety, which have a wide range of application prospects.

## Project structure:

```yaml
/code                          # Code section
|-- Geoformer.py                # Implementation of the Transformer model
|-- SST.py                       # Data preprocessing and augmentation code, including data loading, normalization, etc.
|-- my_tools.py                 # Utility functions, including details of the Transformer structure implementation
|-- myconfig.py                 # Configuration file, including training parameters, hyperparameters, etc.
|-- trainer_2.py                # Main entry point for model training

/data                          # Data section
|-- SST.nc                      # Raw SST dataset, including ERA5 reanalysis data, model data, etc.
|-- err.nc                      # Processed error dataset

```

## Additional notes：

Due to the storage limitations of the website, our original dataset is uploaded on Google Drive. The link is: https://drive.google.com/file/d/1X8z-_1N4pU0YM_XMATPWHaqxc38ENmds/view?usp=drive_link



