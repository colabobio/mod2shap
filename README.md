## Model export to TF

The test notebook saves the logistic regression model specified in the file model.csv, which was trained on the data in the file data.csv, as a Keras model in TensorFlow Lite format.

First, create a conda environment with:

```
> conda create --name tfenv --file requirements.txt --channel conda-forge python=3.12 jupyterlab
```

Then just run the notebook.