# A Comparison of Performance Specialization Learning techniques for Configurable Systems

All results tables are available in tables.pdf

The "Automated Performance Specialization.ipynb" notebook contains the explaination through code of all 3 developped approaches for performance specialization.

The "Experiments.ipynb" notebook contains the scripts made for the experiments and the latex table generation. The "experiments.py" is an equivalent outside of a notebook.

The "Results" notebook allow to recreate the table from the paper from raw results data.

# Linux dataset

The dataset for Linux is too big for Github, we made it available on Zenodo : https://zenodo.org/record/4943884

It is possible to directly download the dataset :

`wget https://zenodo.org/api/files/6008ca9e-bf65-4c35-8b06-992dbd7a1bf8/Linux.csv`

## Docker

We also provide a Docker image to ensure the possibility to run the experiments witht the original packages.

It is available on Docker Hub, accessible with that command : 

`docker run -i -p 8888:8888 hmartinirisa/splc2021`

This will run a Jupyter server that will allow to run the differents notebooks.


It is also possible to build the image locally with the content of this repository : 

`docker build -t splc2021 .`

To run the local image : 

`docker run -i -p 8888:8888 splc2021`