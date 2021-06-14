FROM jupyter/datascience-notebook
COPY . /app
RUN wget https://zenodo.org/api/files/6008ca9e-bf65-4c35-8b06-992dbd7a1bf8/Linux.csv app/datasets/Linux.csv
WORKDIR /app
ENTRYPOINT ["jupyter","notebook"]