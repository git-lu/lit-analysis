# lit-analysis

## Instalar
```
conda env create -f environment.yml
conda activate lit-nlp
conda install -c pytorch pytorch
pip install lit-nlp
```

## Troubleshooting
Debido a una (incompatibilidad con la librería pysentimiento y lit)[https://github.com/PAIR-code/lit/issues/522] es necesario
realizar los siguientes pasos para que todo funcione bien: 


```
AssertionError: Non-consecutive added token '@usuario' found. Should have index 31006 but has index 31002 in saved vocabulary.
```
Para esto es necesario actualizar la versión de transformers: `pip install -U transformers`

Después de esto, va a inicializar LIT pero en la consola se va a ver el siguiente error:
`AttributeError: 'TFBertEmbeddings' object has no attribute 'word_embeddings'`

Y para esto es necesario modificar el archivo `anaconda3/envs/lit-nlp/lib/python3.7/site-packages/lit_nlp/examples/models/glue_models.py`
reemplazando ".word embeddings" por ".weight" en la linea 339


### Verificar la instalación de LIT

```python -m lit_nlp.examples.glue_demo --port=1234 --quickstart```


Ir a http://localhost:5432 para acceder a la UI de lit.

