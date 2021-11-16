# lit-analysis

## Instalar
```
conda env create -f environment.yml
conda activate lit-nlp
conda install -c pytorch pytorch
pip install lit-nlp
```


### Verificar la instalación de LIT

```python -m lit_nlp.examples.glue_demo --port=5432 --quickstart```


Ir a http://localhost:5432 para acceder a la UI de lit.


## Usar LIT con los modelos de pysentimiento:

### [Emotion](https://huggingface.co/finiteautomata/beto-emotion-analysis) 

```python load_beto_emotion.py --port=5432 --data_dir=.```

### [Sentiment](https://huggingface.co/finiteautomata/beto-sentiment-analysis) 

```python load_beto_sentiment.py --port=5432 --data_dir=.```



## Troubleshooting

Debido a una [incompatibilidad](https://github.com/PAIR-code/lit/issues/522) entre versiones de transformers requeridos por lit y pysentimiento pueden ocurrir los siguientes errores:



```
AssertionError: Non-consecutive added token '@usuario' found. Should have index 31006 but has index 31002 in saved vocabulary.
```

Para esto es necesario actualizar la versión de transformers: `pip install -U transformers`

Después de esto, va a inicializar LIT pero en la consola se va a ver el siguiente error:

```
AttributeError: 'TFBertEmbeddings' object has no attribute 'word_embeddings'
```

Y para esto es necesario modificar el archivo `anaconda3/envs/lit-nlp/lib/python3.7/site-packages/lit_nlp/examples/models/glue_models.py`
reemplazando ".word embeddings" por ".weight" en la linea 339. 



