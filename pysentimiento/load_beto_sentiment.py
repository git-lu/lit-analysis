from absl import app
from lit_nlp import dev_server, server_flags
from lit_nlp.examples.models.glue_models import GlueModel

from tass_data import TASSSentimentData, TASSSentLabels


project_path = '/home/work/Documents/lit-analysis/pysentimiento'


class BetoSentiment(GlueModel):
    """Beto Sentiment Classification model on TASS."""

    def __init__(self, *args, **kw):
        super().__init__(
            *args,
            text_a_name="tweet",
            text_b_name=None,
            labels=TASSSentLabels,
            **kw)


def main(_):
    datasets = {
        'tass_sentiment_2020_train': TASSSentimentData(project_path + '/data/sent/train/'),
        'tass_sentiment_2020_dev': TASSSentimentData(project_path + '/data/sent/dev/'),
    }
    models = {
        'beto_sentiment': BetoSentiment('finiteautomata/beto-sentiment-analysis'),
    }
    lit_demo = dev_server.Server(
        models, datasets, **server_flags.get_flags())
    dev_server.Server(models, datasets, )
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
