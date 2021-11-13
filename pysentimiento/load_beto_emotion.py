from absl import app
from lit_nlp import dev_server, server_flags
from lit_nlp.examples.models.glue_models import GlueModel

from tass_data import TASSEmotionLabels, TASSEmotionsData


project_path = '/home/work/Documents/text-mining/proyecto-final'

class BetoEmotion(GlueModel):
    """Beto Emotions Classification model on TASS."""

    def __init__(self, *args, **kw):
        super().__init__(
            *args,
            text_a_name="tweet",
            text_b_name=None,
            labels=TASSEmotionLabels,
            **kw)


def main(_):
    datasets = {
        'tass_emotion_2020_train': TASSEmotionsData(project_path + '/data/emotions/train.tsv'),
        'tass_emotion_2020_dev': TASSEmotionsData(project_path + '/data/emotions/dev.tsv')

    }
    models = {
        'beto_emotions': BetoEmotion('finiteautomata/beto-emotion-analysis'),
    }
    lit_demo = dev_server.Server(
        models, datasets, **server_flags.get_flags())
    dev_server.Server(models, datasets, )
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
