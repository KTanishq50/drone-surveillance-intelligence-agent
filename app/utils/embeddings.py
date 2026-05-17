from fastembed import TextEmbedding
import config

_model = None

def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=config.EMBED_MODEL)
    return _model

def embed(text: str) -> list:
    model = get_model()
    result = list(model.embed([text]))
    return result[0].tolist()