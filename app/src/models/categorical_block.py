from tensorflow.keras import layers as tfl


class CategoricalBlock(tfl.Layer):
    """Encapsula el procesamiento de variables categóricas mediante embeddings."""
    def __init__(self, ticker_vocab, sector_vocab, ticker_embedding_dim=16, sector_embedding_dim=4, **kwargs):
        super().__init__(**kwargs)
        # Ticker Path: StringLookup + Embedding
        self.ticker_lookup = tfl.StringLookup(vocabulary=ticker_vocab, mask_token=None)
        self.ticker_emb = tfl.Embedding(len(ticker_vocab) + 1, ticker_embedding_dim, name="ticker_emb")
        
        # Sector Path: StringLookup + Embedding
        self.sector_lookup = tfl.StringLookup(vocabulary=sector_vocab, mask_token=None)
        self.sector_emb = tfl.Embedding(len(sector_vocab) + 1, sector_embedding_dim, name="sector_emb")
        
        self.flatten = tfl.Flatten()
        self.concat = tfl.Concatenate()

    def call(self, ticker_input, sector_input):
        t_id = self.ticker_lookup(ticker_input)
        t_vec = self.ticker_emb(t_id)
        
        s_id = self.sector_lookup(sector_input)
        s_vec = self.sector_emb(s_id)
        
        return self.concat([self.flatten(t_vec), self.flatten(s_vec)])