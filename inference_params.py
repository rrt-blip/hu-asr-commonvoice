class DummyParams:
    def __init__(self):
        self.vocab_size = 500
        self.blank_id = 0
        self.context_size = 2
        self.decoder_dim = 512
        self.joiner_dim = 512
        self.feature_dim = 80
        self.encoder_dims = "384,384,384,384,384"
        self.nhead = "8,8,8,8,8"
        self.num_encoder_layers = "2,4,3,2,4"
        self.feedforward_dims = "1024,1024,2048,2048,1024"
        self.attention_dims = "192,192,192,192,192"
        self.encoder_unmasked_dims = "256,256,256,256,256"
        self.zipformer_downsampling_factors = "1,2,4,8,2"
        self.cnn_module_kernels = "31,31,31,31,31"
