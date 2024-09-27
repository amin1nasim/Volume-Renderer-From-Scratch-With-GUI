import torch

class Embedder(torch.nn.Module):
    def __init__(self, 
                num_freqs,
                max_freq_log2,
                include_input = True,
                input_dims=3,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos], **kwargs):
        super().__init__()
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs
        
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def output_dim(self):
        return self.out_dim
    
    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)