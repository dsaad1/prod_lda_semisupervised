import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro import poutine

class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        # NB: here we set `affine=False` to reduce the number of learning parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # for the effect of this flag in BatchNorm1d
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale+0.0001
    
    
    
class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)
    
    
    
class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, num_prototypes, hidden, dropout, device, frozen=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_prototypes = num_prototypes
        self.device = device
        self.frozen = frozen
        self.encoder = Encoder(vocab_size, self.num_topics, hidden, dropout).to(self.device)
        self.decoder = Decoder(vocab_size, self.num_topics, dropout).to(self.device)
        self.logtheta_loc_p = torch.zeros(self.num_prototypes, self.num_topics, device=self.device)
        self.logtheta_scale_p = torch.ones(self.num_prototypes, self.num_topics, device=self.device)
        
        # if frozen:
        #     self.eval()

    def model(self, docs_a, docs_b, c, observed, x=None):
        if not self.frozen:
            pyro.module("decoder", self.decoder)
        
        logtheta_loc = docs_a.new_zeros((docs_a.shape[0], self.num_topics))
        logtheta_scale = docs_a.new_ones((docs_a.shape[0], self.num_topics))
        with pyro.plate("documents_a", docs_a.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_a = pyro.sample(
                "logtheta_a", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta_a = F.softmax(logtheta_a, -1)

            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param_a = self.decoder(theta_a)
            # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
            # Because the numbers of words across documents can vary,
            # we will use the maximum count accross documents here.
            # This does not affect the result because Multinomial.log_prob does
            # not require `total_count` to evaluate the log probability.
            total_count_a = int(docs_a.sum(-1).max())
            pyro.sample(
                'obs_a',
                dist.Multinomial(total_count_a, count_param_a),
                obs=docs_a
            )
        with pyro.plate("documents_b", docs_b.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_b = pyro.sample(
                "logtheta_b", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta_b = F.softmax(logtheta_b, -1)

            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param_b = self.decoder(theta_b)
            # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
            # Because the numbers of words across documents can vary,
            # we will use the maximum count accross documents here.
            # This does not affect the result because Multinomial.log_prob does
            # not require `total_count` to evaluate the log probability.
            total_count_b = int(docs_b.sum(-1).max())
            pyro.sample(
                'obs_b',
                dist.Multinomial(total_count_b, count_param_b),
                obs=docs_b
            )
        
        pyro.sample("prototypes", dist.Normal(self.logtheta_loc_p, self.logtheta_scale_p).to_event(1))
        
        if x is not None:        
            valid_pairs = c.clone()
            valid_pairs[~observed] = 0  # ensure all values are valid
            with poutine.mask(mask=observed):
                pyro.sample(
                    'obs_c',
                    dist.Bernoulli(logits=x),
                    obs=valid_pairs
                )


    def guide(self, docs_a, docs_b, c, observed, x=None):
        if not self.frozen:
            pyro.module("encoder", self.encoder)
        with pyro.plate("documents_a", docs_a.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            logtheta_loc_a, logtheta_scale_a = self.encoder(docs_a)
            logtheta_a = pyro.sample(
                "logtheta_a", dist.Normal(logtheta_loc_a, logtheta_scale_a).to_event(1))
        with pyro.plate("documents_b", docs_b.shape[0]): 
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            logtheta_loc_b, logtheta_scale_b = self.encoder(docs_b)
            logtheta_b = pyro.sample(
                "logtheta_b", dist.Normal(logtheta_loc_b, logtheta_scale_b).to_event(1)) 
        
        
        p = pyro.param('p', torch.randn(self.num_prototypes, self.num_topics, device=self.device))
        p = pyro.sample("prototypes", dist.Delta(p).to_event(1))
        
        a = pyro.param('a', torch.tensor(1.0, device=self.device))
        b = pyro.param('b', torch.tensor(0.0, device=self.device))
        
        
        sA = logtheta_loc_a @ p.t()
        sB = logtheta_loc_b @ p.t()
        x =  pyro.deterministic("x", a * (sA * sB).sum(dim=-1) + b)
        return x

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.detach().T
    
    def calc_kl_divergence(self, x):
        # Calculate KL Divergence of latent document distributions
        z_loc, z_sigma = self.encoder(x)
        size = z_loc.shape[1]
        kl_loss = torch.sum(
            (-torch.log(z_sigma) + (torch.square(z_sigma) + torch.square(z_loc))/2 - 1/2), (0, 1))

        return kl_loss
    
    def reconstruct_doc(self, x):
        x=x.to(self.device).squeeze()
        
        with torch.no_grad():
            z_loc, _ = self.encoder(x)
            theta = F.softmax(z_loc, dim=-1)
            word_probs = self.decoder(theta)
                              
        return word_probs
        
        
# this custom elbo allows us to pass deterministic values from the guide 
# into the model
def custom_elbo(model, guide, *args, **kwargs):
    # run the guide and trace its execution
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)

    # get the 'x_loc' value from the guide trace
    x = guide_trace.nodes['x']['value']

    # run the model and replay it against the samples from the guide
    # notice where x_loc is passed into the model
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(x = x, *args, **kwargs)
    
    guide_term = guide_trace.log_prob_sum()
    model_term = model_trace.log_prob_sum()

    
    # construct the elbo loss function
    return -1*(model_term - guide_term)