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
    def __init__(self, vocab_size, num_topics, hidden, dropout, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.device = device
        self.encoder = Encoder(vocab_size, self.num_topics, hidden, dropout).to(self.device)
        self.decoder = Decoder(vocab_size, self.num_topics, dropout).to(self.device)

    def model(self, docs_a, docs_b, c, observed):
        pyro.module("decoder", self.decoder)
            
        pyro.deterministic("c", c)
        pyro.deterministic("observed", observed)
        
        logtheta_loc = docs_a.new_zeros((docs_a.shape[0], self.num_topics))
        logtheta_scale = docs_a.new_ones((docs_a.shape[0], self.num_topics))
        with pyro.plate("documents_a", docs_a.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_a = pyro.sample(
                "logtheta_a", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta_a = pyro.deterministic("theta_a",(F.softmax(logtheta_a, -1)))
                                         
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
            theta_b = pyro.deterministic("theta_b",(F.softmax(logtheta_b, -1)))

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


    def guide(self, docs_a, docs_b, c, observed):
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
        
        
class LossClass():
    def __init__(self, num_topics, alpha, device):
        self.num_topics = num_topics
        self.alpha = alpha
        self.device = device
        
    def get_kld(self, theta_a, theta_b, indices_to_drop = None):
        """
        Symmetrized and normalized KL Divergence
        indices_to_drop: optional argument to drop specified topics
        """
        
        mask = torch.ones(self.num_topics, device=self.device)
        if indices_to_drop is not None:
            mask[indices_to_drop] = 0
        
        kld = -((theta_a*torch.log(theta_b+1e-20)-theta_a*torch.log(theta_a+1e-20))+
                (theta_b*torch.log(theta_a+1e-20)-theta_b*torch.log(theta_b+1e-20)))
        
        kld = (kld*mask).sum(axis=-1)/(torch.log(torch.tensor([self.num_topics], device=self.device))*2)
        return kld
        
    def custom_loss(self, model, guide, *args, **kwargs):
        # run the guide and trace its execution
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)

        # run the model and replay it against the samples from the guide
        # notice where x_loc is passed into the model
        model_trace = poutine.trace(
            poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

        theta_a = model_trace.nodes['theta_a']['value']
        theta_b = model_trace.nodes['theta_b']['value']
        c = model_trace.nodes['c']['value'].type(torch.bool)
        observed = model_trace.nodes['observed']['value']

        batch_size = theta_a.shape[0]

        similarity_loss = self.get_kld(theta_a, theta_b)
        
        similarity_loss = torch.where(c, similarity_loss, -1*similarity_loss)
        similarity_loss = torch.where(observed, similarity_loss, torch.zeros(1, device=self.device))                                    

        model_term = model_trace.log_prob_sum()
        guide_term = guide_trace.log_prob_sum()
        similarity_loss = similarity_loss.sum(axis=-1)
        
        # construct the elbo loss function
        return -1*(model_term - guide_term) + self.alpha*similarity_loss