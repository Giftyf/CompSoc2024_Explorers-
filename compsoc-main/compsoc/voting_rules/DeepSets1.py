from compsoc.voter_model import generate_random_votes
from compsoc.profile import Profile
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from compsoc.voting_rules.DeepSets import generate_training_data



class DeepSetOriginal(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=512):
        super(DeepSetOriginal, self).__init__()
        self.dim_input = dim_input
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, num_outputs * dim_output))
    def forward(self, X, **kwargs):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X.squeeze(1) 
    
    #utility function to calculate the score for the given ranking
    def utility_for_ranking(self, ranking, pairs):
        overall_utility = 0.0
        num_candidates = 20
        utility_increments = [(num_candidates - i) / (num_candidates * 1.0) for i in range(num_candidates)]
        
        for pair in pairs:
            total_utility = 0.0
            for i in range(len(pair[1])):  
                total_utility += utility_increments[ranking.tolist().index(pair[1][i])]
            overall_utility += total_utility * pair[0]

        return overall_utility

    #using Pairwise ranking hinge loss
    def ranking_loss(self, scores, targets, pairs):
       batch_size = scores.size(0)
       loss = 0.0

       for i in range(batch_size):
            _, predicted_ranking = torch.sort(scores[i], descending=True)
            predicted_utility = self.utility_for_ranking(predicted_ranking, pairs)
            best_utility = self.utility_for_ranking(targets[i], pairs)
            loss += best_utility - predicted_utility
       return loss / batch_size
    
        # Pairwise Ranking Loss
    def pairwise_ranking_loss(self, predicted_scores, true_rankings):
        batch_size = predicted_scores.size(0)
        num_candidates = predicted_scores.size(1)
        loss = 0.0
        
        for i in range(batch_size):
            for j in range(num_candidates):
                for k in range(j + 1, num_candidates):
                    true_diff = true_rankings[i][j] - true_rankings[i][k]
                    pred_diff = predicted_scores[i][j] - predicted_scores[i][k]
                    
                    if true_diff > 0:
                        loss += torch.max(torch.tensor(0.0), 1.0 - pred_diff)
                    elif true_diff < 0:
                        loss += torch.max(torch.tensor(0.0), 1.0 + pred_diff)
                    # No loss if true_diff == 0 (equal ranking)

        return loss / (batch_size * num_candidates * (num_candidates - 1) / 2)
    
    #ListNet Loss
    def listnet_loss(self, predicted_scores, true_scores):
        predicted_scores = predicted_scores.squeeze(-1)
        predicted_prob = torch.softmax(predicted_scores, dim=1)
        true_prob = torch.softmax(true_scores, dim=1)
        loss = -torch.sum(true_prob * torch.log(predicted_prob + 1e-10), dim=1).mean()
        print("Loss:", loss)
        return loss   



def deepSets(profile: Profile, candidate: int) -> int:

    model = DeepSetOriginal(20,20, 1) 
    model.load_state_dict(torch.load('model_state_dict.pth'))

    def predict_one(profile,model):
        model.eval()
        with torch.no_grad():
            if not isinstance(pairs, torch.Tensor):
                pairs = torch.tensor(pairs)
            if pairs.dim()==1:
                pairs = pairs.unsqueeze(0)
            output = model(pairs)
            output = output.squeeze(0)

        return output
    
    prediction = predict_one(profile.pairs, model)
    prediction = prediction.cpu().detach().numpy().tolist()
    
    num_cand = len(profile.candidates)
    final_ranking_list=[]
    for i in range(num_cand):
        index_max = prediction.index(max(prediction))
        final_ranking_list[i] =index_max
        prediction[index_max] = -1.1
    
    cand_position = final_ranking_list.index(candidate)

    return ((len(profile.candidates)) - cand_position)





