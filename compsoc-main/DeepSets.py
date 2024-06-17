import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from compsoc.profile import Profile
from collections import Counter
import random
from typing import List, Optional, Tuple
#from compsoc.voting_rules.greedyN import greedyN_rule
from compsoc.voter_model import generate_random_votes
from torchinfo import summary

#Generate trianing Data


#Function compute the optimal ranking using the greedayN_rule
def compute_ranking(profile: Profile) -> List[int]:
  candidate_scores =[]
  for candidate in profile.candidates:
    score = greedyN_rule(profile, candidate)
    candidate.scores.append((candidate, score))

  sorted_candidates = sorted(candidate_scores, key=lambda x:x[1], reverse = True )

  ranking = []
  for candidate, score in sorted_candidates:
    ranking.append(candidate)
  return ranking


#Function to produce training data
def generate_training_data(num_samples: int) ->List[Tuple[Profile, List[int]]]:
  training_data =[]
  for _ in range(num_samples):
    num_cand = random.randint(4, 5)
    num_voters = random.randint(10, 10000)
    votes = generate_random_votes(num_voters, num_cand)
    profile = Profile(votes)
    ranking = compute_ranking(profile)
    training_data.append((profile, ranking))
  return training_data

#fucntion to calculate the utility for the given ranking
def utility_for_ranking(ranking: torch.Tensor, profile) -> float:
    overall_utility = 0.0
    num_candidates = len(profile.candidates)
    utility_increments = [(num_candidates - i) / (num_candidates * 1.0) for i in range(num_candidates)]
    
    for pair in profile.pairs:
        total_utility = 0.0
        for i in range(len(pair[1])):  
            total_utility += utility_increments[ranking.tolist().index(pair[1][i])]
        overall_utility += total_utility * pair[0]

    return overall_utility


#define the model
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
        return X
    
    #utility function to calculate the score for the given ranking
    def utility_for_ranking(self, ranking, profile):
        overall_utility = 0.0
        num_candidates = len(profile.candidates)
        utility_increments = [(num_candidates - i) / (num_candidates * 1.0) for i in range(num_candidates)]
        
        for pair in profile.pairs:
            total_utility = 0.0
            for i in range(len(pair[1])):  
                total_utility += utility_increments[ranking.tolist().index(pair[1][i])]
            overall_utility += total_utility * pair[0]

        return overall_utility

    #using Pairwise ranking hinge loss
    def ranking_loss(self, scores, targets, profile):
       batch_size = scores.size(0)
       loss = 0.0

       for i in range(batch_size):
            _, predicted_ranking = torch.sort(scores[i], descending=True)
            predicted_utility = self.utility_for_ranking(predicted_ranking, profile)
            best_utility = self.utility_for_ranking(targets[i], profile)
            loss += best_utility - predicted_utility
       return loss / batch_size


model = DeepSetOriginal(dim_input=10, num_outputs=5, dim_output=1)
print(model)
print(summary(model))