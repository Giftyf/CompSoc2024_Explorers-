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




def greedyN_rule(profile: Profile, candidate: int) -> int:
    topN = 1


    """
    Computes the scores for candidates using a simple greedy, brute force rule that considers each ballot's top N candidates.


    :param profile: The voting profile.
    :type profile: VotingProfile
    :param candidate: The base candidate for scoring.
    :type candidate: int
    :return: The greedyN score for the candidate.
    :rtype: int
    """
    def scoreOne(profile: Profile, candidate: int) -> int:
        scores = 0
        for pair in profile.pairs:
            if candidate in pair[1]:
                for i in range(topN):
                    if pair[1].index(candidate) <= i:
                        #len(profile.candidates) / 2:
                        scores += pair[0]
        return scores
    
    def utilityForRanking (ranking: list[int], topN_parm: int) -> int:
        # Calculate utility of a given ranking for a given topN_parm value
        # iterate over the voters
        overall_utility = 0.0
        for pair in profile.pairs:
            # Utility of the ballot multipled by its counts
            num_candidates = len(profile.candidates)
            utility_increments = [(num_candidates - i) / (num_candidates * 1.0) for i in range(num_candidates)]
            total_utility = 0.0
            for i in range(min(topN_parm, len(pair[1]))):
                total_utility += utility_increments[ranking.index(pair[1][i])]
            overall_utility += total_utility * pair[0]

        return overall_utility

    def compute_permutations(candidates):
        # base cases
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [candidates]
        
        permutations = []
        for candidate in range(len(candidates)):
            fir_cand = candidates[candidate]
            rest_cand = candidates[:candidate] + candidates[candidate + 1:]
            for rest in compute_permutations(rest_cand):
                permutations.append([fir_cand] + rest)
        return permutations

    # Compute optimal when topN is each possible value to find live options
    # Find whole ranking for topN=1, topN=2, etc    
    liveOptions = []
    if(len(profile.candidates)<=5):
        liveOptions = compute_permutations(list(profile.candidates))

    else:
        for j in range(len(profile.candidates) // 2):
            topN = j + 1
            rankings = profile.ranking(scoreOne)
            liveOptions.append(rankings)
        temp_ranking = []
        #remove the scores from the ranking
        for pairs in liveOptions:
            temp =[]
            for j in pairs:
                temp.append(j[0])
            temp_ranking.append(temp)
        liveOptions = temp_ranking

    
    # Figure out which of the live options is the best expected value

    expectedUtilities = [0]* (len(liveOptions))
    # iterate over the live options
    for j in range(len(liveOptions)):
        # iterate over possible values of topN
        for m in range(len(profile.candidates) // 2):
            expectedUtilities[j] += utilityForRanking(liveOptions[j], m+1)


    # Find highest utility
    bestLiveOption = expectedUtilities.index(max(expectedUtilities))

    # Return the score for this candidate based on that ordering

    temp_ranking = liveOptions[bestLiveOption].index(candidate)
    
    return ((len(profile.candidates)) - temp_ranking)


#Generate trianing Data


#Function compute the optimal ranking using the greedayN_rule
def compute_ranking(profile: Profile) -> List[int]:
  candidate_scores =[]
  for candidate in profile.candidates:
    score = greedyN_rule(profile, candidate)
    candidate_scores.append((candidate, score))

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
    ranking = ranking + list(range(num_cand, 21))
    duplicated_votes = []
    for count, ballot in profile.pairs:
        ballot = ballot + tuple(range(num_cand, 20))
        duplicated_votes.extend([ballot]* count)
    
    #duplicated_votes get the num_of_cand and subract if from 20
    
    num_dum_voter = 10000 - num_voters
    dum_votes = [[20]*20]*num_dum_voter

    padded_profiles = duplicated_votes + dum_votes  
    training_data.append((padded_profiles, ranking))



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


"""model = DeepSetOriginal(dim_input=10, num_outputs=5, dim_output=1)
print(model)
print(summary(model))"""

training_data = generate_training_data(1)
print(training_data)
