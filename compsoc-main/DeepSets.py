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
from torchmetrics import RetrievalMRR
"""from torchinfo import summary
from torchvision import datasets
import torchvision.transforms as transforms"""
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, TensorDataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import stats
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

def compute_score(profile:Profile) ->List[float]:
    ranking = compute_ranking(profile)
    scores = [0] *len(ranking)
    for i in range(len(ranking)):
        index = ranking[i]
        scores[index]= (len(ranking) - i)/len(ranking)
    return scores

#Function to produce training data
def generate_training_data(num_samples: int) ->List[Tuple[Profile, List[int]]]:
  training_data =[]
  for _ in range(num_samples):
    num_cand = random.randint(4, 6)
    num_voters = random.randint(10, 10000)
    votes = generate_random_votes(num_voters, num_cand)
    profile = Profile(votes)
    scores = compute_score(profile)
    scores = scores + [0]*(20-num_cand)
    duplicated_votes = []
    duplicated_scores = [0]*len(scores)

    for count, ballot in profile.pairs:
        ballot = ballot + tuple(range(num_cand, 20))
        duplicated_votes.extend([ballot]* count)

    #rand[0]
    rand = np.random.permutation(20)
    print(rand)
    #shuffle the candidates
    for j in range(len(duplicated_votes)):
        ballot = duplicated_votes[j]
        ballot = list(ballot)
        for i in range(len(ballot)):
            ballot[i] = rand[ballot[i]]
        ballot = tuple(ballot)
        duplicated_votes[j] = ballot


    for i in range(len(scores)):
        duplicated_scores[i]=  scores[rand[i]] 

    print("scores:",scores)
    print(duplicated_scores)
    

    
    #duplicated_votes get the num_of_cand and subract if from 20
    
    num_dum_voter = 10000 - num_voters
    dum_votes = [[20]*20]*num_dum_voter

    padded_profiles = duplicated_votes + dum_votes  
    padded_profiles_tensor = torch.tensor(padded_profiles, dtype=torch.float32)


    ranking_tensor = torch.tensor(duplicated_scores, dtype=torch.float32)

    training_data.append((padded_profiles_tensor, ranking_tensor))



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



"""model = DeepSetOriginal(dim_input=10, num_outputs=5, dim_output=1)
print(model)
print(summary(model))"""
print(253)
training_data = generate_training_data(10)
for x, y in training_data:
    print("Scores:", y)
    break

profiles_list = [item[0] for item in training_data]
rankings_list = [item[1] for item in training_data]

# Stack the lists into tensors
profiles_tensor = torch.stack(profiles_list)
rankings_tensor = torch.stack(rankings_list)

#add profiles and numerical rankings into a TensorDataset
dataset = TensorDataset(profiles_tensor, rankings_tensor)

#the sizes for the splits
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

#Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print(270)

#create DataLoaders for each dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(276)
"""for profiles_batch, rankings_batch in train_loader:
    print("Profiles batch:", profiles_batch)
    print("Numerical Rankings batch:", rankings_batch)
    break 
"""
print(282)
# Instantiate the model 
model = DeepSetOriginal(20, 20, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20




def train_model(train_loader, val_loader, model, optimizer, epochs):
    metric_info = {}
    metric_info['val_spearman_list']= []
    metric_info['train_spearman_list']= []
    metric_info['val_loss_list'] = []
    metric_info['train_loss_list'] = []

    for epoch in range(epochs):
    
        model.train()
        total_train_loss = 0.0
        total_train_spearman = 0.0
        
        for profiles_batch, rankings_batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(profiles_batch)
        
            loss = model.listnet_loss(outputs, rankings_batch)
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            outputs = outputs.squeeze(-1)
            
            batch_spearman = 0.0
            
                
            for i in range(outputs.size(0)):
                preds = outputs[i].detach().cpu().numpy()
                targets = rankings_batch[i].detach().cpu().numpy()
                spearman_corr, _ = stats.spearmanr(preds, targets)
                batch_spearman += spearman_corr
                
            batch_spearman /= outputs.size(0)
            total_train_spearman += batch_spearman
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_spearman = total_train_spearman / len(train_loader)
        metric_info['train_loss_list'].append(avg_train_loss)
        metric_info['train_spearman_list'].append(avg_train_spearman)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Spearman: {avg_train_spearman:.4f}')
        
        model.eval()
        total_val_loss = 0.0
        total_val_spearman = 0.0
        with torch.no_grad():
            for profiles_batch, rankings_batch in val_loader:
                outputs = model(profiles_batch)
                loss = model.listnet_loss(outputs, rankings_batch)
                total_val_loss += loss.item()
                outputs = outputs.squeeze(-1)
                
                batch_spearman = 0.0
                for i in range(outputs.size(0)):
                    preds = outputs[i].detach().cpu().numpy()
                    targets = rankings_batch[i].detach().cpu().numpy()
                    spearman_corr, _ = stats.spearmanr(preds, targets)
                    batch_spearman += spearman_corr
                
                batch_spearman /= outputs.size(0)
                total_val_spearman += batch_spearman
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_spearman = total_val_spearman / len(val_loader)
        metric_info['val_loss_list'].append(avg_val_loss)
        metric_info['val_spearman_list'].append(avg_val_spearman)
        
        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}, Val Spearman: {avg_val_spearman:.4f}')
    
    return metric_info





# train the model
metric_info = train_model(train_loader, val_loader, model, optimizer, num_epochs)


def drawgraph(metric_info):
    
    x = np.arange(len(metric_info['train_spearman_list']))
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, metric_info['train_spearman_list'], "ro-", label="train_spearman")
    plt.plot(x, metric_info['val_spearman_list'], "b--", label="val_spearman")

    plt.title("Training and Validation Spearman")
    plt.xlabel("Epochs")
    plt.ylabel("Spearman")
    plt.legend(loc="lower right")


    x = np.arange(len(metric_info['train_loss_list']))
    plt.subplot(1, 2, 2)
    plt.plot(x, metric_info['train_loss_list'], "ro-", label="train_loss")
    plt.plot(x, metric_info['val_loss_list'], "b--", label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")

    plt.show()

drawgraph(metric_info)


def model_test(test_loader, model):
    model.eval()
    total_test_loss = 0.0
    total_test_spearman = 0.0
    with torch.no_grad():
        for profiles_batch, rankings_batch in test_loader:
            outputs = model(profiles_batch)
            loss = model.listnet_loss(outputs, rankings_batch)
            total_test_loss += loss.item()
            outputs = outputs.squeeze(-1)
            
            batch_spearman = 0.0
            for i in range(outputs.size(0)):
                preds = outputs[i].detach().cpu().numpy()
                targets = rankings_batch[i].detach().cpu().numpy()
                spearman_corr, _ =stats.spearmanr(preds, targets)
                batch_spearman += spearman_corr
                
            batch_spearman /= outputs.size(0)
            total_test_spearman += batch_spearman
    
    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_spearman = total_test_spearman / len(test_loader)
    
    print(f'Test Loss: {avg_test_loss:.4f}, Test Spearman: {avg_test_spearman:.4f}')
    
    return avg_test_loss, avg_test_spearman

# test the model
avg_test_loss, avg_test_spearman = model_test(test_loader, model)


def predict_one(profile,model):
    model.eval()
    with torch.no_grad():
        if not isinstance(profile, torch.Tensor):
            profile = torch.tensor(profile)
        if profile.dim()==1:
            profile = profile.unsqueeze(0)
        output = model(profile)
        output = output.squeeze(0)

    return output


training_data = generate_training_data(1)
sample_profiles = [item[0] for item in training_data]
sample_rankings =[item[1] for item in training_data]

print("Best Ranking: ", sample_rankings)

print("Profile:",sample_profiles)
for profile in sample_profiles:
    prediction = predict_one(profile, model)
    print("Prediction: ", prediction)

