# WEIGHTED TRANSFORMER CODEMASTER
# CODE BY CATALINA JARAMILLO

import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import gutenberg
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


from players.codemaster import codemaster
import random
import scipy
import re

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

from transformers import XLNetTokenizer, XLNetModel
from scipy.spatial.distance import cosine
from more_itertools import powerset, locate
import bisect
from torch.nn import functional as F
import time
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_distances


class ai_codemaster(codemaster):


    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        #write any initializing code here

        # DEFINE THRESHOLD VALUE
        self.dist_threshold = 0.3

        self.all_guesses = []
        self.red_words = []
        self.blue_words = []
        self.civil_words = []
        self.assassin_word = [] 
        self.start_state_red_words = []
        self.start_state_blue_words = []
        self.start_state_civil_words = []
        self.start_state_assasin_word = []
        self.red_combs = []
        #self.red_comb_sim_red = []
        self.red_comb_sim_blue = []
        self.red_comb_sim_civil =[]
        self.red_comb_sim_assasin = []
        self.best_combs = []
        self.device = "cuda:0"

        # 1. GET EMBEDDING FOR RED WORDS USING BERT base uncased
        #torch.set_grad_enabled(False)
        with torch.no_grad():
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            self.model = XLNetModel.from_pretrained('xlnet-large-cased', output_hidden_states = False, from_tf=True)
            self.model.eval()
        #self.tokenizerbert = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model (weights)
         #AutoModelForCausalLM.from_pretrained('gpt2')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        #print("BERT model loaded")
        
        # If you have a GPU, put everything on cuda
        #self.tokenizer = self.tokenizer.to('cuda')
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.model.to(self.device)
        

        #get stop words and what-not
        nltk.download('popular',quiet=True)
        nltk.download('words',quiet=True)
        self.corp_words = set(nltk.corpus.words.words())
        self.nnfit = 0
  
        return

    def receive_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def give_clue(self, game_condition):

        # 1. GET THE RED WORDS
        count = 0
        #self.prev_state_red_words = self.red_words
        #self.prev_state_blue_words = self.blue_words
        #self.prev_state_civilian_words = self.civil_words
        #self.prev_state_assasin_word = self.assassin_word
        
        self.red_words = []
        self.blue_words = []
        self.civil_words = []
        self.assassin_word = []        
        
#         # Creates Red-Labeled Word arrays, and everything else arrays
#         for i in range(25):
#             if self.words[i][0] == '*':
#                 continue
#             elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
#                 bad_words.append(self.words[i].lower())
#             else:
#                 self.red_words.append(self.words[i].lower())

#        #print("RED:\t", self.red_words)
###############
        # Creates Red-Labeled Word arrays, and Blue-labeled, and Civilian-labeled, and Assassin-labeled arrays
        t1 = time.perf_counter()
        for i in range(25):
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin":
                self.assassin_word.append(self.words[i].lower())
                if len(self.all_guesses)==0: 
                    self.start_state_assasin_word.append(self.words[i].lower())
               
            elif self.maps[i] == "Blue":
                self.blue_words.append(self.words[i].lower())
                if len(self.all_guesses)==0: 
                    self.start_state_blue_words.append(self.words[i].lower())
                    
            elif self.maps[i] == "Civilian":
                self.civil_words.append(self.words[i].lower())
                if len(self.all_guesses)==0: 
                    self.start_state_civil_words.append(self.words[i].lower())
                    
            else:
                self.red_words.append(self.words[i].lower())
                if len(self.all_guesses)==0: 
                    self.start_state_red_words.append(self.words[i].lower())
        t2 = time.perf_counter()
        print("Red, blue assasin labelled arrays creation only once"+ str(t2-t1))
        
        
        t2 = time.perf_counter()
        current_red = [indx for indx,word in enumerate(self.start_state_red_words) if word in self.red_words]
        changes_red = []
        if len(self.start_state_red_words)!=len(self.red_words):
            changes_red = [indx for indx,word in enumerate(self.start_state_red_words) if word not in self.red_words]
            #changes_red = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_red_words, self.red_words)) if e1 != e2][0][0]
          
        current_blue = [indx for indx,word in enumerate(self.start_state_blue_words) if word in self.blue_words]
        changes_blue = []
        if len(self.start_state_blue_words)!=len(self.blue_words):
            changes_blue = [indx for indx,word in enumerate(self.start_state_blue_words) if word not in self.blue_words]  
            #changes_blue = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_blue_words, self.blue_words)) if e1 != e2][0][0]
           
        current_civil = [indx for indx,word in enumerate(self.start_state_civil_words) if word in self.civil_words]    
        changes_civil = []
        if len(self.start_state_civil_words)!=len(self.civil_words):
            changes_civil = [indx for indx,word in enumerate(self.start_state_civil_words) if word not in self.civil_words]
            #changes_civilian = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_civilian_words, self.civilian_words)) if e1 != e2][0][0]
            
            
        changes_assasin = []
        if len(self.start_state_assasin_word)!=len(self.assassin_word):
            changes_assasin = [indx for indx,word in enumerate(self.start_state_assasin_word) if word not in self.assassin_word]
            #changes_assasin = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_assasin_word, self.assassin_word)) if e1 != e2][0][0]
        t2 = time.perf_counter()
        print("changes in arrays identification in every guess"+ str(t2-t1))
        
            
        #print("RED:\t", self.red_words)


        ''' WRITE NEW CODE HERE '''
        # 1. Add \u0120 in front of every word to get a better embedding
        #spec_red_words = list(map(lambda w: "\u0120" + w, self.red_words))
###############
        #spec_blue_words = list(map(lambda w: "\u0120" + w, self.blue_words))
        #spec_civil_words = list(map(lambda w: "\u0120" + w, self.civil_words))
        #spec_assassin_word = list(map(lambda w: "\u0120" + w, self.assassin_word))
        #spec_red_words = self.red_words
        #spec_blue_words = self.blue_words
        #spec_civil_words = self.civil_words
        #spec_assassin_word = self.assassin_word


    
        
        #print(self.red_words)
        # 2. CREATE WORD EMBEDDINGS FOR THE RED WORDS
        #print(("self.red_words: "+ str(self.red_words)).encode("utf-8"))
        t1 = time.perf_counter()
        if len(self.all_guesses)==0:
            if len(self.red_words) > 0:
                #print("type self.red_words")
                #print(type(self.red_words))
                #print(self.red_words)
                self.red_emb = self.word_embedding(self.red_words)  #retrieves embedding for red_words from gpt2 layer 0 (static embedding)
                #last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
                #self.red_emb = output.last_hidden_state
                #print("self.red_emb.size")
                #print(self.red_emb.size())
                #print(self.red_emb.shape)
                #print(("self.blue_words: "+ str(self.blue_words)).encode("utf-8"))
            if len(self.blue_words) > 0:
                self.blue_emb = self.word_embedding(self.blue_words)
                #self.blue_emb = output.last_hidden_state
                #print(self.blue_emb.shape)
                #print(("self.assassin_word: "+ str(self.assassin_word)).encode("utf-8"))
            if len(self.assassin_word) > 0:
                self.assassin_emb = self.word_embedding(self.assassin_word)
                #self.assassin_emb = output.last_hidden_state
                #print(self.assassin_emb.shape)
                #print(("self.civil_words: "+ str(self.civil_words)).encode("utf-8"))
            if len(self.civil_words) > 0:
                self.civil_emb = self.word_embedding(self.civil_words)
                #self.civil_emb = output.last_hidden_state
                #print(self.civil_emb.shape)
            emb_matrix = self.model.word_embedding.weight
            #print(type(emb_matrix))
            self.vectors = emb_matrix.cpu().detach().numpy()
            #print("self.vectors.size")
            #print(self.vectors.size)
        t2 = time.perf_counter()
        print("embedding creation only once"+ str(t2-t1))        
       
        """
        else:
            if changes_red is not None:
                self.red_emb = np.delete(self.red_emb, changes_red, 0)
                
            if changes_blue is not None:
                self.blue_emb = np.delete(self.blue_emb, changes_blue, 0)
                
            if changes_civilian is not None:
                self.civil_emb = np.delete(self.civil_emb, changes_civil, 0)
                
            if changes_assasin is not None:
                self.assassin_emb = np.delete(self.assassin_emb, changes_assasin, 0)
        """    
            
        # 3. USE THE K NEAREST NEIGHBOR -LIKE ALGORITHM (FIND K NEIGHBORS BASED ON THRESHOLD)

        '''
            DISTANCE MATRIX FOR EACH VECTOR
    
                a.     b        c
            a | -   | -0.2  | 0.3  |
            b | 0.3 | -     | -0.4 |
            c | 0.1 |  0.6  |  -   |
            Choose the words that has the most neighbors within 
            the distance threshold
        '''
        t1 = time.perf_counter()
        if 1==1:#len(self.all_guesses)==0: 
            num_red_words = self.red_emb.shape[0]      
            num_blue_words = self.blue_emb.shape[0]  
            num_civil_words = self.civil_emb.shape[0]
            #self.red_combs = [list(l1)  for l1 in list(powerset(range(num_red_words))) if len(l1) > 0] # Here you already have the index and not the words
            
            
        t2 = time.perf_counter()    
        print("vacablary embeddings extraction only once"+ str(t2-t1)) 
        
        #creating red clusters using affinity clustering
        #sim_red_all = np.zeros((num_red_words, num_red_words))
        if 1==1:#len(self.all_guesses)==0: 
            #print(self.red_emb.detach().numpy().shape)
            
            #print(torch.flatten(self.red_emb, start_dim=1).detach().numpy().shape)
            #word_cosine = cosine_distances(torch.flatten(self.red_emb, start_dim=1).detach().numpy())
            ##for i in range(num_red_words):
                    #for j in range(num_red_words):
                        #print(self.red_emb[i].shape)
                        #print(self.red_emb[j].shape)
                        #sim_red_all[i][j] = self.cos_sim(self.red_emb[i][0],self.red_emb[j][0])
            
            old_red_combs = self.red_combs
            self.red_combs = []
            if len(current_red) == 1:
                
                self.red_combs.append(current_red) 
                #print("I'm Here2")
            #else:    
            #delete the combinations of already guessed red words from best combs and check the length of best_combs. If still >0 then no need to recreate clusters. first finish off existing best_combs. Do this tomorrow morning
            #print("I'm Here3")
            #first get all the old cobination indexes wherein any red word element hasbeen deleted
            if len(self.best_combs)!=0:
                #print("I'm Here4")
                to_be_deleted_comb_indexes = [indx for indx, comb in enumerate(old_red_combs) if len(list(set(comb) & set(changes_red)))>0]
                self.best_combs = [comb_element for comb_element in self.best_combs if comb_element[0] not in to_be_deleted_comb_indexes]     
                self.best_combs = [comb_element for comb_element in self.best_combs if comb_element[3] not in self.all_guesses]
                    
                self.best_combs.sort(key=self.takefifth)
                    
            if len(self.best_combs)==0:
                #print("I'm Here5")
                if len(current_red) != 1:
                    max_size_dim2 = max(self.red_emb[current_red].size()[1],self.blue_emb[current_blue].size()[1],self.civil_emb[current_civil].size()[1],self.assassin_emb.size()[1])
                    #print(max_size_dim2)
                    red_emb_padded = self.red_emb[current_red]
                    blue_emb_padded = self.blue_emb[current_blue]
                    civil_emb_padded = self.civil_emb[current_civil]
                    assassin_emb_padded = self.assassin_emb
                    if red_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - red_emb_padded.size()[1]
                        padding = ( 0,0,   # Fill 1 unit in the front and 2 units in the back
                                    0,pad_len,
                                    0,0
                                  )
                        red_emb_padded = F.pad(red_emb_padded, padding)
                    #print(red_emb_padded.size())

                    if blue_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - blue_emb_padded.size()[1]
                        padding = ( 0,0,   # Fill 1 unit in the front and 2 units in the back
                                    0,pad_len,
                                    0,0
                                  )
                        blue_emb_padded = F.pad(blue_emb_padded, padding)
                    #print(blue_emb_padded.size())
                    
                    if civil_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - civil_emb_padded.size()[1]
                        padding = ( 0,0,   # Fill 1 unit in the front and 2 units in the back
                                    0,pad_len,
                                    0,0
                                  )
                        civil_emb_padded = F.pad(civil_emb_padded, padding)
                    #print(civil_emb_padded.size())

                    if assassin_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - assassin_emb_padded.size()[1]
                        padding = ( 0,0,   # Fill 1 unit in the front and 2 units in the back
                                    0,pad_len,
                                    0,0
                                  )
                        assassin_emb_padded = F.pad(assassin_emb_padded, padding)
                    #print(assassin_emb_padded.size())
                    
                    all_current_board_word_embeddings =  torch.cat((red_emb_padded,civil_emb_padded,blue_emb_padded,assassin_emb_padded), 0)
                    total_words = all_current_board_word_embeddings.size()[0]
                    map_pos_list = ['r']* len(current_red)  + ['c'] * len(current_civil) + ['b'] * len(current_blue) + ['a'] * 1
                    print(map_pos_list)                 
                    
                    infinite = 0
                    #clustering_pref = 0.05
                    cluster_damping = 1
                    while(True):
                        if infinite == 0:
                            clustering = AffinityPropagation(affinity='euclidean',random_state=5).fit(torch.flatten(all_current_board_word_embeddings, start_dim=1).cpu().detach().numpy())
                            #clustering = AffinityPropagation(affinity='precomputed',random_state=5).fit(word_cosine)
                            #fluctuates between 0.01 and 0.015, find an ideal point between these 2
                            #0.011 2 clusters
                            #0.012 5 clusters
                            #print(sim_red_all)
                            #print(clustering.labels_)
                            #print(current_red)
                            #print(self.start_state_red_words)
                        else:
                            #clustering_pref = clustering_pref - 0.005
                            cluster_damping = cluster_damping - 0.1
                            print(cluster_damping)
                            clustering = AffinityPropagation(affinity='euclidean', damping = cluster_damping, random_state=5).fit(torch.flatten(all_current_board_word_embeddings, start_dim=1).cpu().detach().numpy())
                            #preference = clustering_pref, 
                        labels = list(clustering.labels_)
                        print(labels)
                        #labels.index for unq_items in labels
                        #print([list(labels, lambda x, unq_item: x == unq_item) for unq_item in list(np.unique(labels))])
                        curr_best_red_combs = []
                        for element in labels[:len(current_red)]:
                            if (element not in labels[len(current_red):]) and (element not in [elem[0] for elem in curr_best_red_combs]):
                                curr_best_red_combs.append((element,labels.count(element)))
                                print("condition1")
                        
                            #There are no such red groups which do not contain bad words, then relax the condition to allow groups of red words with civilian words
                        if len(curr_best_red_combs) == 0:
                            for element in labels[:len(current_red)]:
                                if (element not in labels[len(current_red)+len(current_civil):]) and (element not in [elem[0] for elem in curr_best_red_combs]):
                                    curr_best_red_combs.append((element,labels[:len(current_red)].count(element)))
                                    print("condition2")
                        #print(curr_best_red_combs)
                            #if still no such combinations relax th condition to include blue words, but never assasin word which will end the game
                        if len(curr_best_red_combs) == 0:
                            for element in labels[:len(current_red)]:
                                if (element not in labels[len(current_red)+len(current_civil)+len(current_blue):]) and (element not in [elem[0] for elem in curr_best_red_combs]):
                                    curr_best_red_combs.append((element,labels[:len(current_red)].count(element)))
                                    print("condition3")
                                                
                        print("curr_best_red_combs")            
                        print(curr_best_red_combs)
                        red_combs_curr = []
                        if len(curr_best_red_combs)>0:
                            break
                        else:
                            infinite = infinite + 1
                            print("breakpoint1")
                            if round(cluster_damping,1) == 0.5:
                                #assasin too close to red word embeddings, create seperate group for each red word
                                red_combs_curr = [[i] for i in range(len(current_red))]
                                print("breakpoint2")
                                break
                     
                    if len(red_combs_curr) == 0:
                        for unq_item in list(np.unique([elem[0] for elem in curr_best_red_combs])):#list(np.unique(labels)):
                             red_combs_curr.append(list(locate(labels[:len(current_red)], lambda x: x == unq_item)))
                        print(red_combs_curr)
                    
                    print("breakpoint3")
                    for red_comb in red_combs_curr:
                        self.red_combs.append([current_red[element] for element in red_comb])
                        
                    print("self.red_combs")    
                    print(self.red_combs)  
                    #if len(self.all_guesses) > 1:
                        #exit()
                        #for unq_items in labels
                        #print(clustering.cluster_centers_)
                        #print(clustering.cluster_centers_[0].shape)
                        #exit()                        
            
        
                #for each combination, do below steps
                #1. find their mean, and check the nearest clues to this mean
                #2. get embedding for this clue and check distance between he clue and closet bad word D1 
                #3. get the embedding for this clue and check the distance between the clue and farthest(worst) red word from the combination D2, D2 should be less than threshold
                #4  D1 should be greater then D2, then do (D1 - D2) and choose the combination for which (D1-D2) is highest. also choose a clue such that no of red words in combinaton are more
        
                if 1==1:#len(self.all_guesses)==0:
                    t1 = time.perf_counter()
                    old_best_combs = self.best_combs
                    self.best_combs = []
                    distance_order = []    
                    num_red_combs = len(self.red_combs)
                    #self.start_sim_red = np.zeros((num_red_combs, num_recomm, num_red_words_comb))
                    #self.start_sim_blue = np.zeros((num_red_combs, num_recomm, num_blue_words))
                    #self.red_comb_sim_red = []
                    self.red_comb_sim_blue = []
                    self.red_comb_sim_civil =[]
                    self.red_comb_sim_assasin = []
                    for comb_indx, comb in enumerate(self.red_combs):
                        found_old = 0
                        if comb in old_red_combs:
                            comb_index_old = old_red_combs.index(comb)
                    
                            recommended_clues = [b_comb[3] for b_comb in old_best_combs if b_comb[0]==comb_index_old]
                            #self.best_combs.append([comb_indx,len(comb),i,recommended_clues[i],closest_bad - worst_red,worst_red])
                     
                            if game_condition == "Hit_Red":
                            #last guessed word was red hence no chage in bad words distance
                                old_best_comb_elements = [[comb_indx,b_comb[1],b_comb[2],b_comb[3],b_comb[4],b_comb[5]] for b_comb in old_best_combs if b_comb[0]==comb_index_old]
                                self.best_combs.extend(old_best_comb_elements)
                                found_old = 1
                    
                        else:
                            #center of red word combination, to use this centre to find closest clue words for whole red combination
                
                            t3 = time.perf_counter()
                            center = torch.mean(self.red_emb[comb], dim=0).cpu().detach().numpy()
                            #print(center.shape)
                            #exit()
            
                            recommended_clues = self.getBestCleanWord(center,self.words)
                            t4 = time.perf_counter()
                            print("recommended clues for each combination only once"+ str(t4-t3)) 
                
                            #print("len(recommended_clues)")
                            #print(len(recommended_clues))
            
                        if found_old == 0:
                            t3 = time.perf_counter()
                            
                            recommended_clues_vec_rep = self.word_embedding(recommended_clues)
                            t4 = time.perf_counter()
                            print("recommended clues to embeddings "+ str(t4-t3)) 
                            #recommended_clues_vec_rep = output.last_hidden_state
            
                            #print("recommended_clues_vec_rep.shape")
                            #print(recommended_clues_vec_rep.shape)
                            num_recomm = recommended_clues_vec_rep.shape[0]
      
                            #print("comb")
                            #print(self.start_state_red_words[comb_element] for comb_element in comb)
                            #print("num_recomm")
                            #print(num_recomm)
                            #if num_recomm == 0:
                            #print("*********************************************No Recommendations***********************************************")
                            t3 = time.perf_counter()
                            num_red_words_comb = len(comb)
                            sim_red = np.zeros((num_recomm, num_red_words_comb))
            
                            #similarity of each recommendation with each red word in the combination(This will be needed to calculate worst red)
                            #we do not need to store it in self.red_comb_sim_red because once the red word is guessed, we will delete those combinations itself from the best_combs
                            for i in range(num_recomm):
                                for j in range(num_red_words_comb):
                                    comb_element_index = comb[j]
                                    #print("recommended_clues_vec_rep["+str(i)+"]")
                                    #print(recommended_clues_vec_rep[i].shape)
                                    #print("self.red_emb["+str(comb_element_index)+"]")
                                    #print(self.red_emb[comb_element_index].shape)
                                    sim_red[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.red_emb[comb_element_index])
                            t4 = time.perf_counter()
                            print("sim_red calculation "+ str(t4-t3))
                            
                            #print("sim_red.shape")
                            #print(sim_red.shape)
                            #print(recommended_clues_vec_rep.cpu().detach().numpy().shape)
                            #print(self.red_emb[comb].cpu().detach().numpy().shape)
                            #sim_red_pairwise = cosine_distances(recommended_clues_vec_rep.cpu().detach().numpy(),self.red_emb[comb].cpu().detach().numpy())
                            #print("sim_red_pairwise.shape")
                            #print(sim_red_pairwise.shape)
                            #exit()
                            #self.red_comb_sim_red.append(sim_red)
                
                            #similarity of each recommendation with assasin word
                            sim_assassin = np.zeros((num_recomm),)
                            for i in range(num_recomm):
                                 #print("recommended_clues_vec_rep["+str(i)+"]")
                                 #print(recommended_clues_vec_rep[i].shape)
                                 #print("self.assassin_emb[0]")
                                 #print(self.assassin_emb[0].shape)
                                 sim_assassin[i] = self.cos_sim(recommended_clues_vec_rep[i],self.assassin_emb[0])
            
                            sim_assassin = sim_assassin.reshape(sim_assassin.shape[0],1)
                
                            #self.red_comb_sim_assasin.append(sim_assassin)
       
                            #similarity of each recommendation with each blue word
                            sim_blue = np.zeros((num_recomm, num_blue_words))
                            for i in range(num_recomm):
                                for j in range(num_blue_words):
                                    #print("recommended_clues_vec_rep["+str(i)+"]")
                                    #print(recommended_clues_vec_rep[i].shape)
                                    #print("self.blue_emb["+str(j)+"]")
                                    #print(self.blue_emb[j].shape)
                                    sim_blue[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.blue_emb[j])
        
                            #self.red_comb_sim_blue.append(sim_blue)
            
                            #similarity of each recommendation with each civil word
                            sim_civil = np.zeros((num_recomm, num_civil_words))
                            for i in range(num_recomm):
                                for j in range(num_civil_words):
                                    #print("recommended_clues_vec_rep["+str(i)+"]")
                                    #print(recommended_clues_vec_rep[i].shape)
                                    #print("self.civil_emb["+str(j)+"]")
                                    #print(self.civil_emb[j].shape)
                                    sim_civil[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.civil_emb[j])
                
                            self.red_comb_sim_civil.append(sim_civil)
                
                            self.best_combs.extend([[comb_indx,len(comb),i,recommended_clues[i],min(sim_assassin[i],sim_blue[i].min(),sim_civil[i].min()) - sim_red[i].max(), sim_red[i].max()] for i in range(num_recomm)])
                                                                    
                                                                 
                            #for i in range(num_recomm):
                                 #closest_bad = min(sim_assassin[i],sim_blue[i].min(),sim_civil[i].min())
                                 #worst_red = sim_red[i].max()
                                 #print("I'm Here")
                                 #if (closest_bad - worst_red) > 0:
                                 #   print("I'm Here1")
                                 #self.best_combs.append([comb_indx,len(comb),i,recommended_clues[i],closest_bad - worst_red,worst_red])
                                 #bisect.insort(distance_order, closest_bad - worst_red)
                                 #index = distance_order.index(closest_bad - worst_red)
                                 #self.best_combs.insert(index, [comb_indx,len(comb),i,recommended_clues[i],closest_bad - worst_red,worst_red])
            
                self.best_combs.sort(key=self.takefifth)
                t2 = time.perf_counter()    
                print("clue finding, sim matrix and best combs creation only once"+ str(t2-t1))
                """
                    else:
                        t1 = time.perf_counter()
                        #1. Delete the red conbinations from best combinations which have been deleted(guessed)
                        indx_combs_deleted = list(set([indx for indx, comb in enumerate(self.red_combs) for element_indx in comb if element_indx in changes_red]))
                        self.best_combs = [element for element in self.best_combs if element[0] not in indx_combs_deleted]
                       
                        #2. for the remaining best combinations, recalculate the closest bad, as some bads have been deleted(guessed), and then resort the best_combs list based on closest_bad
            
                        for indx, comb_element in enumerate(self.best_combs):
                            comb_indx = comb_element[0]
                            sim_blue_comb = self.red_comb_sim_blue[comb_indx]
                            sim_blue_comb = np.delete(sim_blue_comb, changes_blue, axis=1)
                            sim_civil_comb = self.red_comb_sim_civil[comb_indx]
                            sim_civil_comb = np.delete(sim_civil_comb, changes_civil, axis=1)
                            sim_assasin_comb = self.red_comb_sim_assasin[comb_indx]
                            sim_assasin_comb = np.delete(sim_assasin_comb, changes_assasin, axis=1)
                            clue_index = comb_element[2]
                            closest_bad = min(sim_assasin_comb[clue_index],sim_blue_comb[clue_index].min(),sim_civil_comb[clue_index].min())
                            worst_red = comb_element[5]
                            comb_element[4] = closest_bad - worst_red
                            self.best_combs[indx] = comb_element
             
                        #print(self.best_combs) 
                        # sort list with key
                        self.best_combs.sort(key=self.takefifth)
                        #print([self.best_combs[comb_indx]]) 
            
                        t2 = time.perf_counter()    
                        print("best combs modification and sorting for every guess"+ str(t2-t1))
                """    
        
        t1 = time.perf_counter()
        #print("self.best_combs")
        #print(self.best_combs)
        #print("self.all_guesses")
        #print(self.all_guesses)
        found = 0
        
        for element in self.best_combs[::-1]:      
            #first taking the combination having higher number of clues
            if ((element[1]>1) and (element[4]>0) and (element[5]<=self.dist_threshold) and (element[3] not in self.all_guesses)):
                clue = element[3]
                clue_num = element[1] 
                found = 1 
                #print("condition1")
                #print(element)
                #print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                break
        
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[4]>0) and (element[5]<=self.dist_threshold) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    #print("condition2")
                    #print(element)
                    #print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                    break
       
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[4]>0) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    #print("condition3")
                    #print(element)
                    #print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                    break 
        
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[5]<=self.dist_threshold) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    #print("condition4")
                    #print(element)
                    #print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                    break
        t2 = time.perf_counter()    
        print("Final clue finding at every guess"+ str(t2-t1))
            
        #clue = best_combs[-1][2]
        #clue_num = best_combs[-1][1]
        self.all_guesses.append(clue)            
        return [clue,clue_num]


    # take fifth element for sort
    def takefifth(self, elem):
        return elem[4]

    """
    #create word vectors for each word
    def word_embedding(self, red_words):
        tokenized_texts = [self.tokenizer.tokenize("[CLS] " + word + " [SEP]") for word in red_words]
        #print(inputs)
        tokenized_texts = [[item[0],item[1]+item[2].replace("#",""),item[3]] if (len(item)>3 and item[2].startswith("#")) else item for item in tokenized_texts]
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
         
        segments_ids = [[1] * len(tokenized_text) for tokenized_text in tokenized_texts]
        
        max_len = max([len(l1) for l1 in indexed_tokens])
        indexed_tokens = [l1 + [0] * (max_len - len(l1)) for l1 in indexed_tokens]
        segments_ids = [l1 + [1] * (max_len - len(l1)) for l1 in segments_ids]
        
        #print(indexed_tokens)
        #print(segments_ids)
        #stacked_tensor_input_ids = torch.stack(input_ids_list)
        #stacked_tensor_attention_masks = torch.stack(attention_mask_list)
        #outputs = self.model(input_ids=stacked_tensor_input_ids, attention_mask=stacked_tensor_attention_masks)  
        input_ids_list = [torch.tensor(l1, dtype=torch.int) for l1 in indexed_tokens]
        attention_mask_list = [torch.tensor(l1, dtype=torch.int) for l1 in segments_ids]
        input_ids_tensor = torch.stack(input_ids_list)
        attention_mask_tensor = torch.stack(attention_mask_list)
        outputs = self.model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)  
        return outputs
        """
       #create word vectors for each word
    def word_embedding(self, red_words):
        with torch.no_grad():
            tokenized_texts = [self.tokenizer("[CLS] " + word + " [SEP]", is_split_into_words=True) for word in red_words]
            max_len = 0
            #for d1 in tokenized_texts:
             #   if len(d1['input_ids'])>max_len:
              #      max_len = len(d1['input_ids'])
            #print(max_len)        
            #batch_size = len(tokenized_texts)#400
            #output_batches = []
        
            #for num, batch_list in enumerate([tokenized_texts[i:i + batch_size] for i in range(0, len(tokenized_texts), batch_size)]):
            #    print(num)  
            outputs_l = []
            
            for d1 in tokenized_texts: #batch_list:
                indexed_token = d1['input_ids']
                segments_id = d1['attention_mask']
                if len(indexed_token)>max_len:
                    max_len = len(indexed_token)        
                tokens_tensor = torch.tensor(indexed_token).unsqueeze(0).to(self.device)
                segments_tensor = torch.tensor(segments_id).unsqueeze(0).to(self.device)
                output = self.model(tokens_tensor, segments_tensor)
                del tokens_tensor
                del segments_tensor
                outputs_l.append(output.last_hidden_state[0])
                del output
                torch.cuda.empty_cache()
            for num,tensor_element in enumerate(outputs_l):
                if tensor_element.shape[0]<max_len:
                    pad_len = max_len - tensor_element.shape[0]
                    padding = ( 0,0,   # Fill 1 unit in the front and 2 units in the back
                                0,pad_len
                              )
                    outputs_l[num] = F.pad(tensor_element, padding)
                    
            #output_batches.extend(outputs_l)    
            outputs = torch.stack(outputs_l)
        return outputs     

    # cosine similarity
    def cos_sim(self, input1, input2):
        #distance.cosine([1, 0, 0], [0, 1, 0]) 1.0
        #distance.cosine([100, 0, 0], [0, 1, 0]) 1.0
        #distance.cosine([1, 1, 0], [0, 1, 0]) 0.29289321881345254
        
        #cos = nn.CosineSimilarity(dim=0,eps=1e-6)
        #cos = 1 - cosine(input1.mean(axis=0).detach().numpy(), input2.mean(axis=0).detach().numpy())
        #print("input1")
        #print(input1)
        #print("input2")
        #print(input2)
        #print(input1.mean(axis=0).detach().numpy())
        #print(input2.mean(axis=0).detach().numpy())
        cos = cosine(input1.mean(axis=0).cpu().detach().numpy(), input2.mean(axis=0).cpu().detach().numpy())
        return cos #cos(input1, input2)

    #clean up the set of words
    def cleanWords(self, embed):
        
        #print("embed")
        #print(embed)
        #print(type(embed))
        
        recomm = [i.lower() for i in embed]
        
        
        
        #print("recomm")
        #print(recomm)
        #print(type(recomm))
        
        #recomm2 = ' '.join(recomm)
        recomm2 = [i.replace(" ", "") for i in recomm]
        recomm2 = " ".join(recomm2)

        #print("recomm2")
        #print(recomm2)
        #print(type(recomm2))
        #print(type(recomm2[0]))
        
        recomm3 = [w for w in nltk.wordpunct_tokenize(recomm2) if w.lower() in self.corp_words or not w.isalpha()]
        
        #print("recomm3")
        #print(recomm3)

        prepositions = open('ai4games/prepositions_etc.txt').read().splitlines() #create list with prepositions
        stop_words = nltk.corpus.stopwords.words('english')        #change set format
        stop_words.extend(prepositions)                    #add prepositions and similar to stopwords
        word_tokens = word_tokenize(' '.join(recomm3)) 
        recomm4 = [w for w in word_tokens if not w in stop_words]

        excl_ascii = lambda s: re.match('^[\x00-\x7F]+$', s) != None        #checks for ascii only
        is_uni_char = lambda s: (len(s) == 1) == True                        #check if a univode character
        recomm5 = [w for w in recomm4 if excl_ascii(w) and not is_uni_char(w) and not w.isdigit()]
        #recomm5 = recomm5.Where(s => s.All(Char.IsLetter)).ToList();
        recomm5 = [s for s in recomm5 if s.isalpha()]

        return recomm5

    def getBestCleanWord(self, center, board):
        tries = 1
        amt = 100
        maxTry = 20
        #print("1")
        if len(self.all_guesses)==0 and self.nnfit == 0:
            t5 = time.perf_counter() 
            self.knn = NearestNeighbors(n_neighbors=maxTry*amt)
            #print("2")
            self.knn.fit(self.vectors)
            self.nnfit = 1
            t6 = time.perf_counter() 
            print("NearestNeighbors fit"+ str(t6-t5))
        
        #print("3")
        #vecinos = knn.kneighbors(center.reshape(1,-1))
        #print(str(np.any(np.isnan(center.detach().numpy()))))
        #print(str(np.all(np.isfinite(center.detach().numpy()))))
        center = np.where(np.isfinite(center) == True, center, 0)
        #print("center shape")
        #print(center.shape)
        t7 = time.perf_counter()  
        vecinos = self.knn.kneighbors(center)
        t8 = time.perf_counter()    
        print("knn.kneighbors(center) "+ str(t8-t7))
        #print(distances.shape)
        #print(distances)
        #print(indices.shape)
        #print(indices)
        #exit()
        #print("vecinos.shape")
        #print(vecinos[1].shape)
        #sys.stdout.write("4")
        #print(type(vecinos[0]))
        #print(type(vecinos[1]))
        #print(vecinos[1][0])
        #print(type(vecinos[1][0]))
        #print(vecinos[1].shape)
        #print(vecinos[1][0].shape)

        low_board = list(map(lambda w: w.lower(), board))
        
        t9 = time.perf_counter() 
        vecinos_arr_t_flatten = vecinos[1].transpose().flatten()
        t10 = time.perf_counter()
        print("vecinos[1].transpose().flatten() "+ str(t10-t9))
        #print("vecinos_arr_t.shape")
        #print(vecinos_arr_t_flatten.shape)
        #print(self.tokenizer.batch_decode(vecinos_arr_t_flatten.tolist(), skip_special_tokens = True, clean_up_tokenization_spaces = True))
        #exit()
        #while (tries < 5):
         
          
        t11 = time.perf_counter() 
        # 6. WORD CLEANUP AND PARSING
        recomm = self.tokenizer.batch_decode(vecinos_arr_t_flatten.tolist(), skip_special_tokens = True, clean_up_tokenization_spaces = True)
        recomm = list(set(recomm))
        #[]
        #numrec = (tries-1)*1000
        #for i in range((tries-1)*amt,(maxTry)*amt):#range((tries-1)*amt,(tries)*amt):
            #print(int(vecinos[1][0][i]))
            #print(vecinos_arr_t[i].shape)
            #print(type(vecinos_arr_t[i]))
            #print(vecinos_arr_t[i])
            #print(type(vecinos_arr_t[i][0]))
            #print(self.tokenizer.decode(int(vecinos_arr_t[i][0]), skip_special_tokens = True, clean_up_tokenization_spaces = True))
            #print(self.tokenizer.decode(vecinos_arr_t[i].tolist(), skip_special_tokens = True, clean_up_tokenization_spaces = True))
            #print(' '.join([self.tokenizer.decode(int(vecinos_arr_t[i][k]), skip_special_tokens = True, clean_up_tokenization_spaces = True) for k in range(vecinos_arr_t[i].shape[0])]))
            #print(vecinos_arr_t[i])
            #recomm.append(self.tokenizer.decode(vecinos_arr_t[i], skip_special_tokens = True, clean_up_tokenization_spaces = True))       
            #print("vecinos_arr_t[i]")
            #print(vecinos_arr_t[i])
            #print("convert_ids_to_tokens")
            #print(self.tokenizer.convert_ids_to_tokens(int(vecinos[1][0][i]), skip_special_tokens = 'True'))
            #print("tokens to words")
            #print("vecinos_arr_t[i].tolist()")
            #print(vecinos_arr_t[i].tolist())
            #print(vecinos_arr_t[i].tolist())
            #recomm.append( self.tokenizer.batch_decode(vecinos_arr_t[i].tolist(), skip_special_tokens = True, clean_up_tokenization_spaces = True)[1])
            #print(type(recomm))
            #print(recomm)
            #recomm = list(set(recomm))
            #for j in range(len(vecinos_arr_t[i])):
                #word = self.tokenizer.decode(int(vecinos_arr_t[i][j]), skip_special_tokens = True, clean_up_tokenization_spaces = True)
                #if word not in recomm:
                #print(self.tokenizer.decode(int(vecinos_arr_t[i][j]), skip_special_tokens = True, clean_up_tokenization_spaces = True))
                   #recomm.append(word)  
                    
        t12 = time.perf_counter()
        print("recomm  tokenizer.decode"+ str(t12-t11))
        #print("recomm")
        #print(recomm)
        
        #exit()
        
        # exclude words on board from the recommended list
        t13 = time.perf_counter()
        recomm1 = [i for i in recomm if i not in low_board]
        
        # and ("\`" not in r.lower()) and ("=" not in r.lower()) and ("~" not in r.lower()) and ("'" not in r.lower()) and ("/" not in r.lower()) and ("\\" not in r.lower()) and ("+" not in r.lower()) and ("-" not in r.lower()) and ("|" not in r.lower()) and ("\." not in r.lower())
        
        recomm2 = []
        for r in recomm1:
            if (sum([1 if ((r.lower() in w.lower()) or (w.lower() in r.lower())) else 0 for w in low_board])==0):
                recomm2.append(r)
                #print(r)
        t14 = time.perf_counter()
        print("recomm1 & recomm2 exclude words on board from the recommended list"+ str(t14-t13))
        #print("recomm2")
        #print(recomm2)    
        
        t15 = time.perf_counter()
        clean_words = self.cleanWords(recomm2)
        t16 = time.perf_counter()
        print("cleanWords "+ str(t16-t15))
        #print("clean_words")
        #print(clean_words)
        #exit()
        #print("len(clean_words)")
        #print(len(clean_words))
            
        return clean_words #self.weightWords(clean_words,low_board, center)
            
            
        '''
        #7. Get the first word not in the board
        for w in clean_words:
            if w not in low_board:
                return w

        #otherwise try again
        tries+=1
        '''

        #return "??"        #i got nothing out of 5000 words


    
###### new code
    def weightWords(self, rec_words, low_board, center):
        # exclude words on board from the recommended list
        recomm6 = [i for i in rec_words if i not in low_board]
        #print("rec_words")
        #print(rec_words)
        #print("recomm6")
        #print(recomm6)
        recomm7 = []
        for r in recomm6:
            if (sum([1 if ((r.lower() in w.lower()) or (w.lower() in r.lower())) else 0 for w in low_board])==0) and ("`" not in r.lower()):
                recomm7.append(r)
            #for w in low_board:
                #if not (r.lower() in w.lower() or w.lower() in r.lower()):
                        #recomm7.append(r)
        #print("recomm7")
        #print(recomm7)
        # find word embedding for recommended words
        #recomm6_vec = self.word_embedding(list(map(lambda w: "\u0120" + w, recomm7)))
        #print("type recomm7")
        #print(type(recomm7))
        #print(recomm7)
        recomm6_vec = self.word_embedding(recomm7)
        #recomm6_vec = output.last_hidden_state

        #similarity between each recommendation and assassin
        num_recomm = recomm6_vec.shape[0]   #number of words in the embedding matrix

        sim_assassin = np.zeros((num_recomm),)
        for i in range(num_recomm):
            print("recomm6_vec["+str(i)+"]")
            print(recomm6_vec[i].shape)
            print("self.assassin_emb[0]")
            print(self.assassin_emb[0].shape)
            sim_assassin[i] = self.cos_sim(recomm6_vec[i],self.assassin_emb[0])

        sim_assassin = sim_assassin.reshape(sim_assassin.shape[0],1)
        #print("sim_assassin ")
        #print(sim_assassin)   

        # create similarity matrix for recomm and blue words
        num_recomm = recomm6_vec.shape[0]   #number of words in the clean recommendation list
        num_blue = self.blue_emb.shape[0]   #number of words in 'subset' used for centroid

        sim_blue = np.zeros((num_recomm, num_blue))
        for i in range(num_recomm):
            for j in range(num_blue):
                print("recomm6_vec["+str(i)+"]")
                print(recomm6_vec[i].shape)
                print("self.blue_emb["+str(j)+"]")
                print(self.blue_emb[j].shape)
                sim_blue[i][j] = self.cos_sim(recomm6_vec[i],self.blue_emb[j])
        #print("sim_blue ")
        #print(sim_blue)        

        # create similarity matrix for recomm and civilians words
        num_recomm = recomm6_vec.shape[0]   #number of words in the clean recommendation list
        num_civil = self.civil_emb.shape[0]   #number of words in 'subset' used for centroid


        sim_civil = np.zeros((num_recomm, num_civil))
        for i in range(num_recomm):
            for j in range(num_civil):
                print("recomm6_vec["+str(i)+"]")
                print(recomm6_vec[i].shape)
                print("self.civil_emb["+str(j)+"]")
                print(self.civil_emb[j].shape)
                sim_civil[i][j] = self.cos_sim(recomm6_vec[i],self.civil_emb[j])
        #print("sim_civil ")
        #print(sim_civil)        

        #dist similarity recommendations and center
        num_recomm = recomm6_vec.shape[0]   #number of words in the embedding matrix

        #print(type(recomm6_vec[0]))
        #print(type(self.civil_emb[0]))
        #print(type(center))
        center_tensor = torch.from_numpy(center)
        sim_center = np.zeros((num_recomm))
        
        for i in range(num_recomm):
            print("recomm6_vec["+str(i)+"]")
            print(recomm6_vec[i].shape)
            print("center_tensor")
            print(center_tensor.shape)
            sim_center[i] = self.cos_sim(recomm6_vec[i],center_tensor)

        sim_center = sim_center.reshape(sim_center.shape[0],1)
        #print("sim_center ")
        #print(sim_center)

        #find similarity ratio for recommended words between center and (assasin, blue, civil)
        ratio_assasin = sim_center / sim_assassin
        ratio_blue = sim_center / sim_blue
        ratio_civil = sim_center / sim_civil
        
        #print("ratio_assasin")
        #print(ratio_assasin)
        
        #print("ratio_blue")
        #print(ratio_blue)
        
        #print("ratio_civil")
        #print(ratio_civil)

        #define weights for each kind of word
        assassin_weight = 5
        blue_weight = 3
        civil_weight = 1

        #find the total ratio for each recommended word
        recomm_ratio = (ratio_assasin * assassin_weight) + \
            (np.min(ratio_blue, axis=1).reshape(ratio_blue.shape[0],1) * blue_weight) + \
            (np.min(ratio_civil, axis=1).reshape(ratio_civil.shape[0],1) * civil_weight)
             
        #print("recomm_ratio")
        #print(recomm_ratio)
        
        rec_rat = {}
        for r in range(len(recomm7)):
            rec_rat[recomm7[r]] = recomm_ratio[r][0] 
            
        #print(rec_rat)
        
        #[sorted(recomm_ratio, reverse=True).index(x) for x in recomm_ratio] #from low to high
        #rec_rat2 = sorted(rec_rat.items(), key=lambda item: float(item[1]), reverse=True)
        rec_rat2 = {k: v for k, v in sorted(rec_rat.items(), key=lambda item: item[1])}  #, reverse=True
        
        #print("rec_rat2")
        #print(rec_rat2)
        
        #print("list(rec_rat2.keys())[0]")
        #print(list(rec_rat2.keys())[0])
        
        return list(rec_rat2.keys())[0]
##### not working!!!!!!!!!
