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

from transformers import AlbertTokenizer, AlbertModel
from scipy.spatial.distance import cosine
from more_itertools import powerset, locate
import bisect
from torch.nn import functional as F
import time
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from nltk.corpus import words
from random import sample

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
        with torch.no_grad():
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.model = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states = False,)
            self.model.eval()

        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.model.to(self.device)
        

        #get stop words and what-not
        nltk.download('popular',quiet=True)
        nltk.download('words',quiet=True)
        self.corp_words = set(nltk.corpus.words.words())
        self.nnfit = 0
        self.pca = PCA(n_components=128, svd_solver='auto')
        self.train_pca_model()
        return

    def receive_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def give_clue(self, game_condition):

        # 1. GET THE RED WORDS
        count = 0

        self.red_words = []
        self.blue_words = []
        self.civil_words = []
        self.assassin_word = []        
        
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
        print("Red, blue assasin labelled arrays creation"+ str(t2-t1))
        
        
        t2 = time.perf_counter()
        current_red = [indx for indx,word in enumerate(self.start_state_red_words) if word in self.red_words]
        changes_red = []
        if len(self.start_state_red_words)!=len(self.red_words):
            changes_red = [indx for indx,word in enumerate(self.start_state_red_words) if word not in self.red_words]
      
        current_blue = [indx for indx,word in enumerate(self.start_state_blue_words) if word in self.blue_words]
        changes_blue = []
        if len(self.start_state_blue_words)!=len(self.blue_words):
            changes_blue = [indx for indx,word in enumerate(self.start_state_blue_words) if word not in self.blue_words]  
      
        current_civil = [indx for indx,word in enumerate(self.start_state_civil_words) if word in self.civil_words]    
        changes_civil = []
        if len(self.start_state_civil_words)!=len(self.civil_words):
            changes_civil = [indx for indx,word in enumerate(self.start_state_civil_words) if word not in self.civil_words]            
            
        changes_assasin = []
        if len(self.start_state_assasin_word)!=len(self.assassin_word):
            changes_assasin = [indx for indx,word in enumerate(self.start_state_assasin_word) if word not in self.assassin_word]
            
        t2 = time.perf_counter()
        print("changes in arrays identification in every guess"+ str(t2-t1))

        # 2. CREATE WORD EMBEDDINGS FOR THE RED WORDS
        t1 = time.perf_counter()
        if len(self.all_guesses)==0:
            if len(self.red_words) > 0:
                self.red_emb = self.word_embedding(self.red_words)  
                self.red_emb_reduced = self.reduce_dimentions_pca(self.red_emb)
            if len(self.blue_words) > 0:
                self.blue_emb = self.word_embedding(self.blue_words)
            if len(self.assassin_word) > 0:
                self.assassin_emb = self.word_embedding(self.assassin_word)
            if len(self.civil_words) > 0:
                self.civil_emb = self.word_embedding(self.civil_words)
            emb_matrix = self.model.embeddings.word_embeddings.weight
            self.vectors = emb_matrix.cpu().detach().numpy()
        t2 = time.perf_counter()
        print("embedding creation only once"+ str(t2-t1))        
       
        if 1==1:
            num_red_words = self.red_emb.shape[0]      
            num_blue_words = self.blue_emb.shape[0]  
            num_civil_words = self.civil_emb.shape[0]
            
        
        if 1==1:
            old_red_combs = self.red_combs
            self.red_combs = []
            if len(current_red) == 1:
                self.red_combs.append(current_red)     
            #delete the combinations of already guessed red words from best combs and check the length of best_combs. If still >0 then no need to recreate clusters. first finish off existing best_combs.
            #first get all the old cobination indexes wherein any red word element has been deleted
            if len(self.best_combs)!=0:
                to_be_deleted_comb_indexes = [indx for indx, comb in enumerate(old_red_combs) if len(list(set(comb) & set(changes_red)))>0]
                self.best_combs = [comb_element for comb_element in self.best_combs if comb_element[0] not in to_be_deleted_comb_indexes]   
                self.best_combs = [comb_element for comb_element in self.best_combs if comb_element[3] not in self.all_guesses]
                self.best_combs.sort(key=self.takefifth)
                    
            if len(self.best_combs)==0:
                if len(current_red) != 1:
                    max_size_dim2 = max(self.red_emb[current_red].size()[1],self.blue_emb[current_blue].size()[1],self.civil_emb[current_civil].size()[1],self.assassin_emb.size()[1])
                    red_emb_padded = self.red_emb[current_red]
                    blue_emb_padded = self.blue_emb[current_blue]
                    civil_emb_padded = self.civil_emb[current_civil]
                    assassin_emb_padded = self.assassin_emb
                    if red_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - red_emb_padded.size()[1]
                        padding = ( 0,0,
                                    0,pad_len,
                                    0,0
                                  )
                        red_emb_padded = F.pad(red_emb_padded, padding)
                    if blue_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - blue_emb_padded.size()[1]
                        padding = ( 0,0,
                                    0,pad_len,
                                    0,0
                                  )
                        blue_emb_padded = F.pad(blue_emb_padded, padding)
                    if civil_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - civil_emb_padded.size()[1]
                        padding = ( 0,0,
                                    0,pad_len,
                                    0,0
                                  )
                        civil_emb_padded = F.pad(civil_emb_padded, padding)
                    if assassin_emb_padded.size()[1] < max_size_dim2:
                        pad_len = max_size_dim2 - assassin_emb_padded.size()[1]
                        padding = ( 0,0,
                                    0,pad_len,
                                    0,0
                                  )
                        assassin_emb_padded = F.pad(assassin_emb_padded, padding)
                    
                    all_current_board_word_embeddings =  torch.cat((red_emb_padded,civil_emb_padded,blue_emb_padded,assassin_emb_padded), 0)
                    total_words = all_current_board_word_embeddings.size()[0]
                    map_pos_list = ['r']* len(current_red)  + ['c'] * len(current_civil) + ['b'] * len(current_blue) + ['a'] * 1
                    
                    infinite = 0
                    cluster_damping = 1
                    while(True):
                        if infinite == 0:
                            clustering = AffinityPropagation(affinity='euclidean',random_state=5).fit(torch.flatten(all_current_board_word_embeddings, start_dim=1).cpu().detach().numpy())
                        else:
                            cluster_damping = cluster_damping - 0.1
                            clustering = AffinityPropagation(affinity='euclidean', damping = cluster_damping,  random_state=5).fit(torch.flatten(all_current_board_word_embeddings, start_dim=1).cpu().detach().numpy())
                        labels = list(clustering.labels_)
                    
                        curr_best_red_combs = []
                        for element in labels[:len(current_red)]:
                            if (element not in labels[len(current_red):]) and (element not in [elem[0] for elem in curr_best_red_combs]):
                                curr_best_red_combs.append((element,labels.count(element)))
                         
                            #There are no such red groups which do not contain bad words, then relax the condition to allow groups of red words with civilian words
                        if len(curr_best_red_combs) == 0:
                            for element in labels[:len(current_red)]:
                                if (element not in labels[len(current_red)+len(current_civil):]) and (element not in [elem[0] for elem in curr_best_red_combs]):
                                    curr_best_red_combs.append((element,labels[:len(current_red)].count(element)))

                        if len(curr_best_red_combs) == 0:
                            for element in labels[:len(current_red)]:
                                if (element not in labels[len(current_red)+len(current_civil)+len(current_blue):]) and (element not in [elem[0] for elem in curr_best_red_combs]):
                                    curr_best_red_combs.append((element,labels[:len(current_red)].count(element)))
                        
                        red_combs_curr = []
                        if len(curr_best_red_combs)>0:
                            break
                        else:
                            infinite = infinite + 1
                            if round(cluster_damping,1) == 0.5:
                                #assasin too close to red word embeddings, create seperate group for each red word
                                red_combs_curr = [[i] for i in range(len(current_red))]
                                break
                    
                    if len(red_combs_curr) == 0:
                        for unq_item in list(np.unique([elem[0] for elem in curr_best_red_combs])):#list(np.unique(labels)):
                             red_combs_curr.append(list(locate(labels[:len(current_red)], lambda x: x == unq_item)))
                        
                        
                    for red_comb in red_combs_curr:
                        self.red_combs.append([current_red[element] for element in red_comb])
                                             
            
        
                #for each combination, do below steps
                #1. find their mean, and check the nearest clues to this mean
                #2. get embedding for this clue and check distance between he clue and closet bad word D1 
                #3. get the embedding for this clue and check the distance between the clue and farthest(worst) red word from the combination D2, D2 should be less than threshold
                #4  D1 should be greater then D2, then do (D1 - D2) and choose the combination for which (D1-D2) is highest. also choose a clue such that no of red words in combinaton are more
        
                if 1==1:
                    t1 = time.perf_counter()
                    old_best_combs = self.best_combs
                    self.best_combs = []
                    distance_order = []    
                    num_red_combs = len(self.red_combs)
                    self.red_comb_sim_blue = []
                    self.red_comb_sim_civil =[]
                    self.red_comb_sim_assasin = []
                    for comb_indx, comb in enumerate(self.red_combs):
                        found_old = 0
                        if comb in old_red_combs:
                            comb_index_old = old_red_combs.index(comb)
                            recommended_clues = [b_comb[3] for b_comb in old_best_combs if b_comb[0]==comb_index_old]
                     
                            if game_condition == "Hit_Red":
                            #last guessed word was red hence no chage in bad words distance
                                old_best_comb_elements = [[comb_indx,b_comb[1],b_comb[2],b_comb[3],b_comb[4],b_comb[5]] for b_comb in old_best_combs if b_comb[0]==comb_index_old]
                                self.best_combs.extend(old_best_comb_elements)
                                found_old = 1
                    
                        else:
                            #center of red word combination, to use this centre to find closest clue words for whole red combination
                            t3 = time.perf_counter()
                            center = torch.mean(self.red_emb_reduced[comb], dim=0).cpu().detach().numpy()
            
                            recommended_clues = self.getBestCleanWord(center,self.words)
                            t4 = time.perf_counter()
                            print("recommended clues for each combination"+ str(t4-t3)) 
                
                        if found_old == 0:
                            t3 = time.perf_counter()
                            
                            recommended_clues_vec_rep = self.word_embedding(recommended_clues)
                            t4 = time.perf_counter()
                            print("recommended clues to embeddings "+ str(t4-t3)) 
                            num_recomm = recommended_clues_vec_rep.shape[0]
      
                            t3 = time.perf_counter()
                            num_red_words_comb = len(comb)
                            sim_red = np.zeros((num_recomm, num_red_words_comb))
            
                            #similarity of each recommendation with each red word in the combination(This will be needed to calculate worst red)
                            #we do not need to store it in self.red_comb_sim_red because once the red word is guessed, we will delete those combinations itself from the best_combs
                            for i in range(num_recomm):
                                for j in range(num_red_words_comb):
                                    comb_element_index = comb[j]
                                    sim_red[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.red_emb[comb_element_index])
                            t4 = time.perf_counter()
                            print("sim_red calculation "+ str(t4-t3))
                            
                            #similarity of each recommendation with assasin word
                            sim_assassin = np.zeros((num_recomm),)
                            for i in range(num_recomm):
                                 sim_assassin[i] = self.cos_sim(recommended_clues_vec_rep[i],self.assassin_emb[0])
            
                            sim_assassin = sim_assassin.reshape(sim_assassin.shape[0],1)
       
                            #similarity of each recommendation with each blue word
                            sim_blue = np.zeros((num_recomm, num_blue_words))
                            for i in range(num_recomm):
                                for j in range(num_blue_words):
                                    sim_blue[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.blue_emb[j])
            
                            #similarity of each recommendation with each civil word
                            sim_civil = np.zeros((num_recomm, num_civil_words))
                            for i in range(num_recomm):
                                for j in range(num_civil_words):
                                    sim_civil[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.civil_emb[j])
                
                            self.red_comb_sim_civil.append(sim_civil)
                
                            self.best_combs.extend([[comb_indx,len(comb),i,recommended_clues[i],min(sim_assassin[i],sim_blue[i].min(),sim_civil[i].min()) - sim_red[i].max(), sim_red[i].max()] for i in range(num_recomm)])
            
                self.best_combs.sort(key=self.takefifth)
                t2 = time.perf_counter()    
                print("clue finding, sim matrix and best combs creation"+ str(t2-t1))  
        
        t1 = time.perf_counter()
        found = 0
        
        for element in self.best_combs[::-1]:      
            #first taking the combination having higher number of clues
            if ((element[1]>1) and (element[4]>0) and (element[5]<=self.dist_threshold) and (element[3] not in self.all_guesses)):
                clue = element[3]
                clue_num = element[1] 
                found = 1 
                break
        
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[4]>0) and (element[5]<=self.dist_threshold) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    break
       
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[4]>0) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    break 
        
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[5]<=self.dist_threshold) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    break
        t2 = time.perf_counter()    
        print("Final clue finding at every guess"+ str(t2-t1))
            
        self.all_guesses.append(clue)            
        return [clue,clue_num]


    # take fifth element for sort
    def takefifth(self, elem):
        return elem[4]

    #create word vectors for each word
    def word_embedding(self, red_words):
        with torch.no_grad():
            tokenized_texts = [self.tokenizer("[CLS] " + word + " [SEP]") for word in red_words] #, is_split_into_words=True    
            max_len = 0 
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
                    padding = ( 0,0,
                                0,pad_len
                              )
                    outputs_l[num] = F.pad(tensor_element, padding)   
            outputs = torch.stack(outputs_l)
        return outputs    

    def train_pca_model(self):
        sample_words = sample(words.words(), 160)
        output = self.word_embedding(sample_words)
        output = output.view(-1, *output.shape[2:]) 
        self.pca.fit_transform(output.cpu().detach().numpy())
        
    def reduce_dimentions_pca(self, tensor_3d):
        reduced_tesors_list = []
        for num, tensor_element in enumerate(tensor_3d):
            reduced_tesors_list.append(torch.tensor(self.pca.transform(tensor_element.cpu().detach().numpy())))
        output = torch.stack(reduced_tesors_list)
        return output
        
    # cosine similarity
    def cos_sim(self, input1, input2):
        cos = cosine(input1.mean(axis=0).cpu().detach().numpy(), input2.mean(axis=0).cpu().detach().numpy())
        return cos

    #clean up the set of words
    def cleanWords(self, embed):
        recomm = [i.lower() for i in embed]
        recomm2 = [i.replace(" ", "") for i in recomm]
        recomm2 = " ".join(recomm2)
        recomm3 = [w for w in nltk.wordpunct_tokenize(recomm2) if w.lower() in self.corp_words or not w.isalpha()]
        prepositions = open('ai4games/prepositions_etc.txt').read().splitlines() #create list with prepositions
        stop_words = nltk.corpus.stopwords.words('english')        #change set format
        stop_words.extend(prepositions)                    #add prepositions and similar to stopwords
        word_tokens = word_tokenize(' '.join(recomm3)) 
        recomm4 = [w for w in word_tokens if not w in stop_words]
        excl_ascii = lambda s: re.match('^[\x00-\x7F]+$', s) != None        #checks for ascii only
        is_uni_char = lambda s: (len(s) == 1) == True                        #check if a univode character
        recomm5 = [w for w in recomm4 if excl_ascii(w) and not is_uni_char(w) and not w.isdigit()]
        recomm5 = [s for s in recomm5 if s.isalpha()]
        return recomm5

    def getBestCleanWord(self, center, board):
        tries = 1
        amt = 150
        maxTry = 1
        if len(self.all_guesses)==0 and self.nnfit == 0:
            t5 = time.perf_counter() 
            self.knn = NearestNeighbors(n_neighbors=maxTry*amt)
            self.knn.fit(self.vectors)
            self.nnfit = 1
            t6 = time.perf_counter() 
            print("NearestNeighbors fit"+ str(t6-t5))
        
        center = np.where(np.isfinite(center) == True, center, 0)
        t7 = time.perf_counter()  
        vecinos = self.knn.kneighbors(center)
        t8 = time.perf_counter()    
        print("knn.kneighbors(center) "+ str(t8-t7))
        
        low_board = list(map(lambda w: w.lower(), board))
        
        t9 = time.perf_counter() 
        vecinos_arr_t_flatten = vecinos[1].transpose().flatten()
        t10 = time.perf_counter()
        print("vecinos[1].transpose().flatten() "+ str(t10-t9))
          
        t11 = time.perf_counter() 
        # 6. WORD CLEANUP AND PARSING
        recomm = self.tokenizer.batch_decode(vecinos_arr_t_flatten.tolist(), skip_special_tokens = True, clean_up_tokenization_spaces = True)
        recomm = list(set(recomm))            
        t12 = time.perf_counter()
        print("recomm  tokenizer.decode"+ str(t12-t11))

        # exclude words on board from the recommended list
        t13 = time.perf_counter()
        recomm1 = [i for i in recomm if i not in low_board]
        
        recomm2 = []
        for r in recomm1:
            if (sum([1 if ((r.lower() in w.lower()) or (w.lower() in r.lower())) else 0 for w in low_board])==0):
                recomm2.append(r)
        t14 = time.perf_counter()
        print("recomm1 & recomm2 exclude words on board from the recommended list"+ str(t14-t13))  
        
        t15 = time.perf_counter()
        clean_words = self.cleanWords(recomm2)
        t16 = time.perf_counter()
        print("cleanWords "+ str(t16-t15))
            
        return clean_words
           