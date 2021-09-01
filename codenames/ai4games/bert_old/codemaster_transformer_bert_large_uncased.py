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

#AutoModelWithLMHead
from transformers import BertTokenizer, BertModel #AutoModel, AutoModelForCausalLM,  AutoTokenizer, BertTokenizer, BertModel
from scipy.spatial.distance import cosine
from more_itertools import powerset
import bisect


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
        

        # 1. GET EMBEDDING FOR RED WORDS USING BERT base uncased
        #torch.set_grad_enabled(False)
        torch.no_grad()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')  #GPT2Tokenizer.from_pretrained('gpt2')
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-large-uncased',
                                                output_hidden_states = True,) #AutoModelForCausalLM.from_pretrained('gpt2')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        #print("BERT model loaded")
        

        #get stop words and what-not
        nltk.download('popular',quiet=True)
        nltk.download('words',quiet=True)
        self.corp_words = set(nltk.corpus.words.words())
        
  
        return

    def receive_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def give_clue(self):

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
        
        
        changes_red = []
        if len(self.start_state_red_words)!=len(self.red_words):
            changes_red = [indx for indx,word in enumerate(self.start_state_red_words) if word not in self.red_words]
            #changes_red = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_red_words, self.red_words)) if e1 != e2][0][0]
          
        changes_blue = []
        if len(self.start_state_blue_words)!=len(self.blue_words):
            changes_blue = [indx for indx,word in enumerate(self.start_state_blue_words) if word not in self.blue_words]
            #changes_blue = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_blue_words, self.blue_words)) if e1 != e2][0][0]
           
            
        changes_civil = []
        if len(self.start_state_civil_words)!=len(self.civil_words):
            changes_civil = [indx for indx,word in enumerate(self.start_state_civil_words) if word not in self.civil_words]
            #changes_civilian = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_civilian_words, self.civilian_words)) if e1 != e2][0][0]
            
            
        changes_assasin = []
        if len(self.start_state_assasin_word)!=len(self.assassin_word):
            changes_assasin = [indx for indx,word in enumerate(self.start_state_assasin_word) if word not in self.assassin_word]
            #changes_assasin = [(i, e1, e2) for i, (e1, e2) in enumerate(zip(self.prev_state_assasin_word, self.assassin_word)) if e1 != e2][0][0]
        
        
            
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
        if len(self.all_guesses)==0:
            if len(self.red_words) > 0:
                #print("type self.red_words")
                #print(type(self.red_words))
                #print(self.red_words)
                output = self.word_embedding(self.red_words)  #retrieves embedding for red_words from gpt2 layer 0 (static embedding)
                #last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
                self.red_emb = output.last_hidden_state
                #print(self.red_emb.shape)
                #print(("self.blue_words: "+ str(self.blue_words)).encode("utf-8"))
            if len(self.blue_words) > 0:
                output = self.word_embedding(self.blue_words)
                self.blue_emb = output.last_hidden_state
                #print(self.blue_emb.shape)
                #print(("self.assassin_word: "+ str(self.assassin_word)).encode("utf-8"))
            if len(self.assassin_word) > 0:
                output = self.word_embedding(self.assassin_word)
                self.assassin_emb = output.last_hidden_state
                #print(self.assassin_emb.shape)
                #print(("self.civil_words: "+ str(self.civil_words)).encode("utf-8"))
            if len(self.civil_words) > 0:
                output = self.word_embedding(self.civil_words)
                self.civil_emb = output.last_hidden_state
                #print(self.civil_emb.shape)
       
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
        
        if len(self.all_guesses)==0: 
            num_red_words = self.red_emb.shape[0]      
            num_blue_words = self.blue_emb.shape[0]  
            num_civil_words = self.civil_emb.shape[0]
            self.red_combs = [list(l1)  for l1 in list(powerset(range(num_red_words))) if len(l1) > 0] # Here you already have the index and not the words
            emb_matrix = self.model.embeddings.word_embeddings.weight
            self.vectors = emb_matrix.detach().numpy()
        
        
        #for each combination, do below steps
        #1. find their mean, and check the nearest clues to this mean
        #2. get embedding for this clue and check distance between he clue and closet bad word D1 
        #3. get the embedding for this clue and check the distance between the clue and farthest(worst) red word from the combination D2, D2 should be less than threshold
        #4  D1 should be greater then D2, then do (D1 - D2) and choose the combination for which (D1-D2) is highest. also choose a clue such that no of red words in combinaton are more
        
        if len(self.all_guesses)==0:
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
                #center of red word combination, to use this centre to find closest clue words for whole red combination
                center = torch.mean(self.red_emb[comb], dim=0).detach().numpy()
            
                recommended_clues = self.getBestCleanWord(center,self.words)
            
                output = self.word_embedding(recommended_clues)
                recommended_clues_vec_rep = output.last_hidden_state
            
                num_recomm = recommended_clues_vec_rep.shape[0]
     
                if num_recomm == 0:
                   print("*********************************************No Recommendations***********************************************")
            
                num_red_words_comb = len(comb)
                sim_red = np.zeros((num_recomm, num_red_words_comb))
            
                #similarity of each recommendation with each red word in the combination(This will be needed to calculate worst red)
                #we do not need to store it in self.red_comb_sim_red because once the red word is guessed, we will delete those combinations itself from the best_combs
                for i in range(num_recomm):
                    for j in range(num_red_words_comb):
                        comb_element_index = comb[j]
                        sim_red[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.red_emb[comb_element_index])
                 
                #self.red_comb_sim_red.append(sim_red)
                
                #similarity of each recommendation with assasin word
                sim_assassin = np.zeros((num_recomm),)
                for i in range(num_recomm):
                    sim_assassin[i] = self.cos_sim(recommended_clues_vec_rep[i],self.assassin_emb[0])

                sim_assassin = sim_assassin.reshape(sim_assassin.shape[0],1)
                
                self.red_comb_sim_assasin.append(sim_assassin)
       
                #similarity of each recommendation with each blue word
                sim_blue = np.zeros((num_recomm, num_blue_words))
                for i in range(num_recomm):
                    for j in range(num_blue_words):
                        sim_blue[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.blue_emb[j])
        
                self.red_comb_sim_blue.append(sim_blue)
            
                #similarity of each recommendation with each civil word
                sim_civil = np.zeros((num_recomm, num_civil_words))
                for i in range(num_recomm):
                    for j in range(num_civil_words):
                        sim_civil[i][j] = self.cos_sim(recommended_clues_vec_rep[i],self.civil_emb[j])
                
                self.red_comb_sim_civil.append(sim_civil)
                
                for i in range(num_recomm):
                    closest_bad = min(sim_assassin[i],sim_blue[i].min(),sim_civil[i].min())
                    worst_red = sim_red[i].max()
                    bisect.insort(distance_order, closest_bad - worst_red)
                    index = distance_order.index(closest_bad - worst_red)
                    self.best_combs.insert(index, [comb_indx,len(comb),i,recommended_clues[i],closest_bad - worst_red,worst_red])
        else:
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
            print([self.best_combs[comb_indx]]) 
              
        found = 0
       
        for element in self.best_combs[::-1]:      
            #first taking the combination having higher number of clues
            if ((element[1]>1) and (element[4]>0) and (element[5]>self.dist_threshold) and (element[3] not in self.all_guesses)):
                clue = element[3]
                clue_num = element[1] 
                found = 1 
                print("condition1")
                print(element)
                print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                break
        
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[4]>0) and (element[5]>self.dist_threshold) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    print("condition2")
                    print(element)
                    print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                    break
       
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[4]>0) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    print("condition3")
                    print(element)
                    print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                    break 
        
        if found == 0:
            for element in self.best_combs[::-1]:
                if ((element[5]>self.dist_threshold) and (element[3] not in self.all_guesses)):
                    clue = element[3]
                    clue_num = element[1] 
                    found = 1 
                    print("condition4")
                    print(element)
                    print([self.start_state_red_words[red_element] for red_element in self.red_combs[element[0]]])
                    break
            
        #clue = best_combs[-1][2]
        #clue_num = best_combs[-1][1]
        self.all_guesses.append(clue)            
        return [clue,clue_num]


    # take fifth element for sort
    def takefifth(self, elem):
        return elem[4]

    #create word vectors for each word
    def word_embedding(self, red_words):
        #word_emb = []
        #for word in red_words:
        #text_index = self.tokenizer.encode(red_words,add_prefix_space=False)
        #tokenized_texts = [self.tokenizer.encode(word, max_length=2, pad_to_max_length=True) for word in red_words]
        tokenized_texts = [self.tokenizer.tokenize("[CLS] " + word + " [SEP]") for word in red_words]
        tokenized_texts = [[item[0],item[1]+item[2].replace("#",""),item[3]] if (len(item)>3 and item[2].startswith("#")) else item for item in tokenized_texts]
        
        #tokenized_texts = [x for x in tokenized_texts if x.count("`")==0]
        #print("tokenized_texts")
        #print(tokenized_texts)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
        #print("indexed_tokens")
        #print(indexed_tokens)
        #print(type(indexed_tokens))
        # Mark each of the 22 tokens as belonging to sentence "1".
        segments_ids = [[1] * len(tokenized_text) for tokenized_text in tokenized_texts]
        
        # Convert inputs to PyTorch tensors
        tokens_tensors = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)
        outputs = self.model(tokens_tensors, segments_tensors)
        #word_emb.append(output)
        #word_emb = self.model.transformer.wte.weight[text_index,:]
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
        cos = cosine(input1.mean(axis=0).detach().numpy(), input2.mean(axis=0).detach().numpy())
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
        

        return recomm5

    def getBestCleanWord(self, center, board):
        tries = 1
        amt = 100
        maxTry = 20
        #print("1")
        knn = NearestNeighbors(n_neighbors=(maxTry*amt))
        #print("2")
        knn.fit(self.vectors)
        #print("3")
        #vecinos = knn.kneighbors(center.reshape(1,-1))
        #print(str(np.any(np.isnan(center.detach().numpy()))))
        #print(str(np.all(np.isfinite(center.detach().numpy()))))
        center = np.where(np.isfinite(center) == True, center, 0)
        vecinos = knn.kneighbors(center)
        #sys.stdout.write("4")

        low_board = list(map(lambda w: w.lower(), board))

        #while (tries < 5):

        # 6. WORD CLEANUP AND PARSING
        recomm = []
        #numrec = (tries-1)*1000
        for i in range((tries-1)*amt,(tries)*amt):
            recomm.append(self.tokenizer.decode((int(vecinos[1][0][i])), skip_special_tokens = True, clean_up_tokenization_spaces = True))         
        #print("recomm")
        #print(recomm)
        
        # exclude words on board from the recommended list
        recomm1 = [i for i in recomm if i not in low_board]
        
        
        recomm2 = []
        for r in recomm1:
            if (sum([1 if ((r.lower() in w.lower()) or (w.lower() in r.lower())) else 0 for w in low_board])==0) and ("`" not in r.lower()):
                recomm2.append(r)
            
        clean_words = self.cleanWords(recomm2)

        #print("clean_words")
        #print(clean_words)
            
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
        output = self.word_embedding(recomm7)
        recomm6_vec = output.last_hidden_state

        #similarity between each recommendation and assassin
        num_recomm = recomm6_vec.shape[0]   #number of words in the embedding matrix

        sim_assassin = np.zeros((num_recomm),)
        for i in range(num_recomm):
            #print("recomm6_vec["+str(i)+"]")
            #print(recomm6_vec[i])
            #print("self.assassin_emb[0]")
            #print(self.assassin_emb[0])
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
                #print("recomm6_vec["+str(i)+"]")
                #print(recomm6_vec[i])
                #print("self.blue_emb["+str(j)+"]")
                #print(self.blue_emb[j])
                sim_blue[i][j] = self.cos_sim(recomm6_vec[i],self.blue_emb[j])
        #print("sim_blue ")
        #print(sim_blue)        

        # create similarity matrix for recomm and civilians words
        num_recomm = recomm6_vec.shape[0]   #number of words in the clean recommendation list
        num_civil = self.civil_emb.shape[0]   #number of words in 'subset' used for centroid


        sim_civil = np.zeros((num_recomm, num_civil))
        for i in range(num_recomm):
            for j in range(num_civil):
                #print("recomm6_vec["+str(i)+"]")
                #print(recomm6_vec[i])
                #print("self.civil_emb["+str(j)+"]")
                #print(self.civil_emb[j])
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
            #print("recomm6_vec["+str(i)+"]")
            #print(recomm6_vec[i])
            #print("center_tensor")
            #print(center_tensor)
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
