# GUESSER TRANSFORMER
# CODE BY CATALINA JARAMILLO

from players.guesser import guesser

import torch
#AutoModelWithLMHead
from transformers import BertTokenizer, BertModel #AutoModel, AutoModelForCausalLM,  AutoTokenizer, BertTokenizer, BertModel

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import cosine
import operator

class guesser():
    words = 0
    clue = 0
    clues = []
    
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        pass


class ai_guesser(guesser):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        #torch.set_grad_enabled(False)

        #self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        torch.no_grad()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')  #GPT2Tokenizer.from_pretrained('gpt2')
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-large-uncased',
                                                output_hidden_states = True,) #AutoModelForCausalLM.from_pretrained('gpt2')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        
        
        self.curGuesses = []
        self.words_start_state = []
        

    def get_clue(self, clue, num):
        self.curGuesses = []
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num, sep=" ")
        li = [clue, num]
        return li

    def get_board(self, words):
        self.words = words
        if len(self.words_start_state) == 0:
            self.words_start_state = list(map(lambda w: w.lower(), words)) 
            self.board_emb = self.word_embedding(self.words_start_state)

    def give_answer(self):
        #clue_emb = self.word_embedding(self.spec_palabras([self.clue]))
        
        if len(self.curGuesses)==0:
            clue_emb = self.word_embedding([self.clue])
            print(clue_emb.shape[0])
            usable_words = [w for w in self.words if "*" not in w]
            usable_words = list(map(lambda w: w.lower(), usable_words))
            print("usable_words")
            print(usable_words)
            #board_emb = self.word_embedding(self.spec_palabras(usable_words))
            #board_emb = self.word_embedding(usable_words)

            num_board_words = len(usable_words)
                    
            sim_board_words = np.zeros((num_board_words),)
            print("self.words_start_state")
            print(self.words_start_state)
            for i in range(num_board_words):
                #get the start index of the word
                curr_word = usable_words[i]
                print("curr_word")
                print(curr_word)
                start_indx = self.words_start_state.index(curr_word)
                print("self.words_start_state[start_indx]")
                print(self.words_start_state[start_indx])
                sim_board_words[i] = self.cos_sim(self.board_emb[start_indx],clue_emb[0]) #(board_emb[i],clue_emb[0])

            #asc_idx = np.argpartition(sim_board_words, self.num)
            #print([usable_words[i] for i in list(asc_idx)])
            #print(asc_idx)
            #self.curGuesses = [usable_words[i] for i in list(asc_idx[:self.num])]
            #print(self.curGuesses)
            print("sim_board_words")
            print(sim_board_words)          
            sim_board_words_list = sim_board_words.tolist()
            print("sim_board_words_list")
            print(sim_board_words_list)
            enumerate_object = enumerate(sim_board_words_list)
            sorted_pairs = sorted(enumerate_object, key=operator.itemgetter(1))
            sorted_indices = [index for index, element in sorted_pairs][:self.num]
            print("words sorted by ascending order of distance")
            print([usable_words[index] for index, element in sorted_pairs])
            self.curGuesses = [usable_words[i] for i in sorted_indices]
            bestGuess = self.curGuesses.pop(0)
        else:
            bestGuess = self.curGuesses.pop(0)
            
            #min_index = np.argmin(sim_board_words)
            #bestGuess = usable_words[min_index]
        # look (number) of nearest neighbors 
        #knn = NearestNeighbors(n_neighbors = self.num)
        #knn.fit(board_emb)
        #vecinos = knn.kneighbors(clue_emb.reshape(1,-1))

        #add the best guesses into a pending list
        #for i in range(self.num):
            #guess = usable_words[vecinos[1][0][i]]
            #d = vecinos[0][0][i]
            #self.curGuesses.append(guess+"|"+str(d))

        #resort the guesses and choose the closest one
        #self.reSortGuesses()
        #while True:
            #bestGuess = self.curGuesses.pop(0).split("|")[0]
            #if bestGuess in usable_words:
                #break
        return bestGuess                #returns a string for the guess



    def keep_guessing(self, clue, num):
        return len(self.curGuesses) #> 0

    def is_valid(self, result):
        if result.upper() in self.words or result == "":
            return True
        else:
            return False




    #add "\u0120" in front of each word to improve embedding result
    def spec_palabras(self, palabras):
        spec_palabras = list(map(lambda w: "\u0120" + w, palabras))
        return spec_palabras
  
    def cos_sim(self, input1, input2):
        cos = cosine(input1.mean(axis=0).detach().numpy(), input2.mean(axis=0).detach().numpy())
        return cos #cos(input1, input2)

    #create word vectors for each word
    def word_embedding(self, words):
        tokenized_texts = [self.tokenizer.tokenize("[CLS] " + word + " [SEP]") for word in words]
        tokenized_texts = [[item[0],item[1]+item[2].replace("#",""),item[3]] if (len(item)>3 and item[2].startswith("#")) else item for item in tokenized_texts]
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
        segments_ids = [[1] * len(tokenized_text) for tokenized_text in tokenized_texts]
        tokens_tensors = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)
        outputs = self.model(tokens_tensors, segments_tensors)
        return outputs.last_hidden_state #word_emb
    
    #sort the guesses based on value
    def reSortGuesses(self):
        #split
        sortD = {}
        for g in self.curGuesses:
            p = g.split("|")
            sortD[str(p[0])] = float(p[1])

        #sort + reform
        newsort = []
        for k, v in sorted(sortD.items(), key=lambda item: float(item[1])):
            if k.upper() in self.words:
                newsort.append(str(k) + "|" + str(v))

        self.curGuesses = newsort



