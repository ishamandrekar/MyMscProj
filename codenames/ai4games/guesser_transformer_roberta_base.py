# GUESSER TRANSFORMER
# CODE BY CATALINA JARAMILLO

from players.guesser import guesser

import torch

from transformers import RobertaTokenizer, RobertaModel

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import cosine
import operator
from torch.nn import functional as F

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
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') #RobertaTokenizer.from_pretrained('roberta-base')  
        #self.tokenizerbert = BertTokenizer.from_pretrained('bert-base-uncased')
        #GPT2Tokenizer.from_pretrained('gpt2')
        # Load pre-trained model (weights)
        self.model = RobertaModel.from_pretrained('roberta-base',
                                                output_hidden_states = False,) #AutoModelForCausalLM.from_pretrained('gpt2')
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
            #self.board_emb = output.last_hidden_state

    def give_answer(self):
        #clue_emb = self.word_embedding(self.spec_palabras([self.clue]))
        
        if len(self.curGuesses)==0:
            clue_emb = self.word_embedding([self.clue])
            #clue_emb = output.last_hidden_state
            #print(clue_emb.shape[0])
            usable_words = [w for w in self.words if "*" not in w]
            usable_words = list(map(lambda w: w.lower(), usable_words))
            #print("usable_words")
            #print(usable_words)
            #board_emb = self.word_embedding(self.spec_palabras(usable_words))
            #board_emb = self.word_embedding(usable_words)

            num_board_words = len(usable_words)
                    
            sim_board_words = np.zeros((num_board_words),)
            #print("self.words_start_state")
            #print(self.words_start_state)
            for i in range(num_board_words):
                #get the start index of the word
                curr_word = usable_words[i]
                #print("curr_word")
                #print(curr_word)
                start_indx = self.words_start_state.index(curr_word)
                #print("self.board_emb[start_indx][0]")
                #print(self.board_emb[start_indx][0].shape)
                #print(type(self.board_emb[start_indx]))
                #print(type(self.board_emb))
                #print("clue_emb[0]")
                #print(clue_emb[0].shape)
                #print(type(clue_emb[0]))
                sim_board_words[i] = self.cos_sim(self.board_emb[start_indx][0],clue_emb[0][0]) #(board_emb[i],clue_emb[0])

            #asc_idx = np.argpartition(sim_board_words, self.num)
            #print([usable_words[i] for i in list(asc_idx)])
            #print(asc_idx)
            #self.curGuesses = [usable_words[i] for i in list(asc_idx[:self.num])]
            #print(self.curGuesses)
            #print("sim_board_words")
            #print(sim_board_words)          
            sim_board_words_list = sim_board_words.tolist()
            #print("sim_board_words_list")
            #print(sim_board_words_list)
            enumerate_object = enumerate(sim_board_words_list)
            sorted_pairs = sorted(enumerate_object, key=operator.itemgetter(1))
            sorted_indices = [index for index, element in sorted_pairs][:self.num]
            #print("words sorted by ascending order of distance")
            #print([usable_words[index] for index, element in sorted_pairs])
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
        #print(input1.mean(axis=0).detach().numpy().shape)
        #print(input2.mean(axis=0).detach().numpy().shape)
        #input1_flattened = input1.mean(axis=0).detach().numpy().flatten()
        #input2_flattened = input2.mean(axis=0).detach().numpy().flatten()
        #maxlen = max([input1_flattened.shape[0],input2_flattened.shape[0]])
        #input1_flattened_padded = np.pad(input1_flattened, (0, maxlen - input1_flattened.shape[0]), 'constant')
        #input2_flattened_padded = np.pad(input2_flattened, (0, maxlen - input2_flattened.shape[0]), 'constant')
        cos = cosine(input1.mean(axis=0).detach().numpy(), input2.mean(axis=0).detach().numpy())
        return cos #cos(input1, input2)

    #create word vectors for each word
    """
    def word_embedding(self, words):
        inputs = [self.tokenizer("[CLS] " + word + " [SEP]", return_tensors="pt") for word in words]
        input_ids_list = []
        attention_mask_list = []
        for element in inputs:
            input_ids_list.append(element['input_ids'][0])
            attention_mask_list.append(element['attention_mask'][0])
        print("Guesser word_embedding")      
        for element in input_ids_list:
            print(element.shape)
        stacked_tensor_input_ids = torch.stack(input_ids_list)
        stacked_tensor_attention_masks = torch.stack(attention_mask_list)
        outputs = model(input_ids=stacked_tensor_input_ids, attention_mask=stacked_tensor_attention_masks)
        return outputs.last_hidden_state #word_emb
        """

    
    #create word vectors for each word
    """
    def word_embedding(self, words):
        inputs = [self.tokenizer("[CLS] " + word + " [SEP]") for word in words]
        input_ids_list = []
        attention_mask_list = []
        max_len = 0
        for element in inputs:
            #print(element)
            #print(type(element['input_ids']))
            curr_len = len(element['input_ids'])
            if curr_len > max_len:
                max_len = curr_len
            input_ids_list.append(element['input_ids'])
            attention_mask_list.append(element['attention_mask'])
        #print("Guesser word_embedding")    
        input_ids_list = [l1 + [0] * (max_len - len(l1)) for l1 in input_ids_list]
        attention_mask_list = [l1 + [0] * (max_len - len(l1)) for l1 in attention_mask_list]
        input_ids_list = [torch.tensor(l1, dtype=torch.int) for l1 in input_ids_list]
        attention_mask_list = [torch.tensor(l1, dtype=torch.int) for l1 in attention_mask_list]
        stacked_tensor_input_ids = torch.stack(input_ids_list)
        stacked_tensor_attention_masks = torch.stack(attention_mask_list)
        print("stacked_tensor_input_ids.size")
        print(stacked_tensor_input_ids.size())
        outputs = self.model(input_ids=stacked_tensor_input_ids, attention_mask=stacked_tensor_attention_masks)  
        return outputs.last_hidden_state"""
     
    #create word vectors for each word
    def word_embedding(self, red_words):
        #word_emb = []
        #for word in red_words:
        #text_index = self.tokenizer.encode(red_words,add_prefix_space=False)
        #tokenized_texts = [self.tokenizer.encode(word, max_length=2, pad_to_max_length=True) for word in red_words]
        tokenized_texts = [self.tokenizer("[CLS] " + word + " [SEP]", is_split_into_words=True) for word in red_words]
        #tokenized_texts = [[item[0],item[1]+item[2].replace("#",""),item[3]] if (len(item)>3 and item[2].startswith("#")) else item for item in tokenized_texts]
        
        #tokenized_texts = [x for x in tokenized_texts if x.count("`")==0]
        #print("tokenized_texts")
        #print(tokenized_texts)
        # Map the token strings to their vocabulary indeces.
        #indexed_tokens = [d1['input_ids'] for d1 in tokenized_texts]
        #[self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
        #print("indexed_tokens")
        #print(indexed_tokens)
        #print(type(indexed_tokens))
        # Mark each of the 22 tokens as belonging to sentence "1".
        #segments_ids = [d1['attention_mask'] for d1 in tokenized_texts] #tokenized_texts['attention_mask']
        #[[1] * len(tokenized_text) for tokenized_text in tokenized_texts]
        
        # Convert inputs to PyTorch tensors
        #tokens_tensors = torch.tensor(indexed_tokens)
        #segments_tensors = torch.tensor(segments_ids)
        #outputs = self.model(tokens_tensors, segments_tensors)
        #word_emb.append(output)
        #word_emb = self.model.transformer.wte.weight[text_index,:]
        
        outputs_l = []
        max_len = 0
        for d1 in tokenized_texts:
            indexed_token = d1['input_ids']
            segments_id = d1['attention_mask']
            if len(indexed_token)>max_len:
                max_len = len(indexed_token)      
            tokens_tensor = torch.tensor(indexed_token)
            segments_tensor = torch.tensor(segments_id)
            #print(tokens_tensor)
            #print(segments_tensor)
            
            output = self.model(tokens_tensor.unsqueeze(0), segments_tensor.unsqueeze(0))
            outputs_l.append(output.last_hidden_state)
            
            #print(output.last_hidden_state.shape)  
            #print(output.last_hidden_state.shape[1])  
        for num,tensor_element in enumerate(outputs_l):
            if tensor_element.shape[1]<max_len:
                #print("I am here")
                pad_len = max_len - tensor_element.shape[1]
                padding = (
                            0,0,   # Fill 1 unit in the front and 2 units in the back
                            0,pad_len,
                            0,0
                           )
                outputs_l[num] = F.pad(tensor_element, padding)
                #print(outputs_l[num].shape)  
            
        outputs = torch.stack(outputs_l)
        return outputs
    
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



