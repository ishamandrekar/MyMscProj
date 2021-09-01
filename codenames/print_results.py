import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd


class print_results():
    

    #bot_res_file = open('bot_results.txt', 'r')
    #res = bot_res_file.readlines()

    data = {}
    codemasters = []
    guessers = []

    cumulative_data = {}

    def __init__(self):
        
        curPair = ""
        curInd = 0
        bot_res_file = open('C:/Users/Isha/Desktop/LJMU/implementation/temp/bot_results_c_paper2_t3_g_paper2.txt', 'r')
        res = bot_res_file.readlines()
        for line in res:
            parts = line.split(" ")
            #TOTAL, BLUE, CIVILLIAN, ASSASSIN, RED, CM, GUESS, TIME
            inner_data = {}
            for p in parts:
                label, val = p.split(":")
                inner_data[label] = val

            #get codemaster and guesser set and make as a pair label
            cm = inner_data["CM"].split("_")[1]
            g = inner_data["GUESSER"].split("_")[1]
            cmk = self.rename_keywords(cm)
            gk =  self.rename_keywords(g)
            if not cmk in self.codemasters:
                self.codemasters.append(cmk)
            if not gk in self.guessers:
                self.guessers.append(gk)
            pair = cmk + "-" + gk

            #reset current pair
            if pair != curPair:
                curPair = pair
                curInd = 0

                #initialize arrays
                self.data[curPair] = {}
                self.data[curPair]['TOTAL'] = []
                self.data[curPair]['B'] = []
                self.data[curPair]['C'] = []
                self.data[curPair]['A'] = []
                self.data[curPair]['R'] = []
                self.data[curPair]['CARDS_LEFT'] = []
                self.data[curPair]['WINS'] = []
                self.data[curPair]['BAD_WORDS_GUESS'] = []

            #add the values
            self.data[curPair]['TOTAL'].append(int(inner_data['TOTAL']))
            self.data[curPair]['B'].append(int(inner_data['B']))
            self.data[curPair]['C'].append(int(inner_data['C']))
            self.data[curPair]['A'].append(int(inner_data['A']))
            self.data[curPair]['R'].append(int(inner_data['R']))

            #calculate the cards left on the board based on the flipped number
            flipped = int(inner_data['B'])+int(inner_data['C'])+int(inner_data['A'])+int(inner_data['R'])
            self.data[curPair]['CARDS_LEFT'].append((25-flipped))

            self.data[curPair]['WINS'].append(((int(inner_data['A']) == 0) and (int(inner_data['B']) != 7)))

            bad_words = int(inner_data['B'])+int(inner_data['C'])+int(inner_data['A'])
            self.data[curPair]['BAD_WORDS_GUESS'].append(bad_words)


 
    #csv_out(data)

    #heatmap_out(data, codemasters, guessers, 'WINS',0.0,1.0)
    #heatmap_out(data, codemasters, guessers, 'CARDS_LEFT')
    #heatmap_out(data, codemasters, guessers, 'R',0.0,8.0)
    #heatmap_out(data, codemasters, guessers, 'BAD_WORDS_GUESS',0.0,17.0)


    def rename_keywords(self,pname):
        if pname=="transformerweighted":
            pkword = "TFW"
        elif pname=="transformer":
            pkword = "TF"
        elif pname=="tfidf":
            pkword = "T"
        elif pname=="naivebayes": 
            pkword = "NB"
        return pkword    
           
    
    def csv_out(self):
        print("Codemaster, Guesser, Win Rate, Average Cards Left, Average Red Words Flipped, Average Bad Words Flipped, Average Turns to Win, Minimum Turns to Win")

        #print as csv string
        for k in self.data.keys():
            #pair
            output = ""
            cg = k.split("-")
            output += (cg[0] + "," + cg[1] + ",")

            #win rate
            wr = self.data[k]['WINS'].count(True)
            #print(wr)
            #print(len(self.data[k]['WINS']))
            output += (str(round(wr/len(self.data[k]['WINS']),2)) + ",")
            self.cumulative_data[k]={}
            self.cumulative_data[k]["WIN_RATE"] = round(wr/len(self.data[k]['WINS']),2)
            

            #avg cards left
            output += (str(round(np.mean(self.data[k]['CARDS_LEFT']),2)) + ",")
            self.cumulative_data[k]["AVG_CARDS_LEFT"] = round(np.mean(self.data[k]['CARDS_LEFT']),2)

            #average red words flipped
            output += (str(round(np.mean(self.data[k]['R']),2)) + ",")
            self.cumulative_data[k]["AVG_RED_WORDS_FLIPPED"] = round(np.mean(self.data[k]['R']),2)

            #average bad words flipped
            output += (str(round(np.mean(self.data[k]['BAD_WORDS_GUESS']),2)) + ",")
            self.cumulative_data[k]["AVG_BAD_WORDS_FLIPPED"] = round(np.mean(self.data[k]['BAD_WORDS_GUESS']),2)

            #average turns to win
            winturns = []
            for t in range(len(self.data[k]['TOTAL'])):
                if self.data[k]['WINS'][t]:
                    winturns.append(self.data[k]['TOTAL'][t])
            if len(winturns) > 0:
                output += (str(round(np.mean(winturns),2)))
                self.cumulative_data[k]["AVG_TURNS_TO_WIN"] = round(np.mean(winturns),2)
            else:
                output += "0"
                self.cumulative_data[k]["AVG_TURNS_TO_WIN"] = 0

            #minimum turns to win
            winturns = []
            for t in range(len(self.data[k]['TOTAL'])):
                if self.data[k]['WINS'][t]:
                    winturns.append(self.data[k]['TOTAL'][t])
            if len(winturns) > 0:
                output += (str(round(np.min(winturns),2)))
                self.cumulative_data[k]["MIN_TURNS_TO_WIN"] = round(np.min(winturns),2)
            else:
                output += "0"
                self.cumulative_data[k]["MIN_TURNS_TO_WIN"] = 0

            #print(output)
     
     
    def heatmap_out(self, col,minn=None,maxx=None): #cm, g, 
        hmDat = np.zeros(shape=(len(self.codemasters),len(self.guessers)))
        for k in self.data.keys():
            self.cg = k.split("-")
            val = self.cumulative_data[k][col]
            hmDat[self.codemasters.index(self.cg[0])][self.guessers.index(self.cg[1])] = val
        #print(hmDat.shape)
        #print(hmDat)
        #hmDat = self.prepare_data_for_plot()
        hmDat = np.transpose(hmDat)
        #print(hmDat)
        if minn != None or maxx != None:
            heat_map = sb.heatmap(hmDat,xticklabels=self.codemasters, yticklabels=self.guessers,annot=True,vmin=minn,vmax=maxx)
        else:
            heat_map = sb.heatmap(hmDat,xticklabels=self.codemasters, yticklabels=self.guessers,annot=True)

        heat_map.xaxis.set_ticks_position('top')
             
        plt.xlabel("Codemasters")
        plt.ylabel("Guessers")
        plt.title("Win Rate",fontsize=20)

        plt.show()


    def barplot_out(self):
        df = pd.DataFrame(columns = ['CG_PAIR', 'AVG_TURNS_TO_WIN', 'MIN_TURNS_TO_WIN', 'AVG_RED_WORDS_FLIPPED', 'AVG_BAD_WORDS_FLIPPED','WIN_RATE'])
        #print(self.cumulative_data)
        for k in self.cumulative_data.keys():
            #print(k)
            new_rec = {'CG_PAIR' : k, 
                       'AVG_TURNS_TO_WIN' : self.cumulative_data[k]['AVG_TURNS_TO_WIN'], 
                       'MIN_TURNS_TO_WIN' : self.cumulative_data[k]['MIN_TURNS_TO_WIN'], 
                       'AVG_RED_WORDS_FLIPPED': self.cumulative_data[k]['AVG_RED_WORDS_FLIPPED'], 
                       'AVG_BAD_WORDS_FLIPPED': self.cumulative_data[k]['AVG_BAD_WORDS_FLIPPED'], 
                       'WIN_RATE':self.cumulative_data[k]['WIN_RATE']}
            df = df.append(new_rec, ignore_index = True)
            
        #print(df)
        
        prep_data1 = pd.melt(df, id_vars=['CG_PAIR'], value_vars=['AVG_TURNS_TO_WIN', 'MIN_TURNS_TO_WIN'],var_name="measure",value_name="value_numbers")
        prep_data2 = pd.melt(df, id_vars=['CG_PAIR'], value_vars=['AVG_RED_WORDS_FLIPPED', 'AVG_BAD_WORDS_FLIPPED'],var_name="measure",value_name="value_numbers")
        
        #print(prep_data1)
        #sns.set_theme(style="whitegrid")
        # Draw a nested barplot by species and sex
        sb.set(rc={'figure.figsize':(11.7,8.27)})
        g = sb.barplot(x="CG_PAIR", y="value_numbers", hue="measure",data=prep_data1)
        plt.xlabel("CG Pair")
        plt.ylabel("")
        plt.title("Avg Turns to Win - Min Turns to Win Comparison for CG Pairs",fontsize=20)
        plt.show()
        
        g = sb.barplot(x="CG_PAIR", y="value_numbers", hue="measure",data=prep_data2)
        plt.xlabel("CG Pair")
        plt.ylabel("")
        plt.title("Avg Red words flipped - Avg Bad words flipped Comparison for CG Pairs",fontsize=20)
        plt.show() 
        
        g = sb.barplot(x="CG_PAIR", y="WIN_RATE",data=df)
        plt.xlabel("CG Pair")
        plt.ylabel("Win Rate")
        plt.title("Win Rate for CG Pairs",fontsize=20)
        plt.show()

    def run(self):
        self.csv_out()
        self.heatmap_out('WIN_RATE',0.0,1.0)
        self.barplot_out()
        #self.heatmap_out('AVG_TURNS_TO_WIN',0.0,1.0)
        #self.heatmap_out('MIN_TURNS_TO_WIN',0.0,1.0)
        #self.heatmap_out('AVG_RED_WORDS_FLIPPED',0.0,1.0)
        #self.heatmap_out('AVG_BAD_WORDS_FLIPPED',0.0,1.0)


#main()
