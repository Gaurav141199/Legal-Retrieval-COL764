# importing library
import matplotlib.pyplot as plt
  
# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
  
if __name__ == '__main__':
    # creating data on which bar chart will be plot
    x = ["TF-IDF", "BM25", "BERT \n Naive", "AILA\nBEST"]
    y = [0.1195054387, 0.114998746, 0.01587100846, 0.1412]

    # making the bar chart on the data
    barlist = plt.bar(x, y)
    barlist[3].set_color('g')  
    # calling the function to add value labels
    addlabels(x, y)
      
    # giving title to the plot
    plt.title("MAP")
      
    # giving X and Y labels
    plt.xlabel("Methods")
    plt.ylabel("Values")
      
    # visualizing the plot
    plt.show()
