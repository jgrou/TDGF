import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class Black_Scholes:
    def __init__(self, r=0.0, sigma=0.25, CP=1):
        self.r = r
        self.sigma = sigma
        self.CP = CP

    # Black-Scholes European option price
    def OptionPrice(self, S, t, sigma=None):
        if sigma == None:
            sigma = self.sigma
            
        d1 = (np.log(S) + (self.r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - (sigma * np.sqrt(t))
        
        if self.CP == 1:
            return S * stats.norm.cdf(d1) - np.exp(-self.r * t) * stats.norm.cdf(d2)
        if self.CP == -1:
            return np.exp(-self.r * t) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

class Plot:
    def __init__(self, matrix, labels, S_plot, name, ymin=-0.1, ymax=2.1):
        self.matrix = matrix
        self.labels = labels
        self.S_plot = S_plot
        self.name = name
        n_plot, self.N_t, _ = np.shape(self.matrix)
        self.n_plot = min(n_plot, len(self.labels)) # Only plot lines for which both values and label is defined
        self.linestyle = ['-',':','--','-.']
        self.colors = ['b','g','r','c','m','y','k']
        self.ymin = ymin
        self.ymax = ymax

    def figures(self, title, n_figures=4, SaveOutput=False, xlabel='Moneyness'):    
        plt.figure(figsize = (12, 5 * (n_figures+1)//2)) # Figure options

        for m,j in enumerate([int(x) for x in np.linspace(0,self.N_t-1,n_figures)]):
            plt.subplot(int(np.ceil(n_figures/2)), 2, m+1) # Specify subplot
            
            # Plot option prices of simulated process values
            for k in range(self.n_plot):
                plt.plot(self.S_plot, self.matrix[k,j], color=self.colors[k%len(self.colors)], label=self.labels[k], linestyle=self.linestyle[k%len(self.linestyle)]) 
    
            # Subplot options
            plt.xlim(xmin=np.min(self.S_plot), xmax=np.max(self.S_plot))
            plt.ylim(ymin=self.ymin, ymax=self.ymax)
            plt.xlabel(xlabel)
            plt.ylabel(r"Option price")
            plt.title(title[j])
            plt.grid(linestyle=':')
    
            if j == 0:
                plt.legend()

        # Save figure
        if SaveOutput:
            plt.savefig('Figures/'+self.name+'.png', bbox_inches='tight') # Remove white space around image
        
        plt.show()
        plt.close()
        
    # Plot of the error
    def error(self, times, SaveOutput=False, order='2'):
        plt.figure()
        
        for j in range(1, self.n_plot):
            if order == '2':
                error = np.linalg.norm(self.matrix[j] - self.matrix[0], 2, 1) / np.linalg.norm(self.matrix[0], 2, 1)
            elif order == 'max': 
                error = np.linalg.norm(self.matrix[j] - self.matrix[0], np.inf, 1)
            plt.plot(times, error, label = self.labels[j], color=self.colors[j%len(self.colors)], linestyle=self.linestyle[j%len(self.linestyle)])
            
        plt.ylim(ymin=0.0)
        plt.xlim(xmin=0.0, xmax=np.max(times))
        plt.xlabel('t')
        if order == '2':
            plt.ylabel('Relative L2-error')
        elif order == 'max':
            plt.ylabel('Max absolute error')
        plt.legend()
        
        # Save figure
        if SaveOutput:
            plt.savefig('Figures/'+self.name+'_'+order+'error.png')
            
        plt.show()
        plt.close()
        
    # Make an image of each timestep for animation
    def make_animation(self, flip=True):
        for j in range(self.N_t):
            ax = plt.axes(xlim=(np.min(self.S_plot),np.max(self.S_plot)), ylim=(self.ymin,self.ymax))
            ax.set_xlabel('Moneyness')
            ax.set_ylabel('Option price')
            ax.grid(linestyle=':')
            
            for k in range(self.n_plot):
                ax.plot(self.S_plot, self.matrix[k][int(2 * (0.5 - flip) * j + flip * (self.N_t - 1))], lw=2, color=self.colors[k%len(self.colors)], label=self.labels[k])
            
            plt.legend()    
            plt.savefig('Figures/'+self.name+'_animation-'+str(j)+'.png')
            plt.close()