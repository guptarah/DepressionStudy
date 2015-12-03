import numpy
import sys
sys.path.append('home/rcf-proj/pg/guptarah/DepressionStudy/Scripts/Exp2/Evaluation/')
from LoadData import LoadData
import matplotlib.pyplot as plt

[depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/Data/PreparedData')

num_sessions = len(vad_ts)
to_plot_mat = numpy.zeros((num_sessions,4))
for i in range(len(vad_ts)):
   to_plot_mat[i,0] = depression_scores[i] 
   to_plot_mat[i,1:4] = numpy.std(vad_ts[i].T,axis=0) 
  

plt.plot(to_plot_mat[:,0],to_plot_mat[:,1],'rs',to_plot_mat[:,0],to_plot_mat[:,2],'bs',to_plot_mat[:,0],to_plot_mat[:,3],'g^') 
