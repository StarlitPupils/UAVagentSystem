import json, os, matplotlib.pyplot as plt, numpy as np
baseline = {'MOTA':0.003183,'IDF1':0.005543,'Precision':1.0,'Recall':0.284458}
optimized = {'MOTA':0.0,'IDF1':0.0,'Precision':0.0,'Recall':0.0}
metrics = ['MOTA','IDF1','Precision','Recall']
baseline_vals = [baseline[m] for m in metrics]
optimized_vals = [optimized[m] for m in metrics]
x = np.arange(len(metrics)); width=0.35
fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x-width/2,baseline_vals,width,label='Baseline (YOLOv8n)',color='#2E86AB')
rects2 = ax.bar(x+width/2,optimized_vals,width,label='Optimized (YOLO11l + Cache)',color='#A23B72')
ax.set_ylabel('Score'); ax.set_title('Tracking Performance Comparison')
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.legend(); ax.grid(axis='y',linestyle='--',alpha=0.5)
for rect in rects1+rects2:
    height=rect.get_height()
    ax.annotate(f'{height:.3f}',xy=(rect.get_x()+rect.get_width()/2,height),xytext=(0,3),textcoords="offset points",ha='center',va='bottom',fontsize=9)
plt.tight_layout(); plt.savefig('output/figures/optimization_comparison.png',dpi=300,bbox_inches='tight'); plt.close()
print("对比图表已保存至 output/figures/optimization_comparison.png")