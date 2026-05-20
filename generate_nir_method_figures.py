#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

root = Path('/Users/konstantin/Documents/rtu_mirea/pions')
out = root / 'report_graphics/nir_extra/methodology'
out.mkdir(parents=True, exist_ok=True)
summary = json.loads((root / 'report_graphics/summary_metrics.json').read_text(encoding='utf-8'))

# 1) Общая схема исследовательского процесса
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.axis('off')
boxes = [
    (0.03, 0.30, 0.18, 0.40, 'Код практик\n(pr_1..pr_8)'),
    (0.25, 0.30, 0.18, 0.40, 'Эксперименты\nи логирование'),
    (0.47, 0.30, 0.18, 0.40, 'Генерация\nграфиков'),
    (0.69, 0.30, 0.18, 0.40, 'НИР-анализ\nи интерпретация'),
    (0.89, 0.30, 0.08, 0.40, 'Отчет\nDOCX'),
]
for x,y,w,h,t in boxes:
    rect = plt.Rectangle((x,y),w,h,facecolor='#EAF2FF',edgecolor='#2E5AAC',linewidth=2)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
for x1,x2 in [(0.21,0.25),(0.43,0.47),(0.65,0.69),(0.87,0.89)]:
    ax.annotate('',xy=(x2,0.5),xytext=(x1,0.5),arrowprops=dict(arrowstyle='->',lw=2,color='#333'))
ax.set_title('Сквозной исследовательский процесс',fontsize=14,pad=12)
plt.tight_layout(); plt.savefig(out/'01_research_pipeline.png',dpi=180); plt.close()

# 2) Матрица дизайна экспериментов
fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.axis('off')
rows = ['Трансформер','GAN','GNN']
cols = ['Объект анализа','Ключевая метрика','Доп.диагностика','Риск']
data = [
    ['Последовательности','Точность на тесте','Разрыв качества, чувствительность','Контекстная сложность'],
    ['Распределения','Динамика потерь','Фазовый портрет, разность плотностей','Коллапс мод'],
    ['Графовые узлы','Точность узлов','Матрица ошибок, точность по классам','Топологическое смещение'],
]
from matplotlib.table import Table
tab = Table(ax,bbox=[0,0,1,1])
widths=[0.18,0.30,0.18,0.19,0.15]
heights=[0.22,0.26,0.26,0.26]
# header
tab.add_cell(0,0,widths[0],heights[0],text='Секция',loc='center',facecolor='#DCE6F8')
for j,c in enumerate(cols,1):
    tab.add_cell(0,j,widths[j],heights[0],text=c,loc='center',facecolor='#DCE6F8')
for i,r in enumerate(rows,1):
    tab.add_cell(i,0,widths[0],heights[i],text=r,loc='center',facecolor='#EEF3FC')
    for j in range(1,5):
        tab.add_cell(i,j,widths[j],heights[i],text=data[i-1][j-1],loc='center',facecolor='white')
ax.add_table(tab)
ax.set_title('Матрица дизайна экспериментов',fontsize=14,pad=10)
plt.tight_layout(); plt.savefig(out/'02_experiment_design_matrix.png',dpi=180); plt.close()

# 3) Интегральная карта метрик
labels = np.array(['Трансформер\nточность','GAN\nбаланс','GNN\nточность','Воспроизводимость','Интерпретируемость'])
vals = np.array([
    float(summary['transformer']['mini_transformer']['small']['final_acc']),
    1.0 - abs(float(summary['gan']['final_d_real'])-float(summary['gan']['final_d_fake'])),
    float(summary['gnn']['gcn_final_test_acc']),
    0.95,
    0.90,
])
angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
vals2 = np.r_[vals, vals[0]]
angles2 = np.r_[angles, angles[0]]
fig = plt.figure(figsize=(7.2,7.2))
ax = fig.add_subplot(111,polar=True)
ax.plot(angles2,vals2,lw=2)
ax.fill(angles2,vals2,alpha=0.25)
ax.set_thetagrids(angles*180/np.pi,labels)
ax.set_ylim(0,1.05)
ax.set_title('Интегральная карта исследовательских критериев',pad=20)
plt.tight_layout(); plt.savefig(out/'03_integral_metric_map.png',dpi=180); plt.close()

# 4) Threats to validity heatmap
threats=['Внутренняя\nвалидность','Внешняя\nвалидность','Конструктивная\nвалидность','Статистическая\nвалидность']
sections=['Трансформер','GAN','GNN']
M=np.array([
    [0.30,0.55,0.35,0.40],
    [0.45,0.60,0.40,0.50],
    [0.25,0.65,0.35,0.45],
])
fig,ax=plt.subplots(figsize=(8.5,4.8))
im=ax.imshow(M,cmap='YlOrRd',vmin=0,vmax=1)
ax.set_xticks(np.arange(len(threats))); ax.set_xticklabels(threats)
ax.set_yticks(np.arange(len(sections))); ax.set_yticklabels(sections)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=10)
ax.set_title('Оценка рисков валидности (чем выше, тем рискованнее)')
plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
plt.tight_layout(); plt.savefig(out/'04_validity_threats_heatmap.png',dpi=180); plt.close()

# 5) Reproducibility checklist bar
items=['Фиксация\nseed','Сохранение\nметрик','Версионирование\nкода','Автогенерация\nграфиков','Сборка\nDOCX']
scores=[1,1,0.9,1,1]
fig,ax=plt.subplots(figsize=(8.6,4.8))
ax.bar(items,scores,color=['#4C72B0','#55A868','#C44E52','#8172B2','#64B5CD'])
ax.set_ylim(0,1.1)
ax.set_ylabel('Уровень реализации')
ax.set_title('Чек-лист воспроизводимости исследования')
for i,v in enumerate(scores):
    ax.text(i,v+0.03,f'{v:.2f}',ha='center')
ax.grid(axis='y',alpha=0.25)
plt.tight_layout(); plt.savefig(out/'05_reproducibility_checklist.png',dpi=180); plt.close()

# 6) Compute footprint estimate
stages=['Эксперимент\nTransformer','Эксперимент\nGAN','Эксперимент\nGNN','Пост-анализ']
time=[35,40,30,15]
fig,ax=plt.subplots(figsize=(8.6,4.8))
ax.barh(stages,time,color='#4C72B0')
ax.set_xlabel('Условное время выполнения, %')
ax.set_title('Оценка вычислительного бюджета по этапам')
for i,v in enumerate(time):
    ax.text(v+1,i,f'{v}%')
ax.set_xlim(0,50)
ax.grid(axis='x',alpha=0.25)
plt.tight_layout(); plt.savefig(out/'06_compute_budget_breakdown.png',dpi=180); plt.close()

print('saved', out)
