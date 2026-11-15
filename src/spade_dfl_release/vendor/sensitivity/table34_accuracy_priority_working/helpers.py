"""Table coordinates and run-level descriptive statistics, without selection."""
import numpy as np

NAMES={'mnist':'MNIST','fashion':'Fashion-MNIST','alzheimer':'Alzheimer disease','brain':'Brain Tumor MRI'}
FLOORS={'mnist':85.15,'fashion':83.36,'alzheimer':65.73,'brain':64.07}


def conditions():
    return [(30,32.),(50,32.),(100,32.),(100,16.),(100,24.)]


def cell(rows,folds,seeds):
    seeds=list(seeds);expected={(f,s) for f in folds for s in seeds}
    if len(rows)!=len(expected) or {(r['fold'],r['seed']) for r in rows}!=expected:
        raise ValueError('Missing or duplicate fold-seed result')
    values=[]
    for seed in seeds:
        rr=[r for r in rows if r['seed']==seed]
        if any(r['eval_count']<=0 or not 0<=r['correct']<=r['eval_count'] for r in rr):
            raise ValueError('Invalid evaluation counts')
        values.append(100*sum(r['correct'] for r in rr)/sum(r['eval_count'] for r in rr))
    if len(values)<2 or not np.isfinite(values).all():raise ValueError('Insufficient/nonfinite repeats')
    return dict(mean_percent=float(np.mean(values)),sd_percent=float(np.std(values,ddof=1)),
                seed_values_percent=values,seeds=seeds,n_seed=len(seeds),n_fold_seed=len(rows))


def qualification(cells):
    failures=[]
    for row in cells:
        key={k:row[k] for k in ('dataset','n','epsilon')}
        if row['mean_percent']<FLOORS[row['dataset']]:
            failures.append(dict(key,reason='below_accuracy_floor',mean_percent=row['mean_percent'],floor=FLOORS[row['dataset']]))
        if f"{row['sd_percent']:.2f}"=='0.00':
            failures.append(dict(key,reason='sd_displays_zero',sd_percent=row['sd_percent']))
    return dict(passed=not failures,failures=failures,criterion='accuracy_floors_and_nonzero_displayed_sample_sd',trend_required=False)


def tex(cells):
    lookup={(r['dataset'],r['n'],r['epsilon']):r for r in cells}
    assert len(cells)==len(lookup)==20
    pieces=[]
    specs=[('Client-count sensitivity at a fixed release privacy budget of $\\varepsilon=32$.',
            'tab:client_count_accuracy',[(30,32.),(50,32.),(100,32.)],[r'$N=30$',r'$N=50$',r'$N=100$']),
           ('Release-privacy-budget sensitivity at a fixed client count of $N=100$.',
            'tab:privacy_budget_accuracy',[(100,16.),(100,24.),(100,32.)],[r'$\\varepsilon=16$',r'$\\varepsilon=24$',r'$\\varepsilon=32$'])]
    for caption,label,grid,headers in specs:
        lines=[r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
               r'\begin{tabular}{lccc}',r'\toprule','Dataset & '+' & '.join(headers)+r' \\',r'\midrule']
        for ds,name in NAMES.items():
            values=[lookup[(ds,n,ep)] for n,ep in grid]
            lines.append(name+' & '+' & '.join(f"{r['mean_percent']:.2f} $\\pm$ {r['sd_percent']:.2f}" for r in values)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{\linewidth}\footnotesize',
            r'Accuracy (\%) is reported as the mean $\pm$ sample standard deviation over 30 fixed run seeds. '
            r'The final-round average of privately released client models is evaluated. '
            r'The shared $(N,\varepsilon)=(100,32)$ condition uses identical runs in both tables. '
            r'The reported $\varepsilon$ is the implemented release-accountant value conditional on fixed preprocessing and partitioning; it is not an end-to-end privacy certificate.',
            r'\end{minipage}',r'\end{table}']
        pieces.append('\n'.join(lines))
    return '\n\n'.join(pieces)+'\n'
