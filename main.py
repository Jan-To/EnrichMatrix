#!/usr/bin/env python
# coding: utf-8

### Import Libraries

import numpy as np
import pandas as pd
from math import pi
import os

from matplotlib.colors import ListedColormap, rgb2hex, LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib import cm

def get_cmap_from_mpl(key):
    cmap = cm.get_cmap(key, 128)
    cmap_list = []
    for i in range(cmap.N):
        cmap_list.append(rgb2hex(cmap(i)))
    return cmap_list
summer = get_cmap_from_mpl('summer')
winter = get_cmap_from_mpl('winter')
cool   = get_cmap_from_mpl('cool')
highlight_color = '#faa43a'
select_color    = '#4d4d4d'
area_select_color = '#000000'
unselect_color  = '#cccccc'
area_unselect_color = '#adadad'
line_color      = '#333333'
selection_colors = ["#5da4da", "#f15754", "#60bd67", "#b276b2"] * 10
unselection_colors = ["#95B8D3", "#DF9290", "#96C59A", "#BFA1BF"] * 10

css = '''
button.bk.bk-btn.bk-btn-primary {
    color: #000 !important;
    background: rgba(68, 140, 202, 0.2) !important;
}
button.bk.bk-btn.bk-btn-primary.bk-active {
    color: #000 !important;
    background: rgba(68, 140, 202, 1) !important;
}
button.bk.bk-btn.bk-btn-success {
    color: #000 !important;
    background: rgba(96, 189, 104, 0.2) !important;
}
button.bk.bk-btn.bk-btn-success.bk-active {
    color: #000 !important;
    background:rgba(96, 189, 104, 1) !important;
}
button.bk.bk-btn.bk-btn-warning {
    color: #000 !important;
    background: rgba(241, 88, 84, 0.2) !important;
}
button.bk.bk-btn.bk-btn-warning.bk-active {
    color: #000 !important;
    background: rgba(241, 88, 84, 1) !important;
}
button.bk.bk-btn.bk-btn-danger {
    color: #000 !important;
    background: rgba(178, 118, 178, 0.2) !important;
}
button.bk.bk-btn.bk-btn-danger.bk-active {
    color: #000 !important;
    background: rgba(178, 118, 178, 1) !important;
}
button.bk.bk-btn.bk-btn-default {
    color: #000 !important;
    background: rgba(250, 163, 56, 1) !important;
}
'''

from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, Range1d, CDSView, IndexFilter, Title
from bokeh.palettes import RdYlGn, Category10, Category20, Viridis256, Turbo256, Greys256
from bokeh.models.tools import TapTool, BoxSelectTool, HoverTool, WheelZoomTool, LassoSelectTool
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.layouts import gridplot, row
from bokeh.transform import linear_cmap

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances
import scipy.cluster.hierarchy as shc
from statsmodels.nonparametric.kde import KDEUnivariate

import networkx as nx
import pygraphviz as pgv

import panel as pn
pn.extension(raw_css=[css], loading_spinner='dots', loading_color=highlight_color)


### Load data

### TODO clean up data loading to make it more general purpose ###

df_solu = pd.read_csv('./data/solu.csv').convert_dtypes()
df_solv = pd.read_csv('./data/solv.csv').convert_dtypes()

physical_prop_names = ['Dipol Mom._normed', 'Polarizab._normed', 'norm. Aniso._normed', 'H-Bond Acc._normed', 'H-Bond Don._normed']
violin_names = ['Dipol Mom.','Mol.Weight','Polarizab.','Anisotr.','norm. Aniso.','H-Bond Acc.','H-Bond Don.','Tm [K]','Tc [K]']
unifac_names = df_solu.columns[6:59].tolist()

solute_names  = pd.read_excel('./data/physicochemical_attributes.xlsx', sheet_name='Full_solu_names', engine='openpyxl').Name.tolist()
solvent_names = pd.read_excel('./data/physicochemical_attributes.xlsx', sheet_name='Full_solv_names', engine='openpyxl').Name.tolist()
# MCM predictions as matrix
full_m = pd.read_csv("./data/MCM_Predictions_Full.csv")
full_m.columns = solvent_names
full_m.index = solute_names
whisky_m = pd.read_csv("./data/Whisky_Full.csv")
whisky_m.columns = solvent_names
whisky_m.index = solute_names
# Experimental ground truth as matrix
truth_m = pd.read_csv("./data/Ground_Truth.csv")
truth_m.columns = solvent_names
truth_m.index = solute_names
# UNIFAC predictions as matrix
UNIFAC_m = pd.read_csv("./data/UNIFAC_Predictions_Full.csv")
UNIFAC_m.columns = solvent_names
UNIFAC_m.index = solute_names

WHI_pred = whisky_m.loc[df_solu['Name'],df_solv['Name']]
MCM_pred = full_m.loc[df_solu['Name'],df_solv['Name']]
UNI_pred = UNIFAC_m.loc[df_solu['Name'],df_solv['Name']]
TRU_pred = truth_m.loc[df_solu['Name'],df_solv['Name']]

def loadLegend(name='Single Label', key='solutes'):
    if name == 'Multiple Labels':
        if key == 'solutes':
            df = pd.read_excel('./data/physicochemical_attributes.xlsx', sheet_name='chem_solu', engine='openpyxl', usecols='A:H')
            legend = df.loc[0:48,'Abb.':'Beschreibung']
        elif key == 'solvents':
            print('----------- ERROR: EXCEL DATA IS STILL MISSING --------------')
            df = pd.read_excel('./data/physicochemical_attributes.xlsx', sheet_name='chem_solv', engine='openpyxl', usecols='A:H')
            legend = df.loc[0:48,'Abb.':'Beschreibung'] # TODO change number to correct index, when data is available
        return legend
    elif name == 'Single Label':
        if key == 'solutes':
            df = pd.read_excel('./data/physicochemical_attributes.xlsx', sheet_name='subgroups_solu', engine='openpyxl', usecols='A:K')
            legend = df.iloc[0:41,7:11].rename(columns={'alias':'Abb.', 'physikochemische Grundlage':'Beschreibung'})
        elif key == 'solvents':
            df = pd.read_excel('./data/physicochemical_attributes.xlsx', sheet_name='subgroups_solv', engine='openpyxl', usecols='A:K')
            legend = df.iloc[0:51,7:11].rename(columns={'alias':'Abb.', 'physikochemische Grundlage':'Beschreibung'})
        legend.drop(['c.g.n.','explizit'], axis=1, inplace=True)
        return legend
    else:
        print('ERROR loading legend with name:', name)
        
### Clustering

def cluster(solu_basis, solv_basis, distance_func, linkage_func, distance_thres=0.001):
    ### cluster affiliation ###
    solu_cluster_model = AgglomerativeClustering(distance_threshold=distance_thres, metric=distance_func, linkage=linkage_func, n_clusters=None)
    solv_cluster_model = AgglomerativeClustering(distance_threshold=distance_thres, metric=distance_func, linkage=linkage_func, n_clusters=None)
    solu_cluster_model.fit(solu_basis)
    solv_cluster_model.fit(solv_basis)
    ### hierarchical sorting ###
    solu_dend = shc.dendrogram(shc.linkage(solu_basis, metric=distance_func if distance_func != 'manhattan' else 'cityblock', method=linkage_func), no_plot=True)
    solv_dend = shc.dendrogram(shc.linkage(solv_basis, metric=distance_func if distance_func != 'manhattan' else 'cityblock', method=linkage_func), no_plot=True)
    
    return solu_cluster_model, solv_cluster_model, np.argsort(solu_dend['leaves']), np.argsort(solv_dend['leaves'])

def createClusterTree(clu, horizontal=True, order=None):
    def getClusterTree(clu, hor):
        chdr = clu.children_

        Gpgv = pgv.AGraph(directed=True)
        Gpgv.graph_attr["rankdir"] = "RL" if hor else "TB"

        for p,(l,r) in enumerate(chdr, len(chdr)+1):
            Gpgv.add_edge(str(p),str(l))
            Gpgv.add_edge(str(p),str(r))
            
        # force order over additional edges
        if order is not None:
            for l,r in zip(order, order[1::]):
                Gpgv.add_edge(str(l), str(r))

        B=Gpgv.add_subgraph(list(range(len(chdr)+1)),name='s1')
        B.graph_attr['rank'] = 'same'

        Gpgv.layout(prog="dot")
        
        # remove additional edges after layouting
        if order is not None:
            for l,r in zip(order, order[1::]):
                Gpgv.remove_edge(str(l), str(r))

        return Gpgv
        
    Gpgv = getClusterTree(clu, horizontal)
    Gnx = nx.DiGraph(Gpgv)

    n = len(clu.distances_) + 1
    k = 1 if horizontal else 0
    x_max = max([float(p.split(',')[k]) for i,p in Gnx.nodes(data='pos')])
    x_min = min([float(p.split(',')[k]) for i,p in Gnx.nodes(data='pos')])
    def scale(x, flip=False):
        scaled = (float(x) - x_min) / (x_max-x_min) * (n-2) + 0.5
        return str(-1*scaled + n-1) if flip else str(scaled)
    clu_id, clu_x, clu_y = [], [], []

    for nd in nx.dfs_postorder_nodes(Gnx, source=str(2*(n-1))):
        nid = int(nd)
        if nid >= n:
            if horizontal:
                x = clu.distances_[int(nd)-n]
                y = np.mean([float(Gnx.nodes()[b]['pos'].split(',')[1]) for a,b in Gnx.out_edges(nd)])
            else:
                x = np.mean([float(Gnx.nodes()[b]['pos'].split(',')[0]) for a,b in Gnx.out_edges(nd)])
                y = clu.distances_[int(nd)-n]
            Gnx.nodes()[nd]['pos'] = '{:f},{:f}'.format(x,y)
            clu_id.append(nd)
            clu_x.append(x)
            clu_y.append(y)
        else:
            x,y = Gnx.nodes()[nd]['pos'].split(',')
            Gnx.nodes()[nd]['pos'] = '0,'+scale(y) if horizontal else scale(x,flip=True)+',0'
            
    nx.set_node_attributes(Gnx, 0, "selected")
    nx.set_node_attributes(Gnx, -1, "selection_id")
            
    return Gnx, Gpgv, clu_id, clu_x, clu_y


### Util Functions (Measures / Colormaps)

def frgb2hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*[int(min(i,0.999)*256) for i in color[:3]])

def createColormap(center=0.5, res=256):
    cdict = {}
    blues = [[0,49/255,105/255], [0/255, 106/255, 171/255], [73/255,159/255,203/255], [161/255, 206/255, 226/255], [229/255, 228/255, 226/255]]
    oranges = [[229/255, 228/255, 226/255], [1,218/255,185/255], [1, 167/255, 106/255], [248/255, 109/255, 41/255], [196/255,57/255,13/255], [129/255,35/255,9/255]]
    for channel, i in zip(['red','green','blue'],[0,1,2]):
        channel_colors = [[(j/(len(blues)-1))*center, blues[j][i], blues[j][i]] for j in range(len(blues))]
        channel_colors.extend([[(j/(len(oranges)-1))*(1-center)+center, oranges[j][i], oranges[j][i]] for j in range(len(oranges))])
        cdict[channel] = channel_colors
    tempcm = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=res)(np.linspace(0, 1, res))
    colors = [frgb2hex(c) for c in tempcm]
    return colors
    
def getColormap(name):
    if   name == 'viridis': return Viridis256
    elif name == 'rainbow': return Turbo256
    elif name == 'greyscale': return Greys256[::-1]
    elif name == 'InverseGreyscale': return Greys256
    elif name == 'RedYellowGreen': return RdYlGn[11]
    elif name == 'summer': return summer
    elif name == 'winter': return winter
    elif name == 'cool': return cool
    elif name == 'CoolWarmDistortedOld': ### Blue/White/DarkOrange
        nblue = int(100 * 3/19)
        norange = 100 - nblue
        top = cm.get_cmap('Blues_r', nblue)
        bottom = cm.get_cmap('Oranges', norange)
        newcolors = np.vstack((top(np.linspace(0, 1, nblue)),[1,1,1,1],
                               bottom(np.linspace(0, 1, norange-8))))
        newcmp = ListedColormap(newcolors, name='OrangeBlue')
        colors = [frgb2hex(c) for c in newcmp.colors]
        return colors
    elif name == 'CoolWarmDistorted':
        return createColormap(center=0.17)
    elif name == 'CoolWarm':
        return createColormap(center=0.5)
    elif name == 'Warm':
        newcmp = cm.get_cmap('Oranges', 256)(np.linspace(0, 1, 256))
        colors = [frgb2hex(c) for c in newcmp]
        return colors
    
cm_low = -1
cm_high = 5

def computePrevalence(n, G, df, group=None):
    group = group if group is not None else slice(df_legend.iloc[0].loc['Abb.'], df_legend.iloc[-1].loc['Abb.'])
    # get all leaves
    leaf_indices = [int(nd) for nd in nx.dfs_preorder_nodes(G, source=n) if int(nd) < len(df)]
    # get their prevalence
    prevalence = df.loc[leaf_indices, group].mean()
    # remove zeros and sort
    prevalence = prevalence.loc[prevalence != 0].sort_values(ascending=False)
    if len(prevalence) == 0:
        prevalence = pd.Series(0)
    return prevalence

def getTopNPrevalence(i, G, df, as_text=False, n=5):
    series = computePrevalence(i, G, df)
    if series.size < n:
        series = pd.concat([series,(pd.Series([0]*(n-series.size), index=[str(i) for i in range(series.size+1,n+1)]))])
    topN = series.iloc[:n]
    if not as_text:
        return topN
    topN_text = [str(int(v*100))+'% '+str(i) for i,v in topN.items()]
    return topN_text

# coming from BinaryCrossEntropy-Loss: 
# -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean() 
# with y=labels and y_hat=predictions. We assume perfect groups with y = 1
def meanOfBinaryCrossEntropy(series):
    return np.mean(np.array([-1*np.log2(p) for p in series if p > 0.0001]))

def entropy(series):
    return np.sum(np.array([-p*np.log2(p) for p in series if p > 0.0001]))

# compute set-variance = (sum of dist(v,v_mean))**2 / |set| with dist = mahattan/euclidean
def setVariance(n, G, df):
    # get all leaves
    leaf_indices = [int(nd) for nd in nx.dfs_preorder_nodes(G, source=n) if int(nd) < len(df)]
    v_list = df.loc[leaf_indices, df_legend.iloc[0].loc['Abb.']:df_legend.iloc[-1].loc['Abb.']].to_numpy()
    v_mean = df.loc[leaf_indices, df_legend.iloc[0].loc['Abb.']:df_legend.iloc[-1].loc['Abb.']].mean().to_numpy()
    dividend = np.mean(np.linalg.norm(v_list - v_mean, ord=1, axis=1)**2)
    return dividend

def setStandardDeviation(n,G,df):
    return np.sqrt(setVariance(n,G,df))

# compute Gini Index. Does not make sense with multi-label, as they do not sum to 1?
def GiniIndex(n, G, df):
    series = computePrevalence(n, G, df)
    series = series / np.sum(series)
    return 1 - np.sum(np.square(series))

coloring_options = ['Entropy', 'Information Gain', 'meanBinaryCrossEntropy', 'mBCEIncreaseToChildren', 'Physical Props Mean SD', 'Physical Props Mean SD Diff']

# define coloring schema 
def getColoring(df, df_legend, G, clu_id, color_by, cm, basis):
    if color_by == 'None':
        node_colors = (select_color,)*len(clu_id)
    elif color_by == 'Most frequent group':
        no1groups = [df_legend[df_legend['Abb.'] == computePrevalence(i,G,df).index[0]].index[0] for i in clu_id]
        # get occuring first groups
        occ_groups = np.unique(no1groups)
        if len(occ_groups) <= 20:
            hint_layout.objects = [pn.pane.Markdown('#### There are '+str(len(occ_groups))+' occuring most frequent groups. Color assignment in tree is unique.', width=1000)]
        else:
            hint_layout.objects = [pn.pane.Markdown('#### Warning: There are '+str(len(occ_groups))+' occuring most frequent groups for 20 colors. Color assignment in tree is NOT UNIQUE!', width=1000)]
        cm = Category20[20]*3 + ('#444444',)*40
        node_colors = [cm[np.where(occ_groups == i)[0][0]] for i in no1groups]
    elif color_by == 'meanBinaryCrossEntropy' or color_by == 'mBCETopN':
        if color_by == 'mBCETopN':
            node_colors = np.array([meanOfBinaryCrossEntropy(getTopNPrevalence(i,G,df)) for i in clu_id])
        else:
            node_colors = np.array([meanOfBinaryCrossEntropy(computePrevalence(i,G,df)) for i in clu_id])
    elif color_by == 'mBCEIncreaseToChildren':
        mBCE = np.array([meanOfBinaryCrossEntropy(computePrevalence(i,G,df)) for i in clu_id])
        for i, parent_id in enumerate(clu_id):
            child_mBCE = [0 if int(child_id) < len(df) else mBCE[clu_id.index(child_id)] for child_id in G.adj[parent_id]]
            mBCE[i] -= np.mean(child_mBCE)
        node_colors = mBCE
    elif color_by == 'setVariance':
        node_colors = np.array([setVariance(i,G,df) for i in clu_id])
    elif color_by == 'setStandardDeviation':
        node_colors = np.array([setStandardDeviation(i,G,df) for i in clu_id])
    elif color_by == 'setVarianceIncreaseToChildren':
        set_variances = np.array([setVariance(i,G,df) for i in clu_id])
        for i, parent_id in enumerate(clu_id):
            child_set_variances = [0 if int(child_id) < len(df) else set_variances[clu_id.index(child_id)] for child_id in G.adj[parent_id]]
            set_variances[i] -= np.mean(child_set_variances)
        node_colors = set_variances
    elif color_by == 'Entropy':
        node_colors = np.array([entropy(computePrevalence(i,G,df)) for i in clu_id])
    elif color_by == 'Information Gain':
        # compute size of each node
        node_sizes = np.ones(len(clu_id))
        for i, parent_id in enumerate(clu_id): 
            node_sizes[i] = len([nd for nd in nx.dfs_preorder_nodes(G, source=parent_id) if int(nd) < len(df)])
        # compute entropy of each node
        node_entropy = np.array([entropy(computePrevalence(i,G,df)) for i in clu_id])
        # substract entropy of child nodes from parent (weighted by child size)
        mutualInformation = np.zeros(len(node_entropy))
        for i, parent_id in enumerate(clu_id):
            child_entropy = [0 if int(child_id) < len(df) else (node_sizes[clu_id.index(child_id)] / node_sizes[clu_id.index(parent_id)]) * node_entropy[clu_id.index(child_id)] for child_id in G.adj[parent_id]]
            mutualInformation[i] = node_entropy[i] - np.sum(child_entropy)
        node_colors = mutualInformation
    elif color_by == 'GiniIndex':
        print('Remember that Gini Index is only sensible, if distribution probabilities sum to 1.')
        node_colors = np.array([GiniIndex(i,G,df) for i in clu_id])
    elif color_by in df_legend['Beschreibung'].tolist():
        symbol = df_legend[df_legend['Beschreibung'] == color_by]
        symbol = symbol['Abb.']
        node_colors = np.array([computePrevalence(i,G,df, group=symbol) for i in clu_id]).reshape(-1)
    elif color_by == 'Physical Props Mean SD':
        if all([x in df.columns for x in physical_prop_names]):
            basis = df[physical_prop_names]
            node_colors = []
            for i in clu_id:
                leaf_indices = [int(nd) for nd in nx.dfs_preorder_nodes(G, source=i) if int(nd) < len(df)]
                leaves = basis.loc[leaf_indices]
                node_colors.append(leaves.std().mean())
    elif color_by == 'Physical Props Mean SD Diff':
        if all([x in df.columns for x in physical_prop_names]):
            # compute physical props mean sd for all nodes
            basis = df[physical_prop_names]
            mean_sd = []
            for i in clu_id:
                leaf_indices = [int(nd) for nd in nx.dfs_preorder_nodes(G, source=i) if int(nd) < len(df)]
                leaves = basis.loc[leaf_indices]
                mean_sd.append(leaves.std().mean())
            # substract mean sd of child nodes from parent
            mean_sd_diff = np.zeros(len(mean_sd))
            for i, parent_id in enumerate(clu_id):
                child_mean_sd = [0 if int(child_id) < len(df) else mean_sd[clu_id.index(child_id)] for child_id in G.adj[parent_id]]
                mean_sd_diff[i] = mean_sd[i] - np.mean(child_mean_sd)
            node_colors = mean_sd_diff
    elif color_by == 'MeanPairwiseDistances':
        node_colors = []
        # for all nodes
        for i in clu_id:
            # get all leaves
            leaf_indices = [int(nd) for nd in nx.dfs_preorder_nodes(G, source=i) if int(nd) < len(df)]
            # get their AC vectors
            leaf_vectors = basis.iloc[leaf_indices]
            # compute all pairwise distances
            pwd = pairwise_distances(leaf_vectors)
            pwd = np.power(pwd, 2)
            pwd = np.sum(pwd)
            node_colors.append(pwd)
    elif color_by == 'MetroidDistances':
        node_colors = []
        # for all nodes
        for i in clu_id:
            # get all leaves
            leaf_indices = [int(nd) for nd in nx.dfs_preorder_nodes(G, source=i) if int(nd) < len(df)]
            # get their AC vectors
            leaf_vectors = basis.iloc[leaf_indices]
            # compute mean
            mean = leaf_vectors.mean(axis=0)
            # compute distances to mean
            dists_to_mean = leaf_vectors.subtract(mean, axis=1)
            MetroidDistances = dists_to_mean.abs().pow(2).sum(axis=0).pow(0.5).mean()
            node_colors.append(4*MetroidDistances)
    elif color_by == 'MeanVariance':
        node_colors = []
        # for all nodes
        for i in clu_id:
            # get all leaves
            leaf_indices = [int(nd) for nd in nx.dfs_preorder_nodes(G, source=i) if int(nd) < len(df)]
            # get their AC vectors
            leaf_vectors = basis.iloc[leaf_indices]
            # compute variance
            var = leaf_vectors.std(axis=1).pow(2).mean()
            node_colors.append(2*var)
    else:
        print('ERROR getColoring with name-code:', color_by)
        
    if color_by == 'Most frequent group' or color_by == 'None':
        mapper = 'color'
    else:
        mapper = linear_cmap(field_name='color', palette=getColormap(cm), low=min(node_colors), high=max(node_colors))
        hint_layout.objects = []
        
    return node_colors, mapper


### Define charts

saved_selections = {}
saved_selection_counter = 0


## AC-Matrix ###
def plot_matrix(df_pred, y_dict, x_dict, y_range, x_range, y='solute'):
    stacked_df = pd.DataFrame(df_pred.stack()).reset_index()
    stacked_df.columns = ['solute', 'solvent', 'ac']
    if y == 'solute':
        stacked_df['ypos'] = stacked_df['solute'].apply(lambda x: y_dict[x])
        stacked_df['xpos'] = stacked_df['solvent'].apply(lambda x: x_dict[x])
    if y == 'solvent':
        stacked_df['ypos'] = stacked_df['solvent'].apply(lambda x: y_dict[x])
        stacked_df['xpos'] = stacked_df['solute'].apply(lambda x: x_dict[x])

    mapper = linear_cmap(field_name='ac', palette=getColormap('CoolWarmDistortedOld'), low=cm_low, high=cm_high)
    
    p = figure(x_range=x_range, y_range=y_range,
               width=500, height=590,
               tools="hover,box_select,tap,reset", toolbar_location=None, #pan,
               tooltips=[('solute', '@solute'), ('solvent', '@solvent'), ('ln(γ)','@ac')])
    p.add_layout(Title(text='Scaled to ln(γ) and in infinite solution', text_font_size="10pt", text_font_style="italic"), 'above')
    p.add_layout(Title(text="Activity coefficients "+select_matrix.value+'\n'), 'above')
    
    p.rect(x="xpos", y="ypos", source=stacked_df, name="rects",
           width=x_range.end/len(df_pred), height=y_range.end/len(df_pred.columns), 
           fill_color=mapper, line_color=None, dilate=True)
    
    color_bar = ColorBar(color_mapper=mapper['transform'])
    color_bar.background_fill_alpha = 0
    p.add_layout(color_bar, 'right')
    p.add_tools(WheelZoomTool(maintain_focus=False))
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.grid.visible = False
    p.xaxis.axis_label = 'Solvents' if y=='solute' else 'Solutes'
    p.yaxis.axis_label = 'Solutes' if y=='solute' else 'Solvents'
    p.axis.axis_label_standoff = 0
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_label_text_font_size = "0px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3
    p.background_fill_color = "#cccccc"

    return p

### Dendrogram ###
def plotClusterTree(G, df, clu_id, clu_x, clu_y, y_range, x_range, color_by='None', horizontal=True, cm='viridis', basis=None):
    p = figure(toolbar_location=None, width=500, height=590, tools='pan, box_select')
    
    # define graph edges
    pxs, pys, ids, edge_colors = [], [], [], []
    for s,e in G.edges():
        xs,ys = [float(v) for v in G.nodes()[s]['pos'].split(',')]
        xe,ye = [float(v) for v in G.nodes()[e]['pos'].split(',')]
#         if G.out_degree(e) == 0:
#             p.text([xe], [ye], text=[str(df.loc[int(e),'Name'])], text_align='right', text_baseline='middle', text_font_size='7pt')

        pxs.append([xs, xs, xe] if horizontal else [xs, xe, xe])
        pys.append([ys, ye, ye] if horizontal else [ys, ys, ye])
        ids.append(e)
        edge_colors.append(select_color)
    
    # define node colors
    node_colors, mapper = getColoring(df, df_legend, G, clu_id, color_by, cm, basis)
    if mapper != 'color':
        color_bar = ColorBar(color_mapper=mapper['transform'], location=(273,420), width=200, title=color_by, orientation='horizontal', margin=0)
        color_bar.background_fill_alpha = 0
        p.add_layout(color_bar)
        
    p.add_layout(Title(text='On '+select_distance.value+' distance with '+select_method.value+' linkage.', text_font_size="10pt", text_font_style="italic"), 'above')
    p.add_layout(Title(text='Hierarchical clustering on '+select_sort.value+'\n'), 'above')
        
    # define ColumnDataSource for tooltip access on nodes
    top5 = pd.DataFrame([], columns=['#'+str(i) for i in range(1,6)])
    try:
        top5_text = [getTopNPrevalence(i, G, df, as_text=True) for i in clu_id]
        for i, val in zip(clu_id, top5_text):
            top5.loc[i] = val
    except KeyError:
        print('Found no groups for coloring & tooltip in this dataframe')
    top5['x'] = clu_x
    top5['y'] = clu_y
    top5['color'] = node_colors
    node_src = ColumnDataSource(top5)

    # define plot rendering
    source = pd.DataFrame({'xs': pxs, 'ys': pys, 'ids': ids, 'color': edge_colors})
    r_tree = p.multi_line(source=source, xs='xs', ys='ys', color='color', name='tree', nonselection_alpha=1, line_width=2)
    r_clu = p.circle(source=node_src, x='x', y='y', color=mapper, size=8, name='clu', nonselection_alpha=1)
    tap = TapTool(renderers=[r_clu])
    hover = HoverTool(tooltips=[('#'+str(i), '@{#'+str(i)+'}') for i in range(1,6)], renderers=[r_clu])
    p.add_tools(tap, hover, WheelZoomTool(maintain_focus=False))
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.grid.visible = False
    p.xaxis.minor_tick_in = 0
    p.xaxis.major_tick_in = 0
    p.axis.axis_label_standoff = 0
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_label_text_font_size = "0px"
    p.axis.major_label_standoff = 0
    p.xaxis.axis_label = 'Cluster dissimilarity →'
    p.outline_line_color = None
    p.x_range = Range1d(0, source['xs'].max()[0], bounds=(0, source['xs'].max()[0])) if horizontal else x_range
    p.y_range = y_range if horizontal else Range1d(source['ys'].max()[0], 0, bounds=(source['ys'].max()[0], 0))
    
    return p

### UNIFAC heatmap ###
def plotUnifacHeatmap(df, scaling_dict, scaled_range):
    stacked_df = pd.DataFrame(df.set_index('Name').stack()).reset_index()
    stacked_df.columns = ['name', 'group', 'count']
    stacked_df = stacked_df[stacked_df['count'] != 0]
    stacked_df['ypos'] = stacked_df['name'].apply(lambda x: scaling_dict[x])

    mapper = linear_cmap(field_name='count', palette=getColormap('Warm'), low=0, high=10)

    p = figure(y_range=scaled_range, x_range=[str(i) for i in df[df[df.columns[1:]].sum(axis=1) != 0]], 
               width=500, height=550, tools='box_select,tap,hover',
               tooltips=[('Name', '@name'), ('UNIFAC main group', '@group'), ('count','@count')],
               toolbar_location=None, title='UNIFAC Main Groups')
    
    color_bar = ColorBar(color_mapper=mapper['transform'])
    color_bar.background_fill_alpha = 0
#     p.add_layout(color_bar, 'right')
    p.rect(x="group", y="ypos", source=stacked_df, width=1, height=1, name='rects',
           selection_fill_color=select_color, nonselection_fill_color=unselect_color, 
           fill_color=select_color, nonselection_alpha=0.7, line_color=None)
    p.background_fill_color = "white"
    p.add_tools(WheelZoomTool(maintain_focus=False))
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.grid.visible = False
    p.yaxis.visible = False
    p.xaxis.major_label_text_font_size = "10px"
    p.xaxis.major_label_standoff = 0
    p.xaxis.minor_tick_in = 0
    p.xaxis.major_tick_in = 0
    p.xaxis.major_label_orientation = -pi / 2
    p.outline_line_color = None
    
    return p

### Data Table ###
def plotTable(df, key):
    columns = [TableColumn(field="Name", title=key+" Name"),
               TableColumn(field="Molecular formula", title="Formula"),
               TableColumn(field="training samples", title="#Samples")]
    return DataTable(source=ColumnDataSource(df[["Name","Molecular formula", "training samples"]]), 
                     view=CDSView(filter=IndexFilter()), columns=columns, 
                     index_position=None, reorderable=False, height=550, width=500)

def plotMDS(df, names, key, metric):
    mds = MDS(random_state=0, dissimilarity='precomputed')
    mds_pos = mds.fit_transform(pairwise_distances(df, metric=metric))
    src = ColumnDataSource(data={'Name':names, 'x':mds_pos[:,0], 'y':mds_pos[:,1], 'color':[select_color]*len(names), 'line_alpha':np.zeros(len(names))})
    p = figure(title='Non-Linear Reduction based on '+select_sort.value.split(' ')[-1]+' ('+select_distance.value+')',
               width=500, height=534, tools='tap,hover', tooltips=[(key, '@{Name}')], toolbar_location=None)
    #p.add_layout(Title(text=' ('+select_distance.value+' distance)', text_font_size="10pt", text_font_style="italic"), 'above')
    p.circle(x="x", y="y", source=src, name='mds', color='color', selection_alpha=1, nonselection_alpha=0.3, size=6, line_color=None) #, line_color=select_color, line_alpha='line_alpha')
    p.background_fill_color = "white"
    p.add_tools(WheelZoomTool(maintain_focus=False), LassoSelectTool(continuous=False))
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.grid.visible = False
    p.axis.visible = False
    p.outline_line_color = None
    
    return p

def violin_plot(df, non_indices, indices, feature, width=250, height=450, style='violin'):
    global saved_selections

    def kde_statsmodels_u(x, grid):
        """Univariate Kernel Density Estimation with Statsmodels"""
        kde = KDEUnivariate(x)
        kde.fit(bw=0.03*(grid[-1]-grid[0]))
        return kde.evaluate(grid)

    def make_kde(df, grid, invert=False):
        kde = kde_statsmodels_u(df, grid)
        #kde = kde / np.amax(kde)
        if invert: 
            kde = -kde
        return kde
    
    def make_histogram(df, grid, invert=False):
        hist, edges = np.histogram(df, range=(grid[0],grid[-1]), bins=15)
        if invert:
            hist = -hist
        return hist, edges
    
    grid = np.linspace(df[feature].min(), df[feature].max(), 100)
    if style=='Histogram':
        make_plot_data = make_histogram
    elif style=='Distribution':
        make_plot_data = make_kde
    else:
        print('ERROR: undefined property plot style. The style was:', style)
    data_sel = make_plot_data(df.loc[indices, feature], grid, invert=True)
    data_nonsel = make_plot_data(df.loc[non_indices, feature], grid)
    draw_data = [(data_sel, area_select_color), (data_nonsel, area_unselect_color)]
    
    # create lines for each instance
    y_jitter = (np.amax(grid) - np.amin(grid)) * 0.003
    line_df = df.loc[:,['Name', feature]]
    line_df.loc[:,'top'] = [i+y_jitter for i in df[feature]]
    line_df.loc[:,'bottom'] = [i-y_jitter for i in df[feature]]
    line_df.loc[:,'left'] = float(0)
    line_df.loc[:,'right'] = float(0)
    line_df.loc[:,'color'] = [line_color]*len(df)
    line_df.loc[:,'alpha'] = [0.35]*len(df)
    
    # add saved selections in color
    #colored_ids = set()
    for selection_id in saved_selections:
        name, id_list = saved_selections[selection_id]
        if len(id_list) == 0 or not isToggled(name):
            continue
        line_df.loc[id_list,'color'] = selection_colors[selection_id]
        if np.array_equal(id_list, indices):
            draw_data[0] = (make_plot_data(df.loc[id_list,feature], grid, invert=True), selection_colors[selection_id])
        else:
            draw_data.append((make_plot_data(df.loc[id_list,feature], grid), selection_colors[selection_id]))
        #colored_ids |= set(id_list)
    if False:
        # remove colored selections from non_selected indices
        non_selected_non_indices = list(set(non_indices) - colored_ids)
        if len(saved_selections) > 0:
            if len(non_selected_non_indices) > 0:
                draw_data[1] = (compute_kde(df.loc[non_selected_non_indices,feature], grid), area_unselect_color)
            else:
                del draw_data[1]
                
    if style == 'Distribution':            
        x_max = 1.05 * max([-np.amin(data_sel)]+[np.amax(data) for data, color in draw_data])
    else:
        x_max = 1.05 * max([-np.amin(data_sel[0])]+[np.amax(data[0]) for data, color in draw_data])
    x_min = -x_max
    line_df.loc[indices,'left'] = -x_max/4
    line_df.loc[non_indices,'right'] = x_max/4
    
    # draw distribution
    p = figure(title=feature, width=width, height=height, tools='', toolbar_location=None, x_range=Range1d(x_min,x_max, bounds=(x_min,x_max)))#, y_range=Range1d(y_min,y_max))
    
    if style=='Distribution':
        for kde, color in draw_data:
            p.harea(x1=np.zeros(len(grid)), x2=kde, y=grid, fill_alpha=0.5, fill_color=color)
    if style=='Histogram':
        for (hist, edges), color in draw_data:
            p.quad(top=edges[1:], bottom=edges[:-1], left=0, right=hist, fill_color=color, line_color=None, alpha=0.5)
        
    
    line_src = ColumnDataSource(line_df)
    r_lin = p.quad(source=line_src, left='left', right='right', top='top', bottom='bottom', alpha='alpha', color='color', line_width=0, name='line', nonselection_alpha='alpha')
    
    # format plot & add tools
    hover = HoverTool(renderers=[r_lin], tooltips=[('', '@Name'+': @{'+feature+'}')])
    p.add_tools(hover, WheelZoomTool(maintain_focus=False), TapTool(renderers=[r_lin]), BoxSelectTool())
    if style=='Distribution' or True: 
        p.xaxis.visible=False
    else:
        p.xaxis.minor_tick_line_color = None
    p.grid.visible=False
    
    p.select('line').data_source.selected.on_change('indices', onViolinSelected(p, 'line'))
    
    return p


# Init global variables
G = clu_id = clu_x = clu_y = p_matrix = p_tree = p_hm = p_table = p_mds = key = df = None

def matrixToListIndices(df, key, matIndices):
    names = set(p_matrix.select('rects').data_source.data[key][matIndices])
    return [int(i) for i in df[df['Name'].isin(names)].index.values], names

def listToMatrixIndices(df, key, listIndices):
    names = df.loc[listIndices]['Name']
    return list(np.nonzero(np.isin(p_matrix.select('rects').data_source.data[key], names))[0]), names

def isToggled(name):
    global toggles
    for t in toggles.objects:
        if type(t) is pn.widgets.Toggle:
            if t.name == name:
                return t.value
    return False

#### Define Interactions
last_matrix_selection = []
last_list_selection = []
last_tap = -1
last_inspect = ''
last_property_style = ''

def updatePlotsToSelection(list_selection=None, matrix_selection=None, skip='', force_update=False, keep_tap=False):
    
    global last_list_selection, last_matrix_selection, df, key, p_matrix, p_tree, p_mds, p_hm, p_table, violins, saved_selections, last_tap
    
    # return on nested call (caused by changes in plot selections of this function)
    if (last_matrix_selection == matrix_selection or last_list_selection == list_selection) and force_update is False:
        return

    # get both selection indices
    if matrix_selection is None:
        matrix_selection, names = listToMatrixIndices(df, key, list_selection)
    if list_selection is None:
        list_selection, names = matrixToListIndices(df, key, matrix_selection)
    last_matrix_selection = matrix_selection
    last_list_selection = list_selection
    if not keep_tap: 
        last_tap = -1

    # update matrix
    if skip != 'Matrix':
        p_matrix.select('rects').data_source.selected.indices = matrix_selection

    # update tree
    if skip != 'Tree':
        # find selected subtree that contains exactly the list_indices
        nx.set_node_attributes(G, 0, "selected")
        nx.set_node_attributes(G, -1, "selection_id")

        for selection_id in saved_selections:
            name, id_list = saved_selections[selection_id]
            if len(id_list) == 0 or not isToggled(name):
                continue
            for i in id_list:
                G.nodes()[str(i)]['selection_id'] = selection_id

        for nd in nx.dfs_postorder_nodes(G, str(2*(len(df)-1))):
            # determine selection
            if G.out_degree(nd) == 0 and (int(nd) in list_selection):
                G.nodes()[nd]['selected'] = 1
            else:
                if sum([G.nodes()[chld]['selected'] for chld in G.adj[nd]]) == 2:
                    G.nodes()[nd]['selected'] = 1
            # determine selection ids
            if G.out_degree(nd) != 0:
                child_sel_ids = [G.nodes()[chld]['selection_id'] for chld in G.adj[nd]]
                if child_sel_ids[0] != child_sel_ids[1] or child_sel_ids[0] == -1:
                    continue
                G.nodes()[nd]['selection_id'] = child_sel_ids[0]

        # color the edges of the subtree
        new_colors = np.empty(len(p_tree.select('tree').data_source.data['color']), dtype='<U7')
        for i, node_id in enumerate(p_tree.select('tree').data_source.data['ids']):
            sel_id = G.nodes()[str(node_id)]['selection_id']
            is_selected = G.nodes()[str(node_id)]['selected']== 1
            if sel_id == -1:
                new_colors[i] = select_color if is_selected else unselect_color
            else:
                new_colors[i] = selection_colors[sel_id] if is_selected else unselection_colors[sel_id]
        p_tree.select('tree').data_source.data['color'] = new_colors

    if skip != 'Heatmap': # TODO maybe this could be faster, what does this even do again?
        df_temp = pd.DataFrame(p_hm.select('rects').data_source.data['name'])
        index_hm = [int(i) for i in df_temp[df_temp[0].isin(names)].index.values]
        p_hm.select('rects').data_source.selected.indices = index_hm

    if skip != 'MDS':
        p_mds.select('mds').data_source.selected.indices = list_selection
    # reset colors
    new_colors = [select_color] * len(df)
    for selection_id in saved_selections:
        name, id_list = saved_selections[selection_id]
        if len(id_list) == 0 or not isToggled(name):
            continue
        for i in id_list:
            new_colors[i] = selection_colors[selection_id]
    p_mds.select('mds').data_source.data['color'] = new_colors

    if skip != 'Table':
        p_table.view.update(filter = IndexFilter(list_selection))

    if skip != 'Violins': # TODO maybe no need for full replot?
        if select_basis.value == 'solutes':
            violins.objects = plot_violins(df, indices=list_selection, features=violin_names+['u1','u2','u3','u4']).objects
        else:
            violins.objects = plot_violins(df, indices=list_selection, features=violin_names+['v1','v2','v3','v4']).objects
        
def updatePlotsToTap(list_index, force_update=False): # list_index as list of int
    global last_tap, p_tree, G, p_mds, df, violins
    
    # return if nothing changed since last tap
    if last_tap == list_index and force_update is False:
        return
    last_tap = list_index
        
    # find path to root of cluster tree
    path = {str(i) for i in list_index}
    for i in list_index:
        path |= set(nx.ancestors(G,str(i)))

    # highlight selected substance's path in tree
    new_colors = [highlight_color if i in path
#                   else p_tree.select('tree').data_source.data['color'][np.where(p_tree.select('tree').data_source.data['ids'] == int(i))]
                  else select_color if G.nodes()[str(i)]['selected'] == 1 
                  else unselect_color
                  for i in p_tree.select('tree').data_source.data['ids']]
    p_tree.select('tree').data_source.data['color'] = new_colors
    
    new_colors = [select_color] * len(df)
    for i in list_index:
        new_colors[i] = highlight_color
    p_mds.select('mds').data_source.data['color'] = new_colors
    
    # update violins
    for obj in violins.objects:
        if type(obj) is pn.pane.plot.Bokeh:
            data = obj.object.select('line').data_source.data
            if type(data['index']) is list:
                data_index = [data['index'].index(i) for i in list_index]
            else:
                data_index = np.where(np.isin(data['index'], list_index))[0]
            if len(data_index) < 1:
                print('ERROR: Could not find tapped point in violin selection. Found occurences:', len(data_index))
                print('index', data['index'], '/nsearch', list_index)
                return
                
            # highlight point in color, alpha, length
            new_colors = [line_color] * len(data['color'])
            new_alphas = [0.3] * len(data['alpha'])
            new_left = np.clip(data['left'], obj.object.x_range.start/4, 0)
            new_right = np.clip(data['right'], 0, obj.object.x_range.end/4)
            for i in data_index:
                new_colors[i] = highlight_color # TODO account for list of data_index
                new_alphas[i] = 1
                new_left[i] = new_left[i] * 3
                new_right[i] = new_right[i] * 3
            data['color'] = new_colors
            data['alpha'] = new_alphas
            data['left'] = new_left
            data['right'] = new_right
    
### AC-Matrix Interactivity ###
def onMatrixSelected(attr, old, new):
    updatePlotsToSelection(matrix_selection=new, skip='Matrix')
        
### Dendrogram Interactivity ###
def onTreeNodeSelected(attr, old, new):
    # transform dendrogram_index to df_index
    n = len([x for x in G.nodes() if G.out_degree(x)==0])
    if len(new) > 0:
        df_index = []
        for node in new:
            df_index = df_index + [int(nd) for nd in nx.dfs_preorder_nodes(G, source=clu_id[node]) if int(nd) < n]
    else:
        df_index = list(range(n))
    
    updatePlotsToSelection(list_selection=df_index)
    
def onTableSelected(attr, old, new):
    if len(new) >= 1:
        updatePlotsToTap(new)

def onMDSSelected(attr, old, new):
    if len(new) == 1:
        updatePlotsToTap(new)
    else:
        updatePlotsToSelection(list_selection=new, skip='MDS')
        
def onViolinSelected(violin, key):
    def f(attr, old, new):
        if len(new) == 1:
            updatePlotsToTap(new)
        else:
            updatePlotsToSelection(list_selection=new)
    return f
    
def onButtonSaveSelection(event):
    addToggle(last_list_selection, text_input.value)
    updatePlotsToSelection(list_selection=last_list_selection, force_update=True)
    
### Create Interface
def layout(inspect, mat, sort, distance, method, color_by, cm, legend, property_style):
    global G, clu_id, clu_x, clu_y, p_matrix, p_tree, p_hm, p_table, p_mds, last_list_selection, last_tap, last_inspect, last_property_style, df_legend, key, df, select_coloring
    
    # load legend
    df_legend = loadLegend(legend, key=inspect)
    select_coloring.options=select_coloring.options[:len(coloring_options)]+df_legend['Beschreibung'].tolist()
    
    # load shown matrix entries
    if mat == 'best computation':
        entries = WHI_pred
    if mat == 'predicted with MCM':
        entries = MCM_pred
    if mat == 'computed with UNIFAC':
        entries = UNI_pred
    if mat == 'from Experiments':
        entries = TRU_pred
    
    # load sorting basis
    if sort == 'best AC':
        basis_solu, basis_solv = WHI_pred, WHI_pred.T      
    if sort == 'MCM-AC':
        basis_solu, basis_solv = MCM_pred, MCM_pred.T        
    if sort == 'UNIFAC-AC':
        UNI_pred_wo_NaN = UNI_pred.fillna(0)
        basis_solu, basis_solv = UNI_pred_wo_NaN, UNI_pred_wo_NaN.T
    if sort == 'UNIFAC-Groups':
        basis_solu, basis_solv = df_solu[unifac_names], df_solv[unifac_names]
    if sort == 'Physical Props':
        basis_solu = df_solu[physical_prop_names]
        if mat == 'best computation':
            basis_solv = WHI_pred.T
        else:
            basis_solv = MCM_pred.T
        
    # create cluster tree
    solu_tree_model, solv_tree_model, solute_tree_indices, solvent_tree_indices = cluster(basis_solu, basis_solv, distance, method)
    solute_tree_indices = np.argsort(solute_tree_indices)
    solvent_tree_indices = np.argsort(solvent_tree_indices)
    
    # load optimal ordering if precomputed
    try:
        solute_tree_indices = np.load('./data/sorted_indices_'+sort+'_'+distance+'_'+method+'_solu.npy')
        solvent_tree_indices = np.load('./data/sorted_indices_'+sort+'_'+distance+'_'+method+'_solv.npy')
    except OSError:
        print('No file for optimal leaf ordering (OLO) found. Regular Hierarchical Clustering is used. Use OLO for a prettier tree.')
        
    if inspect == 'solutes':
        df = df_solu
        key = 'solute'
        basis = basis_solu
        G, _, clu_id, clu_x, clu_y = createClusterTree(solu_tree_model, order=solute_tree_indices)
    else:
        df = df_solv
        key = 'solvent'
        basis = basis_solv
        G, _, clu_id, clu_x, clu_y = createClusterTree(solv_tree_model, order=solvent_tree_indices)
    
    # create axes
    max_tree_pos_y = max([float(p.split(',')[1]) for i,p in G.nodes(data='pos')])+0.5
    max_tree_pos_x = 1 # max([float(p.split(',')[0]) for i,p in G_solv.nodes(data='pos')])+0.5
    y_range = Range1d(0, max_tree_pos_y, bounds=(0, max_tree_pos_y))
    x_range = Range1d(0, max_tree_pos_x, bounds=(0, max_tree_pos_x))
    solute_names_tree_sorted = [df_solu['Name'].iloc[int(i)] for i in solute_tree_indices][::-1]
    solvent_names_tree_sorted = [df_solv['Name'].iloc[int(i)] for i in solvent_tree_indices][::-1]
    y_names_tree_sorted = solute_names_tree_sorted if inspect=='solutes' else solvent_names_tree_sorted
    x_names_tree_sorted = solvent_names_tree_sorted if inspect=='solutes' else solute_names_tree_sorted
    y_dict = {name: (i+0.5) / len(y_names_tree_sorted) * max_tree_pos_y for i, name in enumerate(y_names_tree_sorted)}
    x_dict = {name: (i+0.5) / len(x_names_tree_sorted) * max_tree_pos_x for i, name in enumerate(x_names_tree_sorted)}
    
    # create plots
    p_matrix = plot_matrix(entries, y_dict, x_dict, y_range, x_range, key)
    p_tree = plotClusterTree(G, df, clu_id, clu_x, clu_y, y_range, x_range, color_by=color_by, cm=cm, basis=basis)
    p_table = plotTable(df, 'Solute' if inspect=='solutes' else 'Solvent')
    p_hm = plotUnifacHeatmap(df_solu[['Name']+unifac_names] if inspect=='solutes' else df_solv[['Name']+unifac_names], y_dict, y_range)
    p_mds = plotMDS(basis, df['Name'].tolist(), 'Solute' if inspect=='solutes' else 'Solvent', distance)
    
    # create interactions
    p_matrix.select('rects').data_source.selected.on_change('indices', onMatrixSelected)
    p_tree.select('clu').data_source.selected.on_change('indices', onTreeNodeSelected)
    p_table.source.selected.on_change('indices', onTableSelected)
    p_mds.select('mds').data_source.selected.on_change('indices', onMDSSelected)
    
    # update violins if necessary
    if last_inspect != inspect or last_property_style != property_style:
        if last_inspect != inspect:
            last_list_selection = []
            last_tap = -1
        if inspect=='solutes':
            violins.objects = plot_violins(df, features=violin_names+['u1','u2','u3','u4']).objects
        else:
            violins.objects = plot_violins(df, features=violin_names+['v1','v2','v3','v4']).objects
    last_inspect = inspect
    last_property_style = property_style
    # keep selection and tap-highlight
    if last_list_selection != []:
        updatePlotsToSelection(list_selection=last_list_selection, force_update=True, keep_tap=True)
    if last_tap != -1:
        updatePlotsToTap(last_tap, force_update=True)
    
    gp = pn.Row(p_matrix, p_tree, pn.Tabs(('Substance Table', p_table), ('Non-Linear Reduction', p_mds), ('UNIFAC-Groups', p_hm)))
    
    return gp

def plot_violins(df, indices=[], features=violin_names+['u1','u2','u3','u4'], height=300, width=1500):
    # left spacer for alignment
    violins = pn.Row(pn.Spacer(width=0))
    if df is None: return violins
    # divide into indices
    not_indices = df.index.values
    if len(indices) == 0:
        indices = df.index.values
    elif len(indices) != len(df.index.values):
        not_indices = np.setdiff1d(not_indices, indices)
        
    for f in features:
        if f in df.columns:
            violins.append(violin_plot(df, not_indices, indices, f, width=width//len(features), height=height, style=property_style.value))
        
    return violins

violins = plot_violins(df)

def addToggle(selection, name):
    if selection == []: return
    global saved_selection_counter
    if len(name) == 0:
        name = 'Selection '+str(saved_selection_counter)
    saved_selections[saved_selection_counter] = (name, selection)
    if saved_selection_counter%4 == 0:
        toggle_type = 'primary'
    if saved_selection_counter%4 == 1:
        toggle_type = 'warning'
    if saved_selection_counter%4 == 2:
        toggle_type = 'success'
    if saved_selection_counter%4 == 3:
        toggle_type = 'danger'
    def click_toggle(event):
        updatePlotsToSelection(list_selection=last_list_selection, force_update=True)
    def click_toggle_delete(save_pos):
        def f(event):
            saved_selections[save_pos] = ('', [])
            k = 0
            for i in range(len(saved_selections)):
                if i < save_pos and len(saved_selections[i][1]) > 0:
                    k += 1
            toggles.pop(k*2)
            toggles.pop(k*2)
            updatePlotsToSelection(list_selection=last_list_selection, force_update=True)
        return f
    toggle = pn.widgets.Toggle(name=name, button_type=toggle_type, value=True, width=100)
    toggle.param.watch(click_toggle, 'value')
    toggle_delete = pn.widgets.Button(name='X', button_type=toggle_type, width=35)
    toggle_delete.on_click(click_toggle_delete(saved_selection_counter))
    toggles.append(toggle)
    toggles.append(toggle_delete)
    saved_selection_counter += 1
    
toggles = pn.Row()
hint_layout = pn.Row()

select_basis = pn.widgets.Select(name='Focus of clustering', value='solutes', options=['solutes', 'solvents'], sizing_mode="stretch_width")
select_matrix = pn.widgets.Select(name='Shown Matrix Entries', value='predicted with MCM', options=['best computation','predicted with MCM', 'computed with UNIFAC', 'from Experiments'], sizing_mode="stretch_width")
select_sort = pn.widgets.Select(name='Sorting Basis', value='MCM-AC', options=['best AC', 'MCM-AC', 'UNIFAC-AC', 'UNIFAC-Groups', 'Physical Props'], sizing_mode="stretch_width")
select_distance = pn.widgets.Select(name='Distance Metric', value='euclidean', options=['euclidean', 'cosine', 'manhattan'], sizing_mode="stretch_width")
select_method = pn.widgets.Select(name='Linkage Method', value='complete', options=['ward', 'complete', 'average', 'single'], sizing_mode="stretch_width")
select_coloring = pn.widgets.Select(name='Tree Color Measure', value='Entropy', options=coloring_options, sizing_mode="stretch_width")
select_cm = pn.widgets.Select(name='Colormap', value='Warm', options=['viridis', 'RedYellowGreen', 'rainbow', 'summer', 'winter', 'cool', 'CoolWarmDistorted', 'CoolWarm', 'Warm', 'greyscale', 'InverseGreyscale'], sizing_mode="stretch_width")
select_legend = pn.widgets.Select(name='Chemical Categories', value='Single Label', options=['Single Label', 'Multiple Labels'], sizing_mode="stretch_width")
property_style = pn.widgets.Select(name='Plot Style', value='Distribution', options=['Distribution','Histogram'], sizing_mode="stretch_width")
reactive_layout = pn.bind(layout, inspect=select_basis, mat=select_matrix, sort=select_sort, distance=select_distance, method=select_method, color_by=select_coloring, cm=select_cm, legend=select_legend, property_style=property_style)
text_input = pn.widgets.TextInput(name='Selection Name', placeholder='Selection X', sizing_mode="stretch_width")
button_save = pn.widgets.Button(name='Save current selection', button_type='default', sizing_mode="stretch_width")
button_save.on_click(onButtonSaveSelection)

controls = pn.Column(pn.pane.Markdown("### Data Settings", sizing_mode="stretch_width", height=35), select_basis, select_matrix, 
                     pn.pane.Markdown("### Cluster Settings", sizing_mode="stretch_width", height=35), select_sort, select_distance, select_method, 
                     pn.pane.Markdown("### Tree Settings", sizing_mode="stretch_width", height=35), select_legend, select_coloring, select_cm, 
                     pn.pane.Markdown("### Property Settings", sizing_mode="stretch_width", height=35), property_style,
                     pn.pane.Markdown("### Save Selection", sizing_mode="stretch_width", height=35), text_input, button_save, sizing_mode="stretch_width")

main_layout = pn.Column(reactive_layout, toggles, violins, hint_layout)

### start app ###
template = pn.template.FastListTemplate(
            site="EnrichMatrix", 
            title="Exploring Acitivity Coefficients", 
            sidebar=[controls], 
            main=[main_layout],
            header_background=highlight_color,
            accent_base_color=highlight_color,
        )
app = template.servable()
