#!/usr/bin/env python
# coding: utf-8

# # ML for Thermodynamics

from statistics import stdev, mean

from bokeh.layouts import row

from panel.interact import fixed

import pandas as pd
import numpy as np

import panel as pn
import param

from scripts.gamma_matrix import gamma_matrix
from scripts.error_distribution import Error_distribution_tab
from scripts.neighborhood import neighbor_matrix
from scripts.group import Group_tab
from scripts.classes import Classes
from scripts.heatmap import Heatmap

pn.extension()

# Load MCM data trained on the full set
solutes_df = pd.read_excel(
    "./Data/Data_SD.xlsx", sheet_name="Solutes", engine="openpyxl"
)
solvents_df = pd.read_excel(
    "./Data/Data_SD.xlsx", sheet_name="Solvents", engine="openpyxl"
)

# cast to correct data types
solutes_df = solutes_df.replace(to_replace=",", value=".", regex=True).infer_objects()
solvents_df = solvents_df.replace(to_replace=",", value=".", regex=True).infer_objects()
solutes_df[["u1", "u2", "u3", "u4"]] = solutes_df[["u1", "u2", "u3", "u4"]].apply(
    pd.to_numeric
)
solvents_df[["v1", "v2", "v3", "v4"]] = solvents_df[["v1", "v2", "v3", "v4"]].apply(
    pd.to_numeric
)
solutes_df["Name"] = solutes_df["Name"].astype("str")
solvents_df["Name"] = solvents_df["Name"].astype("str")
whole_df = pd.merge(solutes_df, solvents_df, how="outer")

# load data MCM data trained on a subset
### TODO create u & v for trained on subset

# read physical properties
properties = pd.read_excel(
    "./Data/Physical_Properties.xlsx",
    sheet_name="Solutes",
    index_col=0,
    engine="openpyxl",
)
tmp_proper = pd.read_excel(
    "./Data/Physical_Properties.xlsx",
    sheet_name="Solvents",
    index_col=0,
    engine="openpyxl",
)
properties = (
    pd.concat([properties, tmp_proper], axis=0)
    .drop_duplicates()
    .sort_values(by="DDB No")
)
properties = properties.replace(to_replace=",", value=".", regex=True)
properties.drop(columns=["Name"], inplace=True)
# properties_infered = properties.fillna(0).replace('n.a.', 0).infer_objects()

# whole_df_infered_phys = whole_df.merge(properties_infered, how='left')
whole_df = whole_df.merge(properties, how="left", left_on="DDB ID", right_on="DDB No")
whole_df = whole_df.drop(columns=["Solute ID"]).replace("n.a.", pd.NA)

# Load new descriptor data-set and add it to the above data
descriptor_df = pd.read_excel("./Data/20201109_Descriptors.xlsx", engine="openpyxl")
descriptor_df.columns = [
    "DDB ID",
    "Name",
    "Molar Mass",
    "Relative Polarity",
    "H-Bond Acceptors",
    "H-Bond Donors",
    "Root 1",
    "Root 1 ID",
    "Root 2",
    "Root 2 ID",
    "Root 3",
    "Root 3 ID",
    "Root 4",
    "Root 4 ID",
    "Root 5",
    "Root 5 ID",
]

root_data = []
for _, row in descriptor_df.iterrows():
    roots = []
    i = 1
    while i < 5 and not pd.isna(row["Root " + str(i)]):
        roots.append(row["Root " + str(i)])
        i += 1
    root_data.append(tuple(roots))

descriptor_df["Roots"] = root_data
descriptor_df.drop(
    columns=[
        "Name",
        "Root 1",
        "Root 1 ID",
        "Root 2",
        "Root 2 ID",
        "Root 3",
        "Root 3 ID",
        "Root 4",
        "Root 4 ID",
        "Root 5",
        "Root 5 ID",
    ],
    inplace=True,
)

# define the subsets of variables to use for SPLOMs
variablesA = ["u1", "u2", "u3", "u4"]
variablesB = ["v1", "v2", "v3", "v4"]
variablesC = ["Mol.Weight", "Tc [K]", "Pc [kPa]", "Ac.Factor"]
variablesD = ["SD u1", "SD u2", "SD u3", "SD u4"]
variablesE = ["SD v1", "SD v2", "SD v3", "SD v4"]
variablesF = ["Molar Mass", "Relative Polarity", "H-Bond Acceptors", "H-Bond Donors"]
variables = variablesA + variablesB + variablesC + variablesD + variablesE + variablesF

############################################################
### load Fabian's gammas (expected and predicted with MCM)
gamma_whole = pd.read_excel(
    "./Data/Data_SD.xlsx", sheet_name="gamma", engine="openpyxl"
)
gamma_whole = gamma_whole.replace(to_replace=",", value=".", regex=True)
gamma_whole.rename(
    columns={
        "ln(gamma_ij) exp": "ln(gamma_ij) Experimental",
        "ln(gamma_ij) MCM": "ln(gamma_ij) MCM",
        "DDB No. Solute i": "Solute",
        "DDB No. Solvent j": "Solvent",
    },
    inplace=True,
)
gamma_whole[["T / K", "ln(gamma_ij) MCM", "ln(gamma_ij) Experimental"]] = gamma_whole[
    ["T / K", "ln(gamma_ij) MCM", "ln(gamma_ij) Experimental"]
].apply(pd.to_numeric)
gamma_whole["SoluteName"] = pd.merge(
    gamma_whole, solutes_df, left_on="Solute", right_on="DDB ID", how="left"
)["Name"]
gamma_whole["SolventName"] = pd.merge(
    gamma_whole, solvents_df, left_on="Solvent", right_on="DDB ID", how="left"
)["Name"]

# read prediction gammas
gamma_subset = pd.read_excel(
    "./Data/Predictions.xlsx", usecols=range(5), engine="openpyxl"
)
gamma_subset.rename(
    columns={
        "MCM": "ln(gamma_ij) MCM",
        "Exp": "ln(gamma_ij) Experimental",
        "DDB Solute": "Solute",
        "DDB Solvent": "Solvent",
    },
    inplace=True,
)
gamma_subset["SoluteName"] = pd.merge(
    gamma_subset, solutes_df, left_on="Solute", right_on="DDB ID", how="left"
)["Name"]
gamma_subset["SolventName"] = pd.merge(
    gamma_subset,
    solvents_df[["DDB ID", "Name"]],
    left_on="Solvent",
    right_on="DDB ID",
    how="left",
)["Name"]

# Calculate errors for both gamma-dataframes
for df in [gamma_whole, gamma_subset]:
    df["gamma_exp"] = np.exp(df["ln(gamma_ij) Experimental"])
    df["gamma_MCM"] = np.exp(df["ln(gamma_ij) MCM"])
    df["Error_ln(gamma)"] = df["ln(gamma_ij) MCM"] - df["ln(gamma_ij) Experimental"]
    df["Error_gamma"] = df["gamma_MCM"] - df["gamma_exp"]
    df["ln(Error_gamma)"] = np.log(abs(df["Error_gamma"]))
    df["abs(ln(Error_gamma))"] = abs(df["ln(Error_gamma)"])
    df["Error Sign"] = df["Error_gamma"] / abs(df["Error_gamma"])
    df["MCM / Experimental"] = df["gamma_MCM"] / df["gamma_exp"]
    df["MCM / Experimental Modified"] = df["Error Sign"] * (
        (df["MCM / Experimental"] ** df["Error Sign"]) - 1
    )
    df["ln(MCM / Experimental)"] = np.log(df["MCM / Experimental"])

    for x in ["Solute", "Solvent"]:
        counts = df[x].value_counts().to_frame()
        counts.rename(columns={x: x + " count"}, inplace=True)
        df[x + " count"] = pd.merge(
            df, counts, left_on=x, right_index=True, how="left"
        )[x + " count"]

#####################################
### load DDBST KEEN-Activity-Data ###
#####################################
ddbst = pd.read_excel(
    "./Data/DDBST KEEN-Activity-Data Version I.xlsx",
    engine="openpyxl",
    usecols=range(7),
)
ddbst.columns = [
    "SolventClass",
    "SolventName",
    "Solvent",
    "SoluteClass",
    "SoluteName",
    "Solute",
    "Mean_Value",
]

solvent_groups = (
    ddbst[["Solvent", "SolventClass"]]
    .rename(columns={"Solvent": "DDB ID", "SolventClass": "Class"})
    .drop_duplicates(subset="DDB ID")
    .set_index("DDB ID")
)
solute_groups = (
    ddbst[["Solute", "SoluteClass"]]
    .rename(columns={"Solute": "DDB ID", "SoluteClass": "Class"})
    .drop_duplicates(subset="DDB ID")
    .set_index("DDB ID")
)

groups = pd.concat([solvent_groups, solute_groups])
groups = groups[~groups.index.duplicated(keep="first")].sort_index().reset_index()

## add the classes to the other dataframes
# add classes and roots to whole_df
whole_df["Class"] = pd.merge(whole_df, groups, on="DDB ID", how="left")[
    "Class"
].str.strip()
whole_df = whole_df.merge(descriptor_df, how="left")

# add classes and roots to the gamma_df's
solvent_classes = ddbst[["Solvent", "SolventClass"]].drop_duplicates(subset="Solvent")
solute_classes = ddbst[["Solute", "SoluteClass"]].drop_duplicates(subset="Solute")

for df in [gamma_subset, gamma_whole]:
    df["SolventClass"] = pd.merge(df, solvent_classes, on="Solvent", how="left")[
        "SolventClass"
    ].str.strip()
    df["SoluteClass"] = pd.merge(df, solute_classes, on="Solute", how="left")[
        "SoluteClass"
    ].str.strip()
    df["SolventRoots"] = pd.merge(
        df, descriptor_df, left_on="Solvent", right_on="DDB ID", how="left"
    )["Roots"]
    df["SoluteRoots"] = pd.merge(
        df, descriptor_df, left_on="Solute", right_on="DDB ID", how="left"
    )["Roots"]

#####
## create data frame over individual values for each top class
## there is an entry for each root of each chemical with the chemical's corresponding values
columnsA = ["u1", "u2", "u3", "u4", "v1", "v2", "v3", "v4"]
root_data = {"root_name": [], "substance_name": []}
for column in columnsA:
    root_data[column] = []
for _, row in whole_df.iterrows():
    roots = row["Roots"]
    if pd.isna(roots):
        continue
    for root in row["Roots"]:
        for column in columnsA:
            root_data[column].append(row[column])
        root_data["root_name"].append(root)
        root_data["substance_name"].append(row["Name"])
root_df = pd.DataFrame.from_dict(root_data)

#####
## create data frame over prediction error for the combinations of top classes
## there is an entry for each root of each component (solvent/solute) of the solutions
root_gamma_data = {
    "root_name": [],
    "used_as": [],
    "solvent_name": [],
    "solute_name": [],
}
columnsB = [
    "Error_gamma",
    "ln(Error_gamma)",
    "ln(gamma_ij) MCM",
    "ln(gamma_ij) Experimental",
]
for column in columnsB:
    root_gamma_data[column] = []
for index, row in gamma_whole.iterrows():
    for var in ["Solvent", "Solute"]:
        roots = row[var + "Roots"]
        if pd.isna(roots):
            continue
        for root in roots:
            for column in columnsB:
                root_gamma_data[column].append(row[column])
            root_gamma_data["root_name"].append(root)
            root_gamma_data["solvent_name"].append(row["SolventName"])
            root_gamma_data["solute_name"].append(row["SoluteName"])
            root_gamma_data["used_as"].append(var)

root_gamma_df = pd.DataFrame.from_dict(root_gamma_data)

#####
## create data frame over mean scores for each combination of top class
## contains an entry for each pair of root classes that were combined at least once into a solution
root_means_data = {}
columnsMeans = [
    "Error_gamma",
    "ln(Error_gamma)",
    "abs(ln(Error_gamma))",
    "ln(gamma_ij) MCM",
    "ln(gamma_ij) Experimental",
]
for index, row in gamma_whole.iterrows():
    solute_roots = row["SoluteRoots"]
    solvent_roots = row["SolventRoots"]
    # skip solvents/solutes that are not in the hierarchy
    if pd.isna(solute_roots) or pd.isna(solvent_roots):
        continue
    for solute in solute_roots:
        for solvent in solvent_roots:
            entry = root_means_data.get((solute, solvent), {})
            for column in columnsMeans:
                values = entry.get(column, [])
                values.append(row[column])
                entry[column] = values
            root_means_data[(solute, solvent)] = entry

root_means_data_processed = {"Solvent": [], "Solute": []}
for column in columnsMeans:
    root_means_data_processed[column] = [
        mean(root_means_data[pair][column]) for pair in root_means_data
    ]
    root_means_data_processed[column + "_stdev"] = [
        stdev(root_means_data[pair][column])
        if len(root_means_data[pair][column]) > 1
        else 0
        for pair in root_means_data
    ]
root_means_data_processed["Solute"] = [pair[0] for pair in root_means_data]
root_means_data_processed["Solvent"] = [pair[1] for pair in root_means_data]
root_means_data_processed["count"] = [
    len(root_means_data[pair][columnsMeans[0]]) for pair in root_means_data
]

root_means_df = pd.DataFrame.from_dict(root_means_data_processed)

### TAB: chemical classes
class_tab = Classes(root_df, root_gamma_df, columnsA, columnsB, ["root_name"]).show()

### TAB: heatmap of chemical classes
heatmap_tab = Heatmap(root_means_df, "Solvent", "Solute", columnsMeans).view()


### TAB: matrix of known entries
class matrix(param.Parameterized):
    selected_data = param.ObjectSelector(
        default="Trained on Subset",
        objects=["Trained on Subset", "Trained on Entire Dataset"],
    )
    df = {"Trained on Subset": gamma_subset, "Trained on Entire Dataset": gamma_whole}

    def view(self):
        return gamma_matrix(self.df[self.selected_data])


matrix_obj = matrix()
matrix_tab = pn.Row(matrix_obj.view, matrix_obj.param)

### TAB: gamma distribution
error_distribution_tab = Error_distribution_tab(gamma_subset, gamma_whole).view()

### TAB: neighborhood
kw = dict(
    xs=[variablesA, variablesB, variablesC],
    ys=[variablesA, variablesB, variablesC],
    nbh=[variablesA, variablesB, variablesC],
    neighbors=(1, 30),
    select=[""] + whole_df["Name"].tolist(),
    size=(50, 300),
    alpha=(0.1, 1),
)
i = pn.interact(neighbor_matrix, dataset=fixed(whole_df), **kw)
neighb_tab = pn.Column(
    pn.Row(
        pn.Column(i[0][0], i[0][2]),
        pn.Column(i[0][1], i[0][3]),
        pn.Column(i[0][4], i[0][5]),
    ),
    pn.Column(i[1][0]),
)

### TAB: group & dim red
subset_df = whole_df
group_tab = Group_tab(
    whole_df,
    subset_df,
    gamma_whole,
    gamma_subset,
    root_df,
    root_gamma_df,
    [variablesA, variablesB, variablesC, variablesF],
).view()

### The resulting page
tabs = pn.Tabs(
    ("Gamma Matrix", matrix_tab),
    ("Group", group_tab),
    ("Classes", class_tab),
    ("Heatmap", heatmap_tab),
    ("Neigbhorhood", neighb_tab),
).servable()
tabs.show(port=5006)
