import os
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import pandas as pd
from scipy import stats
import scipy.stats as st
import plotly.graph_objects as go
import ast

data_cuts = [0, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3]
# data_cuts = [1]
cutlis = []
varvals = {"Name": [], "Peripheral": [], "Midzone": [], "Stadard Dev": []}
for ct in data_cuts:
    output_data = {
        "Sample name": [],
        "Total fission events:": [],
        "Number of midzone fission events:": [],
        "Number of peripheral fission events:": [],
        "% midzone": [],
        "% peripheral": [],
    }
    for a in os.listdir("out"):

        output_data["Sample name"].append(a)
        filoc = "out/{}".format(a)
        for file in os.listdir(filoc):
            if "fission" in file:
                fission_events = []
                path = os.path.join(filoc, file)
                with open(path) as f:
                    read = csv.reader(f)
                    for track_number, line in enumerate(read):
                        for frame_number, track_split_from in enumerate(line):
                            if track_split_from not in list(["0", "NaN"]):
                                print(track_split_from)
                                fission_events.append(
                                    [
                                        track_number,
                                        frame_number,
                                        (int(track_split_from) - 1),
                                    ]
                                )
            print("done")
        for file in os.listdir(filoc):
            if "Area" in file:
                fission_results = []
                path = os.path.join(filoc, file)
                with open(path) as f:
                    listofvals = []

                    read = csv.reader(f)
                    for lin in read:
                        listofvals.append(lin)
                    for values in fission_events:
                        for track_number, line in enumerate(listofvals):
                            if track_number == values[0]:
                                for frame_number, volume in enumerate(line):
                                    if frame_number == values[1]:
                                        split_volume_after_fission = volume
                                        # read2 = csv.reader(f)
                                        for track_number_2, line_2 in enumerate(
                                            listofvals
                                        ):
                                            if track_number_2 == (values[2]):
                                                nxt = False
                                                for (
                                                    frame_number_2,
                                                    volume_2,
                                                ) in enumerate(line_2):
                                                    if frame_number_2 == (
                                                        values[1] - 1
                                                    ):
                                                        source_volume_before_fission = (
                                                            volume_2
                                                        )
                                                        if (
                                                            source_volume_before_fission
                                                            == "NaN"
                                                        ):
                                                            print("nan")
                                                        nxt = True
                                                    elif nxt == True:
                                                        source_volume_after_fission = (
                                                            volume_2
                                                        )
                                                        nxt = False
                                                        if (
                                                            source_volume_after_fission
                                                            == "NaN"
                                                        ):
                                                            if (
                                                                source_volume_before_fission
                                                                == "NaN"
                                                            ):
                                                                continue
                                                            else:
                                                                continue
                                                                source_volume_after_fission = abs(
                                                                    float(
                                                                        source_volume_before_fission
                                                                    )
                                                                    - float(
                                                                        split_volume_after_fission
                                                                    )
                                                                )

                                                            print("nan")
                                                        if (
                                                            float(
                                                                source_volume_after_fission
                                                            )
                                                            > ct
                                                        ):
                                                            if (
                                                                float(
                                                                    split_volume_after_fission
                                                                )
                                                                > ct
                                                            ):
                                                                if (
                                                                    float(
                                                                        split_volume_after_fission
                                                                    )
                                                                    / float(
                                                                        source_volume_after_fission
                                                                    )
                                                                ) * 100 > 100:
                                                                    fission_results.append(
                                                                        (
                                                                            float(
                                                                                source_volume_after_fission
                                                                            )
                                                                            / float(
                                                                                split_volume_after_fission
                                                                            )
                                                                        )
                                                                        * 100
                                                                    )
                                                                    print("err")
                                                                elif (
                                                                    float(
                                                                        split_volume_after_fission
                                                                    )
                                                                    / float(
                                                                        source_volume_after_fission
                                                                    )
                                                                ) * 100 < 0:
                                                                    print("err")
                                                                else:
                                                                    fission_results.append(
                                                                        (
                                                                            float(
                                                                                split_volume_after_fission
                                                                            )
                                                                            / float(
                                                                                source_volume_after_fission
                                                                            )
                                                                        )
                                                                        * 100
                                                                    )
                                                                print("stop")

                    print(fission_results)
                    fission_results_fixed = []
                    for val in fission_results:
                        if val > 50:
                            val = abs(100 - val)
                        fission_results_fixed.append(val)
                    print(fission_results_fixed)
                    peri = 0
                    perivals = []
                    midzone = 0
                    midzonevals = []
                    for valuev in fission_results_fixed:

                        if valuev >= 25:
                            midzone += 1
                            midzonevals.append(valuev)
                        if valuev < 25:
                            peri += 1
                            perivals.append(valuev)
                    output_data["Total fission events:"].append(
                        len(fission_results_fixed)
                    )  # = {"Total fission events:":len(fission_results_fixed),"Number of midzone fission events:":midzone,"Number of peripheral fission events:":peri, "% midzone":(midzone/len(fission_results_fixed))*100,"% peripheral":(peri/len(fission_results_fixed))*100}
                    output_data["Number of midzone fission events:"].append(midzone)
                    output_data["Number of peripheral fission events:"].append(peri)
                    if len(fission_results_fixed) != 0:
                        output_data["% midzone"].append(
                            (midzone / len(fission_results_fixed)) * 100
                        )
                    else:
                        output_data["% midzone"].append(0)
                    if len(fission_results_fixed) != 0:
                        output_data["% peripheral"].append(
                            (peri / len(fission_results_fixed)) * 100
                        )
                    else:
                        output_data["% peripheral"].append(0)
                    varvals["Name"].append(filoc.rsplit("/", 1)[1])
                    varvals["Midzone"].append(len(midzonevals))
                    varvals["Peripheral"].append(len(perivals))
                    varvals["Stadard Dev"].append(midzonevals + perivals)
    cutlis.append({ct: output_data})
if os.path.exists("results.csv"):
    os.remove("results.csv")
rowmk = ["1"]
with open("results.csv", "x", newline="") as outfil:
    fields = list(output_data.keys())
    wr = csv.writer(outfil)
    wr.writerow(fields)

    for e, curvl in enumerate(cutlis):
        wr.writerow(["Threshold Value:", list(curvl.keys())[0]])
        items = np.array(list(curvl[list(curvl.keys())[0]].values()))
        liap = []
        for row in items.T:
            rowr = []
            for enr, it in enumerate(row):
                try:
                    rowr.append(float(it))
                except Exception as e:
                    rowr.append(it)
                    pass
            if rowmk[0].rsplit(" ", 1)[0] != rowr[0].rsplit(" ", 1)[0]:
                rowmk = rowr
            elif rowmk[0].rsplit(" ", 1)[0] == rowr[0].rsplit(" ", 1)[0]:
                rowl = []
                rowl.append(rowmk[0].rsplit(" ", 1)[0])
                rowl.append(rowr[1] + rowmk[1])
                rowl.append((rowr[5] + rowmk[5]) / 2)
                print("k")
                liap.append(rowl)
            if rowr[0].rsplit(" ", 1)[0] == "NORM FIS1":
                rowl = []
                rowl.append(rowr[0].rsplit(" ", 1)[0])
                rowl.append(rowr[1])
                rowl.append(rowr[5])
                liap.append(rowl)
                print("k")
        for amv in liap:
            wr.writerow(amv)
        wr.writerow("")
if os.path.exists("resultsfull.csv"):
    os.remove("resultsfull.csv")
with open("resultsfull.csv", "x", newline="") as outfil:
    fields = list(output_data.keys())
    wr = csv.writer(outfil)
    wr.writerow(fields)

    for e, curvl in enumerate(cutlis):
        wr.writerow(["Threshold Value:", list(curvl.keys())[0]])
        items = np.array(list(curvl[list(curvl.keys())[0]].values()))
        liap = []
        for row in items.T:
            rowr = []
            for enr, it in enumerate(row):
                try:
                    rowr.append(float(it))
                except Exception as e:
                    rowr.append(it)
                    pass
            if rowmk[0].rsplit(" ", 1)[0] != rowr[0].rsplit(" ", 1)[0]:
                rowmk = rowr
            elif rowmk[0].rsplit(" ", 1)[0] == rowr[0].rsplit(" ", 1)[0]:
                rowl = []
                rowl.append(rowmk[0].rsplit(" ", 1)[0])
                rowl.append(rowr[1] + rowmk[1])
                rowl.append((rowr[5] + rowmk[5]) / 2)
                print("k")
                liap.append(rowl)
            if rowr[0].rsplit(" ", 1)[0] == "NORM FIS1":
                rowl = []
                rowl.append(rowr[0].rsplit(" ", 1)[0])
                rowl.append(rowr[1])
                rowl.append(rowr[5])
                liap.append(rowl)
                print("k")
        output_data1 = pd.DataFrame(output_data)
        output_data2 = output_data1.to_numpy()
        for amv in output_data2:
            wr.writerow(amv)
        wr.writerow("")


dfval = pd.DataFrame(varvals)
dfarr = dfval.T.to_dict()
prv = ["0"]
dfl = []
rd = {}
for row in dfarr:
    row = dfarr[row]
    try:
        rd[row["Name"].rsplit(" ", 1)[0]].append(
            {
                "Peripheral": row["Peripheral"],
                "Midzone": row["Midzone"],
                "Stadard Dev": row["Stadard Dev"],
            }
        )
    except:
        rd.update({row["Name"].rsplit(" ", 1)[0]: []})
        rd[row["Name"].rsplit(" ", 1)[0]].append(
            {
                "Peripheral": row["Peripheral"],
                "Midzone": row["Midzone"],
                "Stadard Dev": row["Stadard Dev"],
            }
        )
crolis = {}
for row in rd:
    a = 0
    b = 0
    c = []
    rowd = rd[row]

    for e, l in enumerate(rowd):
        a += l["Peripheral"]
        b += l["Midzone"]
        c += l["Stadard Dev"]
        print(l)
    # d = np.mean(c)
    # c = np.std(c)
    crolis.update({row: {"Peripheral": a, "Midzone": b, "Sizes": c}})

precleaned_dataframe = pd.DataFrame(crolis).T
idx = precleaned_dataframe.index.str.rsplit(" ", 1)
idx = [x[0] for x in idx]
precleaned_dataframe.index = idx
indx = precleaned_dataframe.index.str.replace(
    "Pro|Pro | lif|lif |lif|-| Series|Series| -|- | ", ""
)
precleaned_dataframe.index = indx

precleaned_dataframe["Ratio of P/M"] = 100 * (
    precleaned_dataframe["Peripheral"]
    / (precleaned_dataframe["Peripheral"] + precleaned_dataframe["Midzone"])
)

precleaned_dataframe = precleaned_dataframe.drop(["Peripheral", "Midzone"], axis=1)
CI = st.t.interval(
    alpha=0.95,
    df=len(precleaned_dataframe.groupby(precleaned_dataframe.index)["Ratio of P/M"])
    - 1,
    loc=precleaned_dataframe.groupby(precleaned_dataframe.index)["Ratio of P/M"].mean(),
    scale=precleaned_dataframe.groupby(precleaned_dataframe.index)[
        "Ratio of P/M"
    ].sem(),
)
# oo = precleaned_dataframe.groupby(precleaned_dataframe.index)["Sizes"]
# Size_CI = st.t.interval(
#     alpha=0.95,
#     df=len(precleaned_dataframe.groupby(precleaned_dataframe.index)["Sizes"]) - 1,
#     loc=precleaned_dataframe.groupby(precleaned_dataframe.index)["Sizes"].mean(),
#     scale=precleaned_dataframe.groupby(precleaned_dataframe.index)["Sizes"].sem(),
# )
_dataframe = (
    precleaned_dataframe.groupby(precleaned_dataframe.index)["Sizes"]
    .apply(lambda x: ",".join(x.astype(str)))
    .reset_index()
)
# _dataframe = _dataframe.Sizes.apply(pd.Series)
e = [ast.literal_eval(x.strip()) for x in _dataframe.Sizes.values]
data = []
for cult in e:
    data.append([x for y in cult for x in y])
# e = [y for r in e for y in r]
_dataframe = pd.DataFrame(data, index=precleaned_dataframe.index.unique()).astype(float)
# _dataframe.columns = ["sz_{}".format(x + 1) for x in _dataframe.columns]
# df3 = _dataframe.Sizes.apply(pd.Series)
# sz_mean = _dataframe.mean()
# sz_sem = _dataframe.sem()
sz = len(_dataframe)
Size_CI = st.t.interval(
    alpha=0.95,
    df=_dataframe.notnull().sum(axis=1) - 1,
    loc=_dataframe.mean(axis=1),
    scale=_dataframe.sem(axis=1),
)
cleaned_dataframe = precleaned_dataframe.groupby(precleaned_dataframe.index).mean()

cleaned_dataframe["Mean Size"] = _dataframe.mean(axis=1)
cleaned_dataframe["High_PM"] = CI[1] - cleaned_dataframe["Ratio of P/M"]
cleaned_dataframe["Low_PM"] = cleaned_dataframe["Ratio of P/M"] - CI[0]

cleaned_dataframe["High_Sz"] = Size_CI[1] - _dataframe.mean(axis=1)
cleaned_dataframe["Low_Sz"] = _dataframe.mean(axis=1) - Size_CI[0]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="Percentage of Peripheral Splits",
        x=cleaned_dataframe.index,
        y=cleaned_dataframe["Ratio of P/M"],
        error_y=dict(
            arrayminus=cleaned_dataframe["Low_PM"], array=cleaned_dataframe["High_PM"]
        ),
    )
)

fig.add_trace(
    go.Bar(
        name="Mean Size",
        x=cleaned_dataframe.index,
        y=cleaned_dataframe["Mean Size"],
        error_y=dict(
            arrayminus=cleaned_dataframe["Low_Sz"], array=cleaned_dataframe["High_Sz"]
        ),
    )
)

fig.update_layout(barmode="group")
fig.show()
fig.write_html("Bar_plot_of_Peripheral_to_Midzone.html")


print("stop")
