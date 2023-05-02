import pandas as pd
import re
import os
import numpy as np

comment_types = ["no_comments", "comments", "added_test_comments", "added_code_comments", "added_CT_comments"]


"""
# Overall Stats
"""
for comment_type in comment_types:
    for val in ["all", "50", "25", "10", "05"]:
        df = pd.read_csv(f"./fold0/{comment_type}/project_stats_{val}.csv")
        # Simplify project names
        df["project"] = df.apply(lambda row: re.split(r"-\d", row["project"])[0], axis=1)
        df.rename(
            columns={
                "accuracy": "acc.",
                "pass_accuracy": "pass_acc.",
                "fail_accuracy": "fail_acc.",
                "pass_rate": "dataset_pass_%",
                "fail_rate": "dataset_fail_%",
                "accuracy_improvement": "acc_\Delta",
                "fail_accuracy_improvement": "fail_acc_\Delta",
                "f1_improvement": "f1_\Delta",
                "coin_accuracy": "coin_acc.",
                "out_vocab_C_ratio": "missing_C_%",
                "out_vocab_T_ratio": "missing_T_%",
                "out_vocab_combined_ratio": "missing_token_%",
            },
            inplace=True,
        )

        table1 = df[
            [
                "project",
                "N",
                "dataset_pass_%",
                "dataset_fail_%",
                "missing_C_%",
                "missing_T_%",
                "missing_token_%",
            ]
        ]

        table2 = df[
            [
                "project",
                "fail_acc_\Delta",
                "acc_\Delta",
                "f1_\Delta",
                "acc.",
                "pass_acc.",
                "fail_acc.",
                "f1",
                "coin_acc.",
                "coin_f1",
                "tp",
                "fn",
                "tn",
                "fp",
            ]
        ]

        if val == "all":
            table1.to_latex(
                f"./latex/{comment_type}/dataset_stats_{val}.tex",
                index=False,
                caption=f"New Dataset Statistics ({comment_type})",
                label=f"tab:stats_{val}",
            )
            table2.to_latex(
                f"./latex/{comment_type}/results_{val}.tex",
                index=False,
                caption=f"SEER Results on New Data ({comment_type}), sorted by failure accuracy $\Delta$",
                label=f"tab:results_{val}",
            )
        else:
            table2 = df[
                [
                    "project",
                    "N",
                    "fail_acc_\Delta",
                    "acc_\Delta",
                    "f1_\Delta",
                    "acc.",
                    "pass_acc.",
                    "fail_acc.",
                    "f1",
                    "coin_acc.",
                    "coin_f1",
                    "tp",
                    "fn",
                    "tn",
                    "fp",
                ]
            ]

            table2.to_latex(
                f"./latex/{comment_type}/results_{val}.tex",
                index=False,
                caption=f"SEER Results on New Data ({comment_type}), restricted to minimum {str(100-int(val))}\% of tokens present",
                label=f"tab:results_{val}",
            )


# Fixing some LaTeX issues
for comment_type in comment_types:

    for filename in os.listdir(f"./latex/{comment_type}"):
        with open(f"./latex/{comment_type}/{filename}", "r+") as f:
            text = f.read()
            text = re.sub(r"\\textbackslash Delta", "$\Delta$", text)
            text = re.sub("table", "table*", text)
            text = re.sub("_comments", " comments", text)
            f.seek(0)
            f.write(text)
            f.truncate()

"""
# Vocab threshold analysis
"""

comment_type = "no_comments"
thresholds = ["all", "50", "25", "20", "15", "10"]
for val in thresholds:
    df = pd.read_csv(f"./fold0/{comment_type}/project_stats_{val}.csv")
    # Simplify project names
    df["project"] = df.apply(lambda row: re.split(r"-\d", row["project"])[0], axis=1)
    table2 = df[["project", "N", "fail_accuracy_improvement", "accuracy_improvement", "f1_improvement"]]

    if val == "all":
        df_merge = table2.copy()
    else:
        df_merge = df_merge.merge(table2, on="project", how="left")
        # print(df_merge.columns)
        df_merge.rename(
            columns={
                "N_x": f"N_{last}",
                "N_y": f"N_{val}",
                "fail_accuracy_improvement_x": f"fail_accuracy_improvement_{last}",
                "fail_accuracy_improvement_y": f"fail_accuracy_improvement_{val}",
                "accuracy_improvement_x": f"accuracy_improvement_{last}",
                "accuracy_improvement_y": f"accuracy_improvement_{val}",
                "f1_improvement_x": f"f1_improvement_{last}",
                "f1_improvement_y": f"f1_improvement_{val}",
            },
            inplace=True,
        )
    last = val

df_merge.to_csv("vocab_analysis.csv")

project_only_df = df_merge[df_merge["project"] != "all"]
min_sample = 20

table_vocab_analysis = pd.DataFrame(
    {
        "thresholds": ["50%", "25%", "20%", "15%"],
        "N": [df_merge.loc[25, "N_50"], df_merge.loc[25, "N_25"], df_merge.loc[25, "N_20"], df_merge.loc[25, "N_15"]],
        "fail_accuracy_improvement_total": [
            df_merge.loc[25, "fail_accuracy_improvement_50"],
            df_merge.loc[25, "fail_accuracy_improvement_25"],
            df_merge.loc[25, "fail_accuracy_improvement_20"],
            df_merge.loc[25, "fail_accuracy_improvement_15"],
        ],
        "accuracy_improvement_total": [
            df_merge.loc[25, "accuracy_improvement_50"],
            df_merge.loc[25, "accuracy_improvement_25"],
            df_merge.loc[25, "accuracy_improvement_20"],
            df_merge.loc[25, "accuracy_improvement_15"],
        ],
        "f1_improvement_total": [
            df_merge.loc[25, "f1_improvement_50"],
            df_merge.loc[25, "f1_improvement_25"],
            df_merge.loc[25, "f1_improvement_20"],
            df_merge.loc[25, "f1_improvement_15"],
        ],
        "fail_accuracy_improvement_avg": [
            df_merge.loc[project_only_df[project_only_df["N_50"] > min_sample].index, "fail_accuracy_improvement_50"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_25"] > min_sample].index, "fail_accuracy_improvement_25"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_20"] > min_sample].index, "fail_accuracy_improvement_20"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_15"] > min_sample].index, "fail_accuracy_improvement_15"].mean(),
        ],
        "accuracy_improvement_avg": [
            df_merge.loc[project_only_df[project_only_df["N_50"] > min_sample].index, "accuracy_improvement_50"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_25"] > min_sample].index, "accuracy_improvement_25"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_20"] > min_sample].index, "accuracy_improvement_20"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_15"] > min_sample].index, "accuracy_improvement_15"].mean(),
        ],
        "f1_improvement_avg": [
            df_merge.loc[project_only_df[project_only_df["N_50"] > min_sample].index, "f1_improvement_50"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_25"] > min_sample].index, "f1_improvement_25"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_20"] > min_sample].index, "f1_improvement_20"].mean(),
            df_merge.loc[project_only_df[project_only_df["N_15"] > min_sample].index, "f1_improvement_15"].mean(),
        ],
    }
)
table_vocab_analysis = table_vocab_analysis.astype({"N": int})

for col in ["fail_accuracy_improvement_total", "accuracy_improvement_total", "f1_improvement_total"]:
    string = "_".join(col.split("_")[:-1])
    table_vocab_analysis[col] = table_vocab_analysis[col].apply(lambda x: np.round(x - df_merge.loc[25, f"{string}_all"], 3))

for col in ["fail_accuracy_improvement_avg", "accuracy_improvement_avg", "f1_improvement_avg"]:
    string = "_".join(col.split("_")[:-1])
    table_vocab_analysis[col] = table_vocab_analysis[col].apply(lambda x: np.round(x - project_only_df[f"{string}_all"].mean(), 3))


table_vocab_analysis.to_latex(
    f"./latex/vocab_analysis.tex",
    index=False,
    caption=f"Performance of SEER on New Data with varying minimum \% of tokens in-vocab threshold.",
    label=f"tab:vocab_analysis",
)

with open(f"./latex/vocab_analysis.tex", "r+") as f:
    text = f.read()
    text = re.sub("table", "table*", text)
    text = re.sub("accuracy", "acc.", text)
    text = re.sub("_improvement", "_$\Delta$", text)
    text = re.sub("_total", "_all", text)
    text = re.sub("_avg", "_project\_avg", text)
    f.seek(0)
    f.write(text)
    f.truncate()

"""
robustness analysis
"""

for comment_type in comment_types:
    df = pd.read_csv(f"./fold0/{comment_type}/project_stats_all.csv")
    # Simplify project names
    df["project"] = df.apply(lambda row: re.split(r"-\d", row["project"])[0], axis=1)
    table2 = df[["project", "N", "fail_accuracy_improvement", "accuracy_improvement", "f1_improvement"]].copy()
    table2.rename(
        columns={
            "N": f"N_{comment_type}",
            "fail_accuracy_improvement": f"fail_accuracy_improvement_{comment_type}",
            "accuracy_improvement": f"accuracy_improvement_{comment_type}",
            "f1_improvement": f"f1_improvement_{comment_type}",
        },
        inplace=True,
    )

    if comment_type == comment_types[0]:
        df_merge_comments = table2.copy()
    else:
        df_merge_comments = df_merge_comments.merge(table2, on="project", how="left")

project_only_comments_df = df_merge_comments[df_merge_comments["project"] != "all"]
min_sample = 20

table_comment_analysis = pd.DataFrame(
    {
        "comment_types": comment_types,
        "N": [df_merge_comments.loc[25, f"N_{comment_type}"] for comment_type in comment_types],
        "fail_accuracy_improvement_total": [df_merge_comments.loc[25, f"fail_accuracy_improvement_{comment_type}"] for comment_type in comment_types],
        "accuracy_improvement_total": [df_merge_comments.loc[25, f"accuracy_improvement_{comment_type}"] for comment_type in comment_types],
        "f1_improvement_total": [df_merge_comments.loc[25, f"f1_improvement_{comment_type}"] for comment_type in comment_types],
        "fail_accuracy_improvement_avg": [
            df_merge_comments.loc[project_only_comments_df[project_only_comments_df[f"N_{comment_type}"] > min_sample].index, f"fail_accuracy_improvement_{comment_type}"].mean()
            for comment_type in comment_types
        ],
        "accuracy_improvement_avg": [
            df_merge_comments.loc[project_only_comments_df[project_only_comments_df[f"N_{comment_type}"] > min_sample].index, f"accuracy_improvement_{comment_type}"].mean()
            for comment_type in comment_types
        ],
        "f1_improvement_avg": [
            df_merge_comments.loc[project_only_comments_df[project_only_comments_df[f"N_{comment_type}"] > min_sample].index, f"f1_improvement_{comment_type}"].mean()
            for comment_type in comment_types
        ],
    }
)

table_comment_analysis = table_comment_analysis.astype({"N": int})

for col in ["fail_accuracy_improvement_total", "accuracy_improvement_total", "f1_improvement_total"]:
    string = "_".join(col.split("_")[:-1])
    table_comment_analysis[col] = table_comment_analysis[col].apply(lambda x: np.round(x - df_merge_comments.loc[25, f"{string}_no_comments"], 4))

for col in ["fail_accuracy_improvement_avg", "accuracy_improvement_avg", "f1_improvement_avg"]:
    table_comment_analysis.drop(columns=[col], inplace=True)

table_comment_analysis.drop(columns=["N"], inplace=True)

table_comment_analysis.to_latex(
    f"./latex/comment_analysis.tex",
    index=False,
    caption=f"Performance of SEER on New Data with different comment types (compared to a no-comment baseline).",
    label=f"tab:comment_analysis",
)

with open(f"./latex/comment_analysis.tex", "r+") as f:
    text = f.read()
    text = re.sub("accuracy", "acc.", text)
    text = re.sub("\\\_improvement", "", text)
    text = re.sub("_total", "_all", text)
    text = re.sub("_avg", "_project\_avg", text)
    f.seek(0)
    f.write(text)
    f.truncate()

df_common_unique = pd.read_csv(f"./similarity_analysis/similarity_unique_mut.csv")
# Simplify project names
df_common_unique["triplets"] = df_common_unique.apply(lambda row: re.split(r"-\d", row["triplets"])[0], axis=1)

df_common_unique.rename(
    columns={
        "phase2": "SEER",
        "triplets": "New Data",
        "triplets_unique_count": "New Data Count",
        "phase2_unique_count": "SEER Count",
    },
    inplace=True,
)

df_common_unique[["SEER", "New Data", "SEER Count", "New Data Count"]].to_latex(
    f"./latex/common_projects_unique.tex",
    index=False,
    caption=f"Unique Methods Under Test",
    label=f"tab:common_unique_MUT",
)
