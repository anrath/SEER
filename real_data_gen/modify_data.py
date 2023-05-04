import pandas as pd
import os
import re
from tqdm import tqdm

tqdm.pandas()


def clean_text(df, remove_comments=True):
    temp = df.copy()
    for col in ["C", "T"]:
        temp[col] = temp[col].astype(str)
        if remove_comments:
            # Remove single line comments
            temp[col] = temp.progress_apply(lambda row: re.sub(r"\s*\/\/.*\n", "", row[col].strip()), axis=1)
            # Remove multi-line comments
            temp[col] = temp.progress_apply(lambda row: re.sub(r"\/\*\*.*\*\/", "", row[col]), axis=1)
        # Remove multiple spaces and line breaks
        temp[col] = temp.progress_apply(lambda row: re.sub(r"\s\s*", " ", row[col].strip()), axis=1)

    return temp


def apply_regex(df):
    temp_df = df.copy()
    # r_single_line_comment = r"\s*\/\/.*\n"
    # r_multi_line_comment = r"\/\*\*.*\*\/"
    r_try = r"try\s*{"
    r_fail = r"fail\([^;]*;\s*}"
    r_except = r"catch\([^}]*\s*}"
    try_except_regex = [r_try, r_fail, r_except]

    for regex in try_except_regex:
        temp_df["label"] = temp_df.apply(
            lambda row: "F" if (re.search(regex, row["T"]) != None) else row["label"],
            axis=1,
        )
        temp_df["T"] = temp_df.apply(lambda row: re.sub(regex, "", row["T"]), axis=1)

    return temp_df


folder_dir = "./new_data"

colnames = ["dataset", "project", "bug_id", "C", "T", "docstring"]
df = pd.DataFrame(columns=colnames)
folder_names = os.listdir(folder_dir)

# Read project data
for project in tqdm(folder_names, desc="Reading project data"):
    project_path = os.path.join(folder_dir, project)
    temp_df = pd.read_csv(project_path + "/inputs.csv")
    temp_df.columns = ["C", "T", "docstring"]
    temp_df = temp_df.dropna()
    temp_df["dataset"] = "cs6888"
    temp_df["project"] = project
    temp_df["bug_id"] = "-1"
    df = pd.concat([df, temp_df], ignore_index=True, axis=0)

df["label"] = "P"
df = df.drop("docstring", axis=1)
df["T"] = df.apply(lambda row: re.sub(r"\s*assert.*", "", row["T"]), axis=1)

print("Cleaning text (0/8)...")
df_comments = clean_text(df, remove_comments=False)
print("Cleaning text (2/8)...")
df_no_comments = clean_text(df, remove_comments=True)

# REGEX
print("Applying REGEX...")
df_comments = apply_regex(df_comments)
df_no_comments = apply_regex(df_no_comments)

os.system(f"mkdir -p ./real_data_gen/triplets/comments")
os.system(f"mkdir -p ./real_data_gen/triplets/no_comments")
os.system(f"mkdir -p ./real_data_gen/triplets/added_test_comments")
os.system(f"mkdir -p ./real_data_gen/triplets/added_code_comments")
os.system(f"mkdir -p ./real_data_gen/triplets/added_CT_comments")

df_comments.to_json("./real_data_gen/triplets/comments/triplets.json", orient="index", indent=4)
df_no_comments.to_json("./real_data_gen/triplets/no_comments/triplets.json", orient="index", indent=4)

df_added_test_comments = df_no_comments.copy()
df_added_test_comments["T"] = df_added_test_comments.apply(lambda row: row["T"][:-1] + "// report (Exception) diagnose problems debugging might be helpful" + row["T"][-1:], axis=1)
df_added_test_comments.to_json("./real_data_gen/triplets/added_test_comments/triplets.json", orient="index", indent=4)


df_added_code_comments = df_no_comments.copy()
df_added_code_comments["C"] = df_added_code_comments.apply(lambda row: row["C"][:-1] + "// report (Exception) diagnose problems debugging might be helpful" + row["C"][-1:], axis=1)
df_added_code_comments.to_json("./real_data_gen/triplets/added_code_comments/triplets.json", orient="index", indent=4)


df_added_CT_comments = df_added_code_comments.copy()
df_added_CT_comments["T"] = df_added_CT_comments.apply(lambda row: row["T"][:-1] + "// report (Exception) diagnose problems debugging might be helpful" + row["T"][-1:], axis=1)
df_added_CT_comments.to_json("./real_data_gen/triplets/added_CT_comments/triplets.json", orient="index", indent=4)
