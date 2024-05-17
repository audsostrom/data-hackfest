import pandas as pd
import ast
import csv
import numpy as np


def clean_imdb_id(raw_id):
    # remove 'tt' and format imdb id to ensure it has leading zeros to make eight digits
    cleaned_id = str(raw_id).replace("tt", "")
    cleaned_id = f"{cleaned_id:0>{8}}"
    return cleaned_id


def clean_text(string):
    # standardize text by making it lower case, replacing special characters and spaces
    # print(string)
    cleaned_string = string.lower().replace(" |", "").replace("/", "_").replace("&", "and").replace("'", "").replace(" ", "_")
    cleaned_string = cleaned_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    # print(cleaned_string)
    # print("-"*100)
    return cleaned_string


def clean_allmovie_vars(raw_df):
    # find the index of the first column with 'allmovie_' prefix and split dataframe
    target_index = next((i for i, col in enumerate(raw_df.columns) if col.startswith("allmovie_")), None)

    if target_index is None:
        return raw_df

    # Split the DataFrame into two parts
    df_old = raw_df.iloc[:, :target_index]
    df_new = raw_df.iloc[:, target_index:]

    # process and clean specific columns from the 'allmovie_' prefixed data
    def process_column(df_old, df_new, column_name, process_func):
        if column_name in df_new.columns:
            if column_name not in df_old.columns:
                df_old[column_name] = None
            for idx, row in df_new[column_name].items():
                if not pd.isna(row) and "NOT FOUND" not in row:
                    process_func(df_old, idx, row)

    # clean and standardize genres, sub-genres, countries, and other fields
    def process_details(df_old, idx, row):
        details = row.split(" | ")
        details = [item.split(" - ", maxsplit=1) for item in details]
        for detail_pair in details:
            header = "allmovie_details_" + detail_pair[0].lower().replace(" ", "_")
            if header not in df_old.columns:
                df_old[header] = None
            detail = detail_pair[1].replace(" , ", ", ")
            if header == "allmovie_details_genres":
                detail = ', '.join([clean_text(item) for item in detail.split(", ")])
            if header == "allmovie_details_sub-genres":
                detail = ', '.join([clean_text(item) for item in detail.split(", ")])
            if header == 'allmovie_details_release_date':
                detail = detail.replace("), ", ")), ").split("), ")
            if header == 'allmovie_details_run_time':
                detail = detail.replace(" min.", "").replace(" |", "")
            if header == "allmovie_details_countries":
                detail = ', '.join([clean_text(item) for item in detail.split(", ")])
            df_old.at[idx, header] = detail

    # clean and process other 'allmovie_' columns like keywords, themes, etc.
    def process_keywords(df_old, idx, row):
        keywords = ", ".join([item.replace(" ", "_") for item in ast.literal_eval(row)])
        df_old.at[idx, 'allmovie_keywords'] = keywords

    def process_themes(df_old, idx, row):
        themes = ", ".join([item.replace(" ", "_") for item in ast.literal_eval(row)])
        df_old.at[idx, 'allmovie_themes'] = themes

    def process_related_movies(df_old, idx, row):
        related_movies = {item['title']: item['href'].replace("/movie/", "") for item in ast.literal_eval(row)}
        df_old.at[idx, 'allmovie_related_movies'] = related_movies

    def process_url(df_old, idx, row):
        url = row.replace("https://www.allmovie.com/movie/", "")
        df_old.at[idx, 'allmovie_url'] = url

    def process_synopsis(df_old, idx, row):
        df_old.at[idx, 'allmovie_synopsis'] = row.lower()

    # apply the processing functions to relevant columns
    process_column(df_old, df_new, 'allmovie_details', process_details)
    process_column(df_old, df_new, 'allmovie_keywords', process_keywords)
    process_column(df_old, df_new, 'allmovie_themes', process_themes)
    process_column(df_old, df_new, 'allmovie_related_movies', process_related_movies)
    process_column(df_old, df_new, 'allmovie_url', process_url)
    process_column(df_old, df_new, 'allmovie_synopsis', process_synopsis)

    # drop the original 'allmovie_details' column since it's redundant now
    df_old = df_old.drop(columns=['allmovie_details'])

    # prefix non-allmovie columns with 'kaggle_' for clarity
    df_old.columns = [col if col.startswith("allmovie_") or col == "imdbId" else "kaggle_"+col for col in df_old.columns]

    return df_old


def merge_grouplens(grouplens_folder, interim_df):
    # load grouplens ids from csv, format imdb ids with leading zeros for consistency
    grouplens_ids = pd.read_csv(f"{grouplens_folder}/links.csv")
    grouplens_ids['imdbId'] = grouplens_ids['imdbId'].apply(lambda x: f"{x:0>{8}}")
    # prefix all columns except 'imdbId' with 'grouplens_' to distinguish them
    grouplens_ids.columns = ["grouplens_" + col if col != "imdbId" else col for col in grouplens_ids.columns]
    # clean imdb ids in the interim dataframe for accurate merging
    interim_df['imdbId'] = interim_df['kaggle_imdb_id'].apply(lambda x: clean_imdb_id(x))
    # merge grouplens data with the interim data on imdb ids
    merged_data = pd.merge(grouplens_ids, interim_df, on='imdbId', how='right') #outer if we want to merge them including missing
    return merged_data


def remove_bad(df):
    def extract_years(date_list):
        # return a list of years extracted from a list of date strings
        if not isinstance(date_list, list):
            return []
        years = []
        for date in date_list:
            try:
                year = pd.to_datetime(date.split(' (')[0]).year
                years.append(year)
            except Exception as e:
                print(f"Error parsing date: {date}, Error: {e}")
                continue
        return years

    def blank_columns_with_prefix_on_condition(df, check_column, threshold, prefix):
        # blank out columns where the value in check_column exceeds a threshold
        mask = df[check_column] > threshold

        # get columns to blank out based on the prefix
        columns_to_blank = [col for col in df.columns if col.startswith(prefix)]
        print(f"Columns to blank: {columns_to_blank}")

        # set the specified columns to NaN for rows that meet the condition
        df.loc[mask, columns_to_blank] = np.nan

        return df

    # convert runtime columns to numeric and calculate the absolute difference
    df['kaggle_runtime'] = pd.to_numeric(df['kaggle_runtime'], errors='coerce')
    df['allmovie_details_run_time'] = pd.to_numeric(df['allmovie_details_run_time'], errors='coerce')
    df['eval_delta_runtime'] = (df['kaggle_runtime'] - df['allmovie_details_run_time']).abs()

    # extract and clean years from release dates, then calculate the absolute difference
    df['clean_kaggle_releaseyear'] = pd.to_datetime(df['kaggle_release_date'], errors='coerce').dt.year
    df['clean_allmovie_releaseyear'] = df['allmovie_details_release_date'].apply(extract_years).apply(lambda x: min(x) if x else np.nan)
    df['eval_delta_releaseyear'] = (df['clean_kaggle_releaseyear'] - df['clean_allmovie_releaseyear']).abs()

    # apply conditions to blank out columns based on differences in runtime and release year
    df = blank_columns_with_prefix_on_condition(df, "eval_delta_releaseyear", 15, "allmovie_")
    df = blank_columns_with_prefix_on_condition(df, "eval_delta_runtime", 15, "allmovie_")

    return df



raw_data = pd.read_csv("new_processed_movies.csv")
df_pass1 = clean_allmovie_vars(raw_data)
df_pass2 = merge_grouplens("data/grouplens/ml-25m", df_pass1)
df_pass3 = remove_bad(df_pass2)
df_pass3.to_csv('clean_data.csv', index=False)
df_pass3[:2000].to_csv('test_clean_data.csv', index=False)