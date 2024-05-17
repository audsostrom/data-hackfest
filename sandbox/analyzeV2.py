import logging
import sys
import threading
import time
import allmovie_scraper as scraper
import pandas as pd
import os
import csv
from queue import Queue
import subprocess


def write_to_csv(results, output_file, headers):
    # wrap the dictionary in a list if it's not already a list of dictionaries
    results = [results] if isinstance(results, dict) else results
    # create dataframe from results
    df = pd.DataFrame(results)
    # reorder dataframe columns to match the expected csv headers
    df = df[headers]
    # write to csv, append if the file exists
    df.to_csv(output_file, mode='a', header=False, index=False)


def get_existing_ids(output_file):
    try:
        # read csv and return a list of existing ids to avoid duplicates
        existing_df = pd.read_csv(output_file)
        return existing_df['id'].tolist()
    except:
        # return empty list if there's an error, e.g., file not found
        return []


def thread_worker(data, index):
    # wrap the scraper process in a function that works with threading
    result = scraper.process_movie(data, index)
    return [result, index]


def process_chunk(chunk, full_df):
    results = []
    result_queue = Queue()  # thread-safe queue to collect results
    threads = []
    for index, row in chunk.iterrows():
        args = (full_df, index)
        # start a new thread to process each row in the chunk
        thread = threading.Thread(target=lambda q, arg1, arg2: q.put(thread_worker(arg1, arg2)),
                                  args=(result_queue, *args))
        thread.start()
        threads.append(thread)
    # wait for all threads to complete
    for thread in threads:
        thread.join()
    # retrieve results from the queue
    while not result_queue.empty():
        results.append(result_queue.get())
    output = []
    for item in results:
        # merge new data from scraping with existing row data
        new_data = item[0]
        old_data = full_df.loc[item[1]].to_dict()
        merged_data = {**old_data, **new_data}
        output.append(merged_data)
    return output


def main():
    try:
        # load data from a csv file
        #kaggle_data = pd.read_csv("data/test.csv")
        kaggle_data = pd.read_csv("data/kaggle-asaniczka/TMDB_movie_dataset_v11.csv")
        headers = [col for col in kaggle_data.columns] + [
            'allmovie_details', 'allmovie_synopsis', 'allmovie_keywords',
            'allmovie_themes', 'allmovie_related_movies', 'allmovie_url'
        ]
        output_file = 'new_processed_movies.csv'
        existing_ids = get_existing_ids(output_file)
        # filter data to process only new entries
        kaggle_data = kaggle_data[~kaggle_data['id'].isin(existing_ids)]

        # initialize csv file with headers if it does not exist
        if not os.path.isfile(output_file):
            print(f"output file {output_file} doesn't exist...creating...")
            scraper.write_rows_to_csv([], output_file, headers)  # This will just write headers
            print("done.")

        for chunk in scraper.chunks(kaggle_data, size=5):
            start_time = time.time()
            # process each chunk and write results to csv
            results = process_chunk(chunk, kaggle_data)
            write_to_csv(results, output_file, headers)
            print(f"chunk with {len(chunk)} threads took", round((time.time() - start_time), 2), "seconds")
            logging.info((f"chunk with {len(chunk)} threads took", round((time.time() - start_time), 2), "seconds"))
            print("-"*100)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        logging.info(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    main()
