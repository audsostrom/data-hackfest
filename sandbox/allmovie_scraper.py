import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime
import csv
from queue import Queue
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
from selenium.common.exceptions import TimeoutException
import os
import threading
import urllib.parse
from fuzzywuzzy import fuzz
import re
import statistics
import math

logging.basicConfig(filename='scraping.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def encode_search_term(term):
    # function takes a search term as input and returns a URL-encoded version of the term (aka no special characters)
    encoded_term = urllib.parse.quote(term, safe='')
    return encoded_term


def clean_string(text):
    # return none if input text is none to avoid processing errors
    if text is None:
        return None
    else:
        # create translation tables to convert subscript and superscript numbers to normal digits
        subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        supers = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        # apply the translation tables
        translation_table = {**subs, **supers}
        # print(text)
        # apply the translation tables to standardize
        translated_text = text.translate(translation_table)
        # initialize a clean string and include only alphanumeric characters and non-consecutive spaces
        cleaned = ''
        for char in translated_text:
            if char.isalnum():
                cleaned += char
            elif char.isspace() and (not cleaned.endswith(' ')):
                cleaned += char
        # convert to lower case and strip whitespace
        cleaned = cleaned.lower()
        cleaned = cleaned.strip()
        return cleaned


def chunks(df, size=2):
    # yields chunks of the dataframe with specified size, used for the batch processing
    # print('starting chunking')
    """Yield successive n-sized chunks from df."""
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size]
    # print('done chunking')


def create_driver():
    # configure options for the Chrome WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Runs Chrome in headless mode.
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('blink-settings=imagesEnabled=false')
    return webdriver.Chrome(options=options)


def process_movie(full_df, index, webdriver=None):
    # extract the most relevant movie url based on similarity scoring
    def extract_url(soup, search_term=None, search_year=None, search_genres=None):
        # identifies the movie list on the website
        movie_list = soup.select_one("#cmn_wrap > div.content-container > div.content > div.results > ul")
        frame = {'title': [], 'release_year': [], 'directors': [], 'genres': [], 'url': [],
                 'similarity_title': [], 'year_diff': [], 'similarity_genres': []}
        df = pd.DataFrame(frame)

        year = False
        directors = False
        genres = False
        url = False
        similarity_scores = False
        similarity_title = False
        year_diff = False
        similarity_genres_scores = False

        # iterates through movies, calculating similarity to the search term, year, and genres
        if movie_list:
            movies = movie_list.find_all('li', class_='movie')
            for movie in movies:
                title_div = movie.find('div', class_='title')
                title = title_div.get_text(strip=True) if title_div else None

                title_link = movie.find('div', class_='title').find('a', href=True)
                url = title_link['href'].replace('https://www.allmovie.com/movie/', "") if title_link else None
                if title:
                    match = re.match(r"(.+?)\((\d{4})\)$", title.strip())
                    if match:
                        title = match.group(1).strip()
                        if search_term:
                            similarity_title = fuzz.token_sort_ratio(title, search_term)
                        year = match.group(2)
                        if search_year:
                            year_diff = abs(float(year) - float(search_year))

                director_div = movie.find('div', class_='artist')
                if director_div:
                    director_links = director_div.find_all('a')
                    if len(director_links) > 0:
                        directors = [link.get_text(strip=True) for link in director_links]
                    else:
                        directors = []
                else:
                    directors = []

                genre_div = movie.find('div', class_='genres')
                if genre_div:
                    genre_links = genre_div.find_all('a')
                    genres = [link.get_text(strip=True) for link in genre_links]
                    if not math.isnan(search_genres):
                        similarity_scores = []
                        for genre in genres:
                            genre_row = []
                            print(search_genres)
                            for search_d in search_genres:
                                similarity_genre = fuzz.token_sort_ratio(genre, search_d)
                                genre_row.append(similarity_genre)
                            similarity_scores.append(genre_row)
                else:
                    genres = []

                similarity_genres_score = 0
                if similarity_scores:
                    all_scores = [score for sublist in similarity_scores for score in sublist]
                    similarity_genres_score = statistics.mean(all_scores) if all_scores else None

                # create dataframe rows for each movie
                df.loc[len(df)] = [title, year, ', '.join(directors), ', '.join(genres), url,
                                   similarity_title, year_diff, similarity_genres_score]

        # calculate weighted scores and sort dataframe to find the best match
        weights = {
            'similarity_title': 4,
            'similarity_genres': 1,
            'year_diff': -3
        }
        df['weighted_score'] = (
                df['similarity_title'] * weights['similarity_title'] +
                df['similarity_genres'] * weights['similarity_genres'] +
                df['year_diff'] * weights['year_diff']
        )

        # sort the DataFrame by the weighted score in descending order to find the best row
        df_sorted = df.sort_values(by='weighted_score', ascending=False)
        if df_sorted.empty:
            return None
        else:
            best_row = df_sorted.iloc[0]
            return 'https://www.allmovie.com/movie/' + str(best_row['url'])

    def extract_details(soup):
        section = soup.select_one("#cmn_wrap > div.content-container.section-1 > div.content > header > div.details")
        if section:
            return section.get_text(separator=' ', strip=True)
        return None

    def extract_text(soup, selector):
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
        return None

    def extract_keywords(soup):
        section = soup.select_one(
            "#cmn_wrap > div.content-container.section-1 > div.content > section.characteristics > div.keywords")
        if section:
            keywords = section.get_text(strip=True).split(", ")
            keywords[0] = keywords[0].replace("Keywords", "")
            return [clean_string(keyword) for keyword in keywords if keyword]
        return []

    def extract_themes(soup):
        section = soup.select_one(
            "#cmn_wrap > div.content-container.section-1 > div.content > section.characteristics > div.themes")
        if section:
            themes = section.get_text(strip=True).split("|")
            themes[0] = themes[0].replace("Themes", "")
            return [clean_string(theme) for theme in themes if theme]
        return []

    def extract_related_movies(soup):
        section = soup.select_one(
            "#cmn_wrap > div.content-container.section-1 > div.content > section.related-highlights > div.related-movies.clearfix")
        related_movies = []
        if section:
            for link in section.find_all('a', href=True):
                title = link['title']
                href = link['href']
                related_movies.append({'title': title, 'href': href})
        return related_movies

    # main function to collect movie data from allmovie based on a given dataframe row
    def allmovie_collection(movie_df, driver):
        movie_name = movie_df['title']
        release_year = datetime.strptime(movie_df['release_date'], "%m/%d/%Y").year
        genres = movie_df['genres']
        start = time.time()
        clean_movie_name = encode_search_term(movie_name)
        url = f"https://www.allmovie.com/search/all/{clean_movie_name}"
        #print(f"url: {url}")
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        allmovie_url = extract_url(soup, movie_name, release_year, genres)
        print(allmovie_url)
        if allmovie_url:
            #print(f"allmovie_url: {allmovie_url}")
            driver.get(allmovie_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            # extract details from the page
            details = extract_details(soup)
            synopsis = extract_text(soup,
                                    "#cmn_wrap > div.content-container.section-1 > div.content > section.review.read-more.synopsis > div.text")
            keywords = extract_keywords(soup)
            themes = extract_themes(soup)
            related_movies = extract_related_movies(soup)
            end = time.time()
            timestamp = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
            duration = round(end - start, 2)
            return {
                'title': movie_name,
                'allmovie_details': details,
                'allmovie_synopsis': synopsis,
                'allmovie_keywords': keywords,
                'allmovie_themes': themes,
                'allmovie_related_movies': related_movies,
                'allmovie_url': allmovie_url
            }
        else:
            return {
                'title': movie_name,
                'allmovie_details': 'NOT FOUND',
                'allmovie_synopsis': 'NOT FOUND',
                'allmovie_keywords': 'NOT FOUND',
                'allmovie_themes': 'NOT FOUND',
                'allmovie_related_movies': 'NOT FOUND',
                'allmovie_url': 'NOT FOUND'
            }

    # load the movie data from the dataframe and process it using the specified or created webdriver
    movie_df = full_df.loc[index]
    title = movie_df['title']
    print(f"processing {title}")
    logging.info(f"processing {title}")
    if webdriver:
        allmovie_data = allmovie_collection(movie_df, webdriver)
        #print(f"processed {title}")
        logging.info(f"processed {title}")
        return allmovie_data
    else:
        webdriver = create_driver()
        allmovie_data = allmovie_collection(movie_df, webdriver)
        webdriver.quit()
        #print(f"processed {title}")
        logging.info(f"processed {title}")
        return allmovie_data
