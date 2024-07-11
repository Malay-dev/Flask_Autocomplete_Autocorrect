import pandas as pd
from pymongo import MongoClient
from pandas import json_normalize
import tldextract

def replace_nan_for_column(dataframe, column_name, replacement_value):
    dataframe[column_name].fillna(replacement_value, inplace=True)
    return dataframe

def combine_columns_to_new_url_column(dataframe, protocol_column, domain_column, new_column_name):
    dataframe[new_column_name] = dataframe.apply(
        lambda row: f"{row[protocol_column]}//{row[domain_column]}/" if row[protocol_column] else row[domain_column],
        axis=1
    )
    return dataframe

def set_default_subdomain(url):
    ext = tldextract.extract(url)
    if ext.subdomain == '':
        return f"www.{url}"
    return url

def extract_info(url):
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
    domain = ext.domain
    suffix = ext.suffix
    return subdomain, domain, suffix

mongo_uri = "mongodb+srv://admin:gTYc5AEnB9PK6GuI@url-dataset-cluster.kfsuavg.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)

database_name = "test"
collection_name = "datas"

def retrieve_data():
    db = client[database_name]
    collection = db[collection_name]
    cursor = collection.find()
    fields_to_include = [
        'url_protocol',
        'url_domain_name',
        'metadata_location',
        'metadata_global_index',
        'metadata_local_index',
        'metadata_category_rank.rank',
        'metadata_category_rank.category',
        'metadata_category'
    ]
    df = json_normalize(list(cursor), sep='_').filter(items=fields_to_include)
    df = df[df['metadata_category'] != 'adult']
    df = df[df['metadata_category'] != '']
    df = df.dropna(subset=['metadata_global_index'])
    df['url_domain_name'] = df['url_domain_name'].apply(set_default_subdomain)
    df[['subdomain', 'domain', 'suffix']] = df['url_domain_name'].apply(lambda x: pd.Series(extract_info(x)))

    df = replace_nan_for_column(df, "metadata_location", "IN")
    df = combine_columns_to_new_url_column(df, "url_protocol", "url_domain_name", "url")

    # Sorting the dataframe by metadata_global_index
    df = df.sort_values(by='metadata_global_index')

    # Creating a new column 'index' numbered from 1 to end
    df = df.assign(index=range(1, len(df) + 1))

    csv_filename = "data.csv"
    df.to_csv(csv_filename, index=False)
    client.close()

    print(f"Data exported to {csv_filename}")

# Call the retrieve_data function to execute the process
retrieve_data()
