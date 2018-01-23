import pandas




def remove_duplicates(data):
    more_than_one = data.url.value_counts() > 1
    more_than_one = more_than_one[more_than_one].index
    return data[~data.url.isin(more_than_one)]


def format_url(data):
    data['url'] = data.product_page_url
    return data.drop('product_page_url', axis=1)


def combine(cosmetics, vico1, vico2):
    return cosmetics.merge(vico1, how='outer').merge(vico2, how='outer')


columns = ['url', 'sku', 'brand', 'ean', 'gtin', 'price', 'asin', 'gtin13', 'currency']
cosmetics_data = pandas.read_csv('data/CosmeticsTestData.csv', sep=';')
vico1 = pandas.read_csv('data/more_data/VICO_Sample_001_pages_with_features_clean.csv')
vico2 = pandas.read_csv('data/more_data/VICO_Sample_002_pages_with_features_clean.csv')


cosmetics_data = remove_duplicates(cosmetics_data)
vico1 = format_url(vico1)
vico2 = format_url(vico2)

cosmetics_data = cosmetics_data[['url', 'sku', 'brand', 'ean', 'gtin13', 'asin', 'price', 'currency']]
vico1 = vico1[['sku', 'brand', 'ean', 'url']]
vico2 = vico2[['gtin', 'sku', 'ean', 'asin', 'brand', 'price', 'url']]
import ipdb
ipdb.sset_trace()
combined = combine(cosmetics_data, vico1, vico2)

