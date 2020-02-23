import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

'''
For more details on the arxiv api see the docs :)
https://arxiv.org/help/api/user-manual
'''


class ArxivScraper:

    '''
    A basic scraper using the Arxiv API - For more details on the Arxiv API see the docs: https://arxiv.org/help/api/user-manual

    Parameters
    -----------------
    search_terms : A List of search terms - List
    max_results : Number of results you want returned  - int
    write_csv : Boolean for whether you want to write the query results to a csv in th elocal directory - Boolean
    '''

    def __init__(self, search_terms=["neural","network"],max_results = 100, write_csv=False): 

        self.query_string = 'http://export.arxiv.org/api/query?search_query=all:'
        self.query_string = ''.join([self.query_string+ "+"+search_terms[i] for i in range(len(search_terms)) if i >0])
        self.max_results = max_results
        self.query_string = self.query_string + f"&max_results={self.max_results}"
        self.write_csv = write_csv
    
    def scrape(self):
        '''
        Method that:
        - Sends request string to Arxiv API
        - Parses response and extracts 'Title' and 'Summary'
        - Populates a Pandas DataFrame with the data
        - Writes DataFrame to csv(Optional)

        Returns
        -----------------
        df : Pandas DataFrame of Titles and Summaries returned by get request
        '''

        query =requests.get(self.query_string)
        logging.info("retreived results of search string from Arxiv")

        response =query.content
        soup = BeautifulSoup(response,'html.parser')
        summaries = soup.find_all('summary')
        titles= soup.find_all('title')

        df =pd.DataFrame(columns=['title','text'])
        titles = titles[1:]
        for title,summary in zip(titles,summaries):
            temp_df=pd.DataFrame({'title':[title.text.strip('\t').strip()],'text':[summary.text.strip('\t').strip()]})
            df = pd.concat([df,temp_df])

        for row in range(len(df)):
            df['title'].iloc[row]=' '.join(df['title'].iloc[row].split('\n'))
            df['text'].iloc[row]=' '.join(df['text'].iloc[row].split('\n'))

        df=df.drop_duplicates(subset=['title','text'])
        df=df.dropna()
        logging.info("cleaned data")

        if self.write_csv:
            df.to_csv('query_results.csv',sep='\t',header=False,index=False)
            logging.info("written to csv")

        return df

if __name__ == "__main__":
    scraper = ArxivScraper()
    df = scraper.scrape()