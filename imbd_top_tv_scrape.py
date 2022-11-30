import requests
import re


def main():
    get_imdb_source()
    get_imdb_data()
    pass
    
def get_imdb_source():
    url = 'https://www.imdb.com/chart/toptv/?ref_=nv_tvv_250'

    imdb_page = requests.get(url)
    imdb_text = imdb_page.text

    imdb_file = open('TESTSCRAPE.txt', 'w', encoding ='utf-8')
    imdb_file.write(imdb_text)
    imdb_file.close()
    
def get_imdb_data():
    myfile = open('TESTSCRAPE.txt', 'r')
    imdb_text = myfile.read()
    
    imdb_data = open('imdb_top_tv.txt', 'w')
    imdb_data.write('Name\tRelease_Year\tRating\n')
    
    super_start_string = '<td class="posterColumn">'
    super_start = imdb_text.find(super_start_string)
    while super_start != -1:
        name = get_tv_name(imdb_text, super_start)
        release_year = get_release_year(imdb_text, super_start)
        rating = get_rating(imdb_text, super_start)
        print(f"{name} {release_year} {rating}", end='\n\n')
        imdb_data.write(f"{name}\t{release_year}\t{rating}\t\n")
        super_start = imdb_text.find(super_start_string, super_start+1)
    

    myfile.close()
    imdb_data.close()
        
def get_tv_name(imdb_text, super_start):
    start = imdb_text.find('<td class="titleColumn">', super_start)
    end = imdb_text.find('</a>', start)
    attribute = imdb_text[start:end]
    attribute = re.sub('\s+', ' ', attribute)
    attribute = re.sub('.* >', '', attribute)
    
    return attribute

def get_release_year(imdb_text, super_start):
    start = imdb_text.find('<td class="titleColumn">', super_start)
    end = imdb_text.find('</span>', start)
    attribute = imdb_text[start:end]
    attribute = re.sub('\s+', ' ', attribute)
    attribute = re.sub('.*>\(', '', attribute)
    attribute = attribute.strip(')')
    
    return attribute   
  
def get_rating(imdb_text, super_start):
    start = imdb_text.find('<td class="ratingColumn imdbRating">', super_start)
    end = imdb_text.find('</strong>', start)
    attribute = imdb_text[start:end]
    attribute = re.sub('\s+', ' ', attribute)
    attribute = re.sub('.*s">', '', attribute)
    attribute = attribute.strip(')')
    
    return attribute    
    
main()
