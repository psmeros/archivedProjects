import requests
from bs4 import BeautifulSoup
from common_utils import make_sure_path_exists
import os

ROOT_URL = 'https://bigbangtrans.wordpress.com/'
output_dir = 'raw_files'

def main():
    main_page_html = requests.get(ROOT_URL)
    main_soup = BeautifulSoup(main_page_html.text, 'html.parser')
    pages_sidebar_soup = main_soup.find("div", {"id": "pages-2"})
    all_links = [x.find('a')['href'] for x in pages_sidebar_soup.find_all('li') if x.find('a').get_text() != 'About']
    print('Links scraped! Number of episodes:')
    print(len(all_links))
    make_sure_path_exists(output_dir)
    for current_url in all_links:
        current_bowl = BeautifulSoup(requests.get(current_url).text, 'html.parser')
        current_main_text = current_bowl.find('div', {'class': 'entrytext'})
        current_title = current_bowl.find('h2', {'class': 'title'}).get_text()
        current_title = current_title.replace('&nbsp;', ' ')
        current_all_paragraphs = current_main_text.find_all('p')
        current_all_paragraphs = [x.get_text().strip() for x in current_all_paragraphs]
        current_all_paragraphs = [x for x in current_all_paragraphs if len(x)>0]
        while ('Written by' in current_all_paragraphs[-1] or 'Teleplay: ' in current_all_paragraphs[-1] or
                'Story: ' in current_all_paragraphs[-1]):
            print('Removing line:')
            print(current_all_paragraphs[-1])
            current_all_paragraphs = current_all_paragraphs[:-1]

        print('------------')
        print('Description of the first scene')
        print(current_all_paragraphs[0])
        filename = current_title.split('-')[0].lower().strip().replace(' ', '_').replace('/','_') + '.txt'
        with open(os.path.join(output_dir, filename), 'w', encoding='utf8') as f:
            f.write('>> ' + current_title + '\n')
            f.write('\n'.join(current_all_paragraphs))
        print('Document written:')
        print(current_title)
    print('Whew! Finished.')

if __name__ == '__main__':
    main()