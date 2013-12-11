""" =================================================================
HTML Parser

For:
  CSCI-GA 3033 Statistical Natural Language Processing
  @ New York University
  Fall 2013
================================================================= """

import numpy as np
import pandas as p

import re
import json

from unidecode import unidecode
from boilerpipe.extract import Extractor
from bs4 import BeautifulSoup

tags = ['title', 'h1', 'h2', 'h3', 'strong', 'b', 'a', 'img', 'p']

def main():
    training_data = np.array(p.read_table('../data/train.tsv'))
    testing_data = np.array(p.read_table('../data/test.tsv'))

    all_data = np.vstack([training_data[:,0:26], testing_data])

    output = file('../data/html_extracted_url.json', 'w')

    for i, page in enumerate(all_data):
        if i != -1:
            if i%100 == 0:
                print i

            # each document gets its own dictionary that we will fill with tags from the html
            extracted = {}

            # get the raw content for the urlid
            raw_file = '../data/raw_content/' + str(page[1])

            # soupify the page
            with file(raw_file, 'rb') as f:
                content = f.read()
                soup = BeautifulSoup(content, 'lxml')

            # break up the URL by . / \ and : to make some words from it
            url = page[0]
            url = re.sub(r"[./\\:]", ' ', url)
            extracted['url'] = [url]

            # boilerplate is json, we don't want the url, just title and body, so lets parse it
            boilerplate = json.loads(page[2])

            # get the title
            if("title" in boilerplate):
                bp_title = boilerplate['title']
                if(bp_title):
                    bp_title = unidecode(bp_title)
            else:
                bp_title = ''

            # get the body
            if("body" in boilerplate):
                bp_body = boilerplate['body']
                if(bp_body):
                    bp_body = unidecode(bp_body)
            else:
                bp_body = ''

            # put them in as a boilerplate "tag"
            extracted['boilerplate'] = [bp_title, bp_body]

            # go through all tags we care about and get the text wrapped in them
            for tag in tags:
                # each instance of the tag will be an element in the tag's array
                tag_content = []

                for t in soup.find_all(tag):
                    t.extract()

                    # images have alt AND/OR title
                    if tag == "img":
                        try:
                            tag_content.append(t['alt'])
                        except KeyError:
                            pass
                        try:
                            tag_content.append(t['title'])
                        except KeyError:
                            pass
                    # other tags just have text
                    else:
                        tag_content.append(t.text)

                # set the tag_content as the tag's body
                extracted[tag] = tag_content

            # find the description given in the html (this is what google shows in search results)
            tag_content = []
            if soup.find("meta", {"name": "description"}):
                try:
                    tag_content.append(soup.find("meta", {"name": "description"})['content'])
                except KeyError:
                    pass
            extracted['meta_description'] = tag_content

            # keywords might be interesting too
            tag_content = []
            if soup.find("meta", {"name": "keywords"}):
                try:
                    tag_content.append(soup.find("meta", {"name": "keywords"})['content'])
                except KeyError:
                    pass
            extracted['meta_keywords'] = tag_content

            # pulling the <p> might be too much, so lets use boilerpipe to make a new boilerplate
            extractor = Extractor(extractor='ArticleExtractor', html=unicode(soup))
            extracted['summary'] = [extractor.getText()]

            # clean up
            for item in extracted:
                extracted[item] = map(clean, extracted[item])
                extracted[item] = filter(None, extracted[item])

            # put this line in the output
            print >>output, json.dumps(extracted)
            #print json.dumps(extracted)

def clean(incoming):
    incoming = unicode(incoming)
    incoming = unidecode(incoming)
    incoming = re.sub(r"\s+", ' ', incoming)
    incoming = incoming.strip()
    return incoming

def visible_text(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return ''
    result = re.sub('<!--.*-->|\r|\n', '', str(element), flags=re.DOTALL)
    result = re.sub('\s{2,}|&nbsp;', ' ', result)
    return result


if __name__ == "__main__":
    main()
