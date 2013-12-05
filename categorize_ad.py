#!/usr/bin/env python
from __future__ import print_function
from alchemyapi import AlchemyAPI
import json

alchemyapi = AlchemyAPI()

p = open('uncategorized_new','rb')
n = open('categorized_new_test','w') 

q = p.read()

q = q.split('\n')


for i in range (800,1600):

	url = q[i].split(" ")[0]
	url_id = q[i].split(" ")[1]

	print(i,": ",url,' ',url_id)

	try:
		response = alchemyapi.category('url',url)

		if response['status'] == 'OK':
			print('## Response Object ##')
			print(json.dumps(response, indent=4))

			n.write(url+'\t'+response['category']+'\t'+response['score']+'\t'+url_id+'\n')
			print('')
			print('## Category ##')
			print('text: ', response['category'])
			print('score: ', response['score'])
			print('')
		
		else:
			print('Error in text categorization call: ', response['statusInfo'])
	except Exception:
		continue

p.close()
n.close()