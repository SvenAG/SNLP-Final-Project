#!/usr/bin/env python
from __future__ import print_function
from alchemyapi import AlchemyAPI
import json

alchemyapi = AlchemyAPI()

p = open('uncategorized_new','rb')
n = open('categorized_new','w') 

q = p.read()

q = q.split('\n')


for i in range (0,800):

	url = q[i].split(" ")[0]

	print(i,": ",url)

	try:
		response = alchemyapi.category('url',url)

		if response['status'] == 'OK':
			print('## Response Object ##')
			print(json.dumps(response, indent=4))

			n.write(response['url']+'\t'+response['category']+'\t'+response['score']+'\n')
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