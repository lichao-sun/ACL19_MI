#-*- coding=utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import json
from pprint import pprint

count = 0
flag = 0
count_t = 0
question = {}
select = {}
with open('userLogs.json', 'r') as file:
    for line in file:
        j_line = json.loads(line)
        if j_line['a_type'] != unicode('建议问'):
            temp_q = j_line['q_user']
            temp_id = j_line['user_id']
        if j_line['a_type'] == unicode('建议问'):
            count_t += 1
            #print 'last', temp_q, temp_id
            empty = ''
            if temp_q in j_line['a_suggestion']:
                if temp_q not in select:
                    select[temp_q] = 1
                else:
                    select[temp_q] += 1
                #print j_line['a_type']
                count += 1
                empty += str(j_line['user_id']) + ' '
                empty += str(j_line['q_user']) + ' '
                empty += '['
                for a in j_line['a_suggestion']:
                    if a not in question:
                        question[a] = 1
                    else:
                        question[a] += 1
                    empty += str(a) + ','
                empty += ']'
                empty += str(temp_q)
                print empty
    # count3 = 0
    # for key, value in sorted(select.iteritems(), key=lambda (k,v): (v,k)):
    #     if value >= 10:
    #         count3 += 1
    #         print "%s: %s" % (key, value)
    # print count, count_t, count3
        #     flag = 1
        # elif flag == 1:
        #     print j_line['a_type']
        #     for key, value in j_line.items():
        #         if key == 'user_id':
        #             print key, value
        #         if key == 'q_user':
        #             print key, value
        #         if key == 'q_faq':
        #             print key, value
        #     flag = 0
        #     if temp_id == j_line['user_id'] and j_line['q_user'] in temp_suggestion:
        #         print 'Found'

            # if sizeof(value) > 1:
            #     for a in j_line[key]:
            #         print a
            # else:
            #     print value
        #result = "".join(str(key) + str(value) for key, value in j_line.items())
        #print result
        # for key
        # #pprint(j_line)
        # print j_line['user_id']
        # print 'q', j_line['q_faq']
        # try:
        #     for a in j_line['a_suggestion']:
        #         print a
        # except:
        #     pass