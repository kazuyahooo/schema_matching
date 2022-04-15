import os,csv
import re, difflib
import pandas as pd
import translators as ts
from collections import Counter, defaultdict

# Mapping set
ans = defaultdict(list)
for i in range(1,16):
    with open(file_path + str(i) + '/mapping.txt', encoding='utf-8-sig') as f:
        for row in f:
            ans[i].append(row.strip('\n'))

# First Step
file_path = './Data/pair_'
test = dict()
result =  dict()
for num in range(1, 16):
    columns = list() # 抓取兩個table的column
    test[num] = list()
    result[num] = list()
    for i in ['1','2']:
        with open(file_path + str(num) + '/Table' + i + '.csv', encoding='utf-8-sig') as f:
            rows = csv.reader(f)
            for row in rows:
                columns.append(row)
                print(row)
                break
    same = list(set(columns[0]).intersection(columns[1]))
    result[num].extend(['<'+word+', '+word+'>' for word in same])
    # print(same)
    # print('')
    test[num].append([elm for elm in columns[0] if elm not in same + ['']])
    test[num].append([elm for elm in columns[1] if elm not in same + ['']])
    
# Second Step
#將中文轉成英文比對embedding相似度，高於0.7算match到
tables = ['/Table1.csv', '/Table2.csv']
nlp = spacy.load("en_core_web_lg")
nlp_ch = spacy.load("zh_core_web_lg")

for i in range(1,16):
#for i in range(1,2):
    table1 = pd.read_csv(file_path + str(i) + tables[0],low_memory=False)
    table2 = pd.read_csv(file_path + str(i) + tables[1],low_memory=False)
    #只留下還要比對的欄位名稱
    table1 = table1[test[i][0]]
    table2 = table2[test[i][1]]
    #去除欄位名稱有_的，增加配對機率
    trans_tab1 = [ts.google(j).replace('_',' ') for j in table1.columns]
    trans_tab2 = [ts.google(j).replace('_',' ') for j in table2.columns]
    mydict = defaultdict(list)
    simatrix = [list(table2.columns)]
    for idx1, elm1 in enumerate(trans_tab1):
        elm1 = nlp(elm1)
        max_cosine = -1
        # temp = []
        temp2 = []
        for idx2, elm2 in enumerate(trans_tab2):
            elm2 = nlp(elm2)
            if elm1.similarity(elm2) > max_cosine:
                max_cosine = elm1.similarity(elm2)
                temp = [max_cosine, test[i][0][idx1], test[i][1][idx2]]
            temp2.append(elm1.similarity(elm2))
        simatrix.append(temp2)
        if temp[0] >= 0.7: #temp[0]紀錄table1的欄位比對table2欄位最高相似度有無超過0.7
            mydict[temp[2]].append((temp[1], temp[0]))
            print(temp)
    for key, val in mydict.items():
        if len(val) >= 2:
            val = sorted(val, key = lambda x: x[1], reverse = True)
        result[i].append('<'+val[0][0]+', '+key+'>')
        # 刪除此階段配對好的值
        test[i][0].remove(val[0][0])
        test[i][1].remove(key)
    print('')
    
# Third Step
# 比較欄位內容
tables = ['/Table1.csv', '/Table2.csv']
nlp = spacy.load("en_core_web_lg")
nlp_ch = spacy.load("zh_core_web_lg")
clear_col = ['Picture', 'URL', 'Date', 'String']
for i in range(4,16):
#for i in range(6,7):
    table1 = pd.read_csv(file_path + str(i) + tables[0],low_memory=False)
    table2 = pd.read_csv(file_path + str(i) + tables[1],low_memory=False)
    table1 = table1[test[i][0]]
    table2 = table2[test[i][1]]
    table1.dropna(how = 'all', inplace = True) # 清除整條row為空值的測資
    table2.dropna(how = 'all', inplace = True) # 清除整條row為空值的測資
    temp1 = list()
    temp2 = list()
    table = [table1, table2]
    for num in range(0,2):
        for column in table[num].columns:
            temp3 = Counter() #儲存數字、浮點數、NER結果
            fixed_pattern = Counter() #希望找出固定模式的字串
            for j in table[num][column][:10]: # 只跑十筆降低運算時間
                if pd.isna(j) : continue # 空值跳過
                j = str(j) if type(j) != str else j
                pic = ['.jpg', '.jpeg', '.png', '.gif']
                if j[-4:] in pic: # 欄位為圖片
                    j = 'Picture'
                    break
                if j.startswith('http://') or j.startswith('https://'): # 欄位為網址
                    j = "URL"
                    break
                try:
                    pd.to_datetime(j) # 欄位為時間
                    j = "Date"
                    break
                except:
                    pass
                trans = ts.google(j)
                rule = re.compile(r'[^,\s\d]*') # 去除空白及數字
                res = ''.join(rule.findall(trans))
                res_ch = ''.join(rule.findall(j))
                if res == '': temp3['Number'] += 1 # 欄位內容為純數字
                elif res == '.': temp3['Float'] += 1 # 欄位內容為浮點數
                else: 
                    res_ch = 'String' if len(res_ch) >= 8 else res_ch # 長度大於八判定為字串
                    fixed_pattern[res_ch] += 1 # 找出固定pattern
                if len(trans) == 1:
                    for ent in nlp(res).ents:
                        temp3[ent.label_] += 1 # 以NER找出該欄位特徵
            # print(temp3)
            # print(fixed_pattern)
            if j in clear_col:
                column_val = j
            elif not temp3:
                column_val = fixed_pattern.most_common()[0][0]
            elif not fixed_pattern:
                column_val = temp3.most_common()[0][0]
            elif temp3 and fixed_pattern:
                temp3_max = temp3.most_common()[0]
                fixed_max = fixed_pattern.most_common()[0]
                if temp3_max[1] > fixed_max[1]:
                    column_val = temp3_max[0]
                else: 
                    column_val = fixed_max[0]
            if num == 0: temp1.append((column, column_val))
            else: temp2.append((column, column_val))
    # print(temp1)
    # print(temp2)
    str_cnt = 0
    for col2 in temp2: 
        if col2[1] == 'String':
            str_cnt += 1
    mydict = defaultdict(list)
    for col1 in temp1:
        for col2 in temp2:
            check_ch_col = difflib.SequenceMatcher(None, col1[0], col2[0]).quick_ratio()
            check_en_col = difflib.SequenceMatcher(None, ts.google(col1[0]), ts.google(col2[0])).quick_ratio()
            if col1[1] == col2[1]:
                if col1[1] in clear_col: # 須符合以下欄位名稱條件才能納入配對
                    if col1[1] != 'String' or (col1[1] == "String" and str_cnt == 1):
                        mydict[col2[0]].append((col1[0], 1.0))
                else: # 避免欄位內容相同情況為- 無 否 是 等詞
                    if len(col1[1])>=2:
                        mydict[col2[0]].append((col1[0], 1.0))
                    if check_ch_col >= 0.5 or check_en_col >= 0.5:
                        mydict[col2[0]].append((col1[0], max(check_ch_col, check_en_col)))
            else: # 判斷是否為欄位內容格式相似
                check_ch = difflib.SequenceMatcher(None, col1[1], col2[1]).quick_ratio()
                check_en = difflib.SequenceMatcher(None, ts.google(col1[1]), ts.google(col2[1])).quick_ratio()
                if check_ch >= 0.7 or check_en >= 0.7:
                    mydict[col2[0]].append((col1[0], max(check_ch_col, check_en_col)))
    
    for key, val in mydict.items():
        if len(val) >= 2:
            val = sorted(val, key = lambda x: x[1], reverse = True)
        result[i].append('<'+val[0][0]+', '+key+'>')
        # 刪除此階段配對好的值
        if val[0][0] in test[i][1]:
            test[i][0].remove(val[0][0])
        if key in test[i][1]:
            test[i][1].remove(key)
    
#查看最終配對正確率及數量
for i in range(1,16):
    count = 0
    for elm in ans[i]:
        if elm in result[i]:
            count += 1
    print("pari_"+str(i), count, len(ans[i]), "{:.2f}".format(count/len(ans[i])*100))
    
#輸出結果
result_dst = './Result/result_'
for i in range(1,16):
    with open(result_dst + str(i) + '.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pairs in result[i]:
                print(pairs)
                writer.writerow([pairs])
    