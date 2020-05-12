import csv, string, re, pickle
csv_max = 131072; frame_size = 1024;
pattern = re.compile('[^a-zA-Z 0-9`~!@#$%&^*()+=;\'\\:",.?]+')
framesize = 1024
#from langdetect import detect

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#grab the dictionary of user_id and gender from file
label_dict = load_obj('gp_train_dict')
file_in = open('user_status_sorted.csv', 'r',encoding = "ISO-8859-1")
id_stat_data = csv.reader(file_in)
joined_data = open('gps_train.csv', 'w')
csv_out = csv.writer(joined_data, delimiter = ',',quotechar = '"')

n = num_chunks = errors =  0
status=id_cur = ''

for row in id_stat_data:
    if(label_dict.get(row[0]) is not None): #make sure we have the label
	#if pred_lang(row[0]) == 'en':
        if(id_cur == ''): id_cur = row[0]
        if (row[0]==id_cur and len(status + row[2])<framesize):
            status += '|' + pattern.sub('', row[2])
        else:
            #print(label_dict.get(id_cur),type(label_dict.get(id_cur)))
            #print([status.lower()],type([status.lower()]))
            csv_out.writerow(label_dict.get(id_cur)+[status.lower()])
            status = pattern.sub('', row[2])
            id_cur = row[0]
            num_chunks +=1
    else: errors += 1
    n+=1
    if n%10000==0: print(n)#, row, type(row), type(row[0]))

print('Number of missing labels: ', errors)
print ('number of users: ', num_chunks)


file_in.close()
joined_data.close()
