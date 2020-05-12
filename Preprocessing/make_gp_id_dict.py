import csv
import re
import pickle
frame_size = 1024

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# make dictionary of id's and gender
gender_dict = {}
with open('gender_id.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[1]:
            gender_dict[rows[0]] = rows[1]

print(len(gender_dict))
#print('csv.field_size_limit: ',csv.field_size_limit)

#make dictionary of id's and personality
with open('big5.csv', mode='r') as infile:
    reader = csv.reader(infile)
    personality_dict = {rows[0]:rows[1:6] for rows in reader}

print(len(personality_dict))


#combine dictionaries
gp_dict = {}
for user_id in gender_dict:
    if user_id in personality_dict:
         gp_dict[user_id] = [gender_dict[user_id]] + personality_dict[user_id]

save_obj(gp_dict,'gp_dict')

print(len(gp_dict))

