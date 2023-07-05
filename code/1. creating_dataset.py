#import relevant libraries
import pandas as pd
import os
import re

#CADEC Dataset


#Creating Datafarme 1 - store text and their ID
# assign directory
directory = 'Data/original/CADEC/CSIRO_DAP_Metadata/cadec/text/'

txt_id = []
text = []
# iterate over files in the directory
for filename in os.listdir(directory):
    
    #store the filename as the id of the text
    file = filename.replace(".txt", "")
    txt_id.append(file)
    
    file_open = directory + filename
    #open the text file and store the text
    with open(file_open) as f:
        contents = f.read()
        
    text.append(contents)
    

#create a dataframe that holds the filename as the text id as one column
#and the content of the text file as another column
d = {'txt_id': txt_id, 'text': text}
text_df = pd.DataFrame(data=d)


#save as a csv file
text_df.to_csv(r'Data/cleaned/text_cadec.csv', index=False)



#Creating Dataset 2 - text ID, ADR lines, and ADR (long)
# assign directory
directory = 'Data/original/CADEC/CSIRO_DAP_Metadata/cadec/original/'

count = 0
#loop through all the files
for filename in os.listdir(directory):
    
    count += 1
    file_open = directory + filename
    
    try:
        temp_df = pd.read_csv(file_open, sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)
    except:
        continue
    
    #add a column that would indicate the id
    file = filename.replace(".ann", "")
    #print(type(file))
    temp_df['txt_id'] = file

    #construct a dataframe with all the entries
    if count == 1:
        final_df = temp_df
    else:
        final_df = pd.concat([final_df, temp_df])


#separate some of the existing columns into new separated colums
final_df[['type', 'rest']] = final_df[2].str.split(' ', 1, expand=True)
final_df[['location', 'symptom']] = final_df['rest'].str.split('\t', 1, expand=True)
final_df[['start', 'end']] = final_df['location'].str.split(' ', 1, expand=True)

#keep only the ADR and anotator notes ones
final_df = final_df[(final_df['type'] == 'ADR')]
# keep only particular columns
final_df = final_df[[1,2,'type', 'symptom', 'start', 'end', 'txt_id']]

#rename columns
final_df = final_df.rename(columns={1: 'loc', 2: 'original_ann'})

#split the end column by spaces
final_df['end'] = final_df.apply(lambda x: str(x.end).split(" "), axis=1)
#keep only the last value of the list
final_df['end'] = final_df['end'].apply(lambda a: a[-1])
final_df['end'] = final_df['end'].astype(int)
#keep only the row with the max end if duplicates
final_df = final_df.groupby(['txt_id', 'start'], group_keys=False).apply(lambda x: x.loc[x.end.idxmax()]).reset_index(drop=True)

#merge back to the text to obtain the whole part with the adrs
merged_final = pd.merge(final_df, text_df, on = 'txt_id', how = 'left')
start_list = merged_final['start'].to_list()
end_list = merged_final['end'].to_list()
text_list = merged_final['text'].to_list()
new_sym = []

for i in range(len(text_list)):
    new_sym.append(text_list[i][int(start_list[i]):int(end_list[i])])
    
merged_final['check'] = new_sym

final_df['symptom'] = new_sym

#save as a csv file
final_df.to_csv(r'Data/cleaned/labeled_cadec.csv', index=False)


#PsyTar Dataset
#Creating Datafarme 1 - store text and their ID

psytar = pd.read_excel(open('Data/original/PsyTAR_dataset.xlsx', 'rb'),sheet_name='Sample') 

#combine the side effect and comment as a single comment (this is how the sentence labeling is done)
psytar['text'] = psytar['side-effect'] + ' ' + psytar['comment']

#keep only the relevant columns which are the id of the text and the text itself
psytar = psytar[['drug_id', 'text']]

#save as a csv file
psytar.to_csv(r'Data/cleaned/text_psytar.csv', index=False)


#Creating Dataset 2 - text ID, ADR lines, and ADR (long)
psytar_adr = pd.read_excel(open('Data/original/PsyTAR_dataset.xlsx', 'rb'),sheet_name='ADR_Identified')

psytar_adr["new_id"] = psytar_adr.index
#reshape DataFrame from wide format to long format
psytar_adr = pd.wide_to_long(psytar_adr, ["ADR"], i="new_id", j="number")

#drop all the NaN values for the ADR
psytar_adr = psytar_adr.dropna(subset=['ADR'])


mapping_id = pd.read_excel(open('Data/original/PsyTAR_dataset.xlsx', 'rb'),sheet_name='Sentence_Labeling')
#subset only the comment id and sentence id columns whcih we will use for mapping
mapping_id = mapping_id[['drug_id', 'comment_id']]


#map sentences with their corresponding comment ids in the psytar_adr
merged_df = psytar_adr.merge(mapping_id, on='drug_id', how='left')
merged_df = merged_df.drop_duplicates()

#merge back the comment/sentence based on the comment id
combined_df = merged_df.merge(psytar, on='drug_id', how='left')


combined_df['start'] = combined_df.apply(lambda x: str(x.text).find(str(x.ADR)), axis=1)
combined_df['length'] = combined_df.apply(lambda x: len(x.ADR), axis=1)
combined_df['end'] = combined_df['start'] + combined_df['length']
combined_df['ADRs'] = combined_df['ADR'].apply(str.lower)

#now we need to merge with the ADR based on a lexicon

#first let's load the data that contains this information
lexicon_adr = mapping_id = pd.read_excel(open('Data/original/PsyTAR_dataset.xlsx', 'rb'),sheet_name='ADR_Mapped')
lexicon_adr = lexicon_adr[['drug_id', 'sentence_index', 'ADRs', 'UMLS1']]

#merge the dataset we created so far with the ones with the lexicon ADR values
final_df = combined_df.merge(lexicon_adr, on=['drug_id', 'sentence_index', 'ADRs'], how='left')

final_df[['UMLS_code', 'rest']]= final_df['UMLS1'].str.split('/', 1, expand=True)
final_df[['ann_symptom', 'type']]= final_df['rest'].str.split('/', 1, expand=True)

#subset only the relevant columns
final_df = final_df[['drug_id', 'text', 'ADR', 'start','end','UMLS_code', 'ann_symptom']]

#rename columns
final_df = final_df.rename(columns={'drug_id': 'txt_id', 'ADR': 'symptom'})

#save as a csv file
final_df.to_csv(r'Data/cleaned/labeled_psytar.csv', index=False)




#Annotated Drug Review Dataset

#open the text file
with open('Data/original/annotated_drug_reviews.txt') as f:
        contents = f.read()

file1 = open('Data/original/annotated_drug_reviews.txt', 'r')
Lines = file1.readlines()

drug_reviews = []
for line in Lines:
    drug_reviews.append(line)


review_dict = {}

for review in drug_reviews:
    
    #split the annotated symptoms from the text itself
    split_r = review.split('|!|')
    
    review_dict[split_r[1].replace('\n', '')] = split_r[0].split('||')


#here we want to create one row per symptom
txt_id = []
text = []
symptom = []
id_t = 0

for key in review_dict:
    
    for sym in review_dict[key]:
        
        txt_id.append(id_t)
        text.append(key)
        res = re.sub(r'[^\w\s]', '', sym)
        symptom.append(res)
    
    id_t += 1

#create a dataframe
d = {'txt_id': txt_id, 'text': text, 'symptom': symptom}
ann_df = pd.DataFrame(data=d)

#find the position of the side effect in the review
ann_df['start'] = ann_df.apply(lambda x: str(x.text).find(str(x.symptom)), axis=1)
ann_df['length'] = ann_df.apply(lambda x: len(x.symptom), axis=1)
ann_df['end'] = ann_df['start'] + ann_df['length']

#save as a csv file
ann_df.to_csv(r'Data/cleaned/labeled_annotated.csv', index=False)


#Creating Datafarme 1 - store text and their ID
#keep only particulat columns
text_df = ann_df[['txt_id', 'text']]
text_df = text_df.drop_duplicates()

#save as a csv file
text_df.to_csv(r'Data/cleaned/text_annotated.csv', index=False)







#Combine all datasets into a single dataset
#Datafarme 1 - store text and their ID

#load the CAdec dataset
cadec_df_1 = pd.read_csv(r'Data/cleaned/text_cadec.csv')
cadec_df_1['dataset'] = 'CADEC'

#load the PsyTar dataset
psytar = pd.read_csv(r'Data/cleaned/text_psytar.csv')
psytar['dataset'] = 'PsyTar'
psytar = psytar.rename(columns={"drug_id": "txt_id"})

#load the annotated drug review dataset
drug_ann = pd.read_csv(r'Data/cleaned/text_annotated.csv')
drug_ann['dataset'] = 'Annotated_dataset'
drug_ann['txt_id'] = drug_ann['txt_id'].astype(str)

#combine all datasets into a single dataset
pieces = (cadec_df_1,psytar,drug_ann)
combined_df_1 =  pd.concat(pieces, ignore_index = True)
#replace \n and \t from the text
combined_df_1['text'] = combined_df_1.apply(lambda x: str(x.text).replace("\n", " "), axis = 1)
combined_df_1['text'] = combined_df_1.apply(lambda x: str(x.text).replace("\t", " "), axis = 1)


#save the combined dataset
combined_df_1.to_csv(r'Data/combined/combined_df_1.csv', index=False)

#Datafarme 2 - text ID, ADR lines, and ADR (long)

#load the cadec dataset
cadec_df_2 = pd.read_csv(r'Data/cleaned/labeled_cadec.csv')
cadec_df_2['dataframe'] = 'CADEC'
cadec_df_2 = cadec_df_2[['symptom', 'txt_id', 'dataframe', 'start', 'end']]

#load the psytar dataset
psytar_df_2 = pd.read_csv(r'Data/cleaned/labeled_psytar.csv')
psytar_df_2['dataframe'] = 'PsyTar'
psytar_df_2 = psytar_df_2[['txt_id', 'symptom', 'start', 'end', 'dataframe']]

#load the annotated drug review dataset
drug_ann_df_2 = pd.read_csv(r'Data/cleaned/labeled_annotated.csv')
drug_ann_df_2['dataframe'] = 'Annotated_dataset'
drug_ann_df_2 = drug_ann_df_2[['txt_id', 'symptom', 'start', 'end', 'dataframe']]

#combine all the dataframes
pieces = (cadec_df_2,psytar_df_2,drug_ann_df_2)
combined_df_2 =  pd.concat(pieces, ignore_index = True)

#save the combined dataset
combined_df_2.to_csv(r'Data/combined/combined_df_2.csv', index=False)



