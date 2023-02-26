from lib2to3.pgen2.tokenize import tokenize
import random 
from string import ascii_letters, punctuation, digits
import re
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

def tokenizer_check_if_text_too_long(text, tokenizer, max_length):
    data = tokenizer.batch_encode_plus([text],max_length=max_length,truncation=True,return_overflowing_tokens=True )    
    if len(data["input_ids"]) > 1:
        return True
    else:
        return False#, len(data["input_ids"][0])

def delete_characters(text, char_delete_percentage=0.005):
    modifyed_line = []   
    for char in text:
        if random.random() > char_delete_percentage or char in digits:
            modifyed_line.append(char)
    return "".join(modifyed_line)

def insert_characters(text, augmentation_probability=0.01):
    modifyed_line = []   
    for char in text:
        if random.random() <= augmentation_probability and char not in digits:            
            modifyed_line.append(random.choice(ascii_letters))
        modifyed_line.append(char)
    return "".join(modifyed_line)

def replace_characters(text, augmentation_probability=0.01):
    modifyed_line = []   
    for char in text:
        if random.random() <= augmentation_probability and char not in digits:            
            modifyed_line.append(random.choice(ascii_letters))
        else:
            modifyed_line.append(char)
    return "".join(modifyed_line)

def swap_characters_case(text, augmentation_probability=0.01):
    modifyed_line = []   
    for char in text:
        if random.random() <= augmentation_probability:            
            char = char.swapcase()
        modifyed_line.append(char)
    return "".join(modifyed_line)

def lower_case_words(text, augmentation_probability=0.05):
    modifyed_line = []   
    for word in text.split():
        if word[0].islower() == False and random.random() <= augmentation_probability:            
            word = word.lower()
        modifyed_line.append(word)
    return " ".join(modifyed_line)


clean_chars = re.compile(r'[^A-Za-zöäüÖÄÜß,.!?’\'$%€0-9\(\)\- ]', re.MULTILINE)
def cleanup(text):    
    text = clean_chars.sub('', text)
    #print("bug: somehow all numbers are removed - this is might be due to this regex")
    #exit()
    #text = text.replace("\n", "")
    #text = text.replace('"','\\"')
    return text

clean_punctuation = re.compile(r"(?<!\d)[.,;:'?!](?!\d)")
def remove_punctuation(text):
    """Remove all punctuation from string, except if it's between digits"""
    return clean_punctuation.sub("", text)

def combine_sentences(text, sentences, augmentation_probability = 1):
    if random.random() < augmentation_probability:
        sentences_to_sample = random.randint(0,10)
        augmentation_sentences = random.sample(sentences,sentences_to_sample)    
        return text + " " + " ".join(augmentation_sentences)
    else:
        return text

def delete_word(text, augmentation_probability = 0.001):        
    if random.random() < augmentation_probability:
        words = text.split()
        if len(words) < 3:
            # do not delete word in short text, as there will be no context to guess the word
            return text
        word_to_remove = random.randint(0,len(words)-1)
        words.pop(word_to_remove)
        return " ".join(words)
    else:
        return text
#=========================================================================
import unidecode
import numpy as np

chars_regrex = '[aàảãáạăằẳẵắặâầẩẫấậAÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬoòỏõóọôồổỗốộơờởỡớợOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢeèẻẽéẹêềểễếệEÈẺẼÉẸÊỀỂỄẾỆuùủũúụưừửữứựUÙỦŨÚỤƯỪỬỮỨỰiìỉĩíịIÌỈĨÍỊyỳỷỹýỵYỲỶỸÝỴnNvVmMCG]'
same_chars = {
    'a': ['á', 'à', 'ả', 'ã', 'ạ', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ'],
    'A': ['Á','À','Ả','Ã','Ạ','Ấ','Ầ','Ẩ','Ẫ','Ậ','Ắ','Ằ','Ẳ','Ẵ','Ặ'],
    'O': ['Ó','Ò','Ỏ','Õ','Ọ','Ô','Ố','Ồ','Ổ','Ỗ','Ộ','Ơ','Ớ','Ờ','Ở','Ỡ','Ợ','Q'],
    'o': ['ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ','ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'q'],
    'e': ['é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ê'],
    'E': ['É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ê'],
    'u': ['ú', 'ù', 'ủ', 'ũ', 'ụ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ư'],
    'U': ['Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Ư'],
    'i': ['í', 'ì', 'ỉ', 'ĩ', 'ị'],
    'I': ['Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị'],
    'y': ['ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'v'],
    'Y': ['Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ', 'V'],
    'n': ['m'],
    'N': ['N'],
    'v': ['y'],
    'V': ['Y'],
    'm': ['n'],
    'M': ['N'],
    'C': ['G'],
    'G': ['C']
}
def _char_regrex(text):
    match_chars = re.findall(chars_regrex, text)
    return match_chars
def _random_replace(text, match_chars):
    replace_char = match_chars[np.random.randint(low=0, high=len(match_chars), size=1)[0]]
    insert_chars = same_chars[unidecode.unidecode(replace_char)]
    insert_char = insert_chars[np.random.randint(low=0, high=len(insert_chars), size=1)[0]]
    text = text.replace(replace_char, insert_char, 1)

    return text
def change(text):
    match_chars = _char_regrex(text)
    if len(match_chars) == 0:
        return text

    text = _random_replace(text, match_chars)

    return text
def replace_accent_chars(text, ratio=0.01):
    words = text.split()
    mask = np.random.random(size=len(words)) < ratio

    for i in range(len(words)):
        if mask[i]:
            words[i] = change(words[i])
            break          

    return ' '.join(words)

def remove_random_accent(text, ratio=0.05):
        words = text.split()
        mask = np.random.random(size=len(words)) < ratio
        
        for i in range(len(words)):
            if mask[i]:
                words[i] = unidecode.unidecode(words[i])
                break

        return ' '.join(words)

# Space between words
def remove_random_space(text):
    words = text.split()
    n_words = len(words)
    start = np.random.randint(low=0, high=n_words, size=1)[0]

    if start + 3 < n_words:
        end = np.random.randint(low=start, high=start + 3, size=1)[0]
    else:
        end = np.random.randint(low=start, high=n_words, size=1)[0]

    out = ' '.join(words[:start])  + ' ' + ''.join(words[start:end + 1]) + ' ' + ' '.join(words[end + 1:])

    return out.strip()



data_file = "/content/Vietnamese-Corrector/data/data50k.vi.txt" #"data/en.wikidump.processed.24m.txt" #
language = "vi"
num_lines = sum(1 for line in open(data_file, 'r'))

with open(data_file, 'r') as file:
    sentences = file.readlines()

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small") # for vi
#xlm-roberta-base
#facebook/bart-base
#, encoding="utf-8"
count = 0
datas = []
with open(language+".csv", "w") as output:        
    with open(data_file,'r') as file:
        for line in tqdm(file, total=num_lines):
            # if count == 0:
            #   count += 1
            #   continue
            #line = cleanup(line)
            if len(line) < 1:
                continue 

            line = line.replace('"', '')
            
            #line = combine_sentences(line,sentences)
            # print()
            # print(len(line), line)              
            if tokenizer_check_if_text_too_long(line,tokenizer,max_length=1024):
                #print(f"skipping line as its too long ({len(line)}):\n"+line)
                continue
            
            if random.random() > 0.02:
                # we will leave 2% of the data untouched, to teach the 
                # model, not to "overact" on the texts
                # my custom
                new_line = remove_random_space(line)
                new_line = remove_random_accent(new_line)
                new_line = replace_accent_chars(new_line)

                # original
                new_line = swap_characters_case(new_line)  
                new_line = delete_word(new_line)                    
                new_line = delete_characters(new_line)
                new_line = insert_characters(new_line)
                new_line = replace_characters(new_line)
                new_line = lower_case_words(new_line)                                           
                new_line = remove_punctuation(new_line)
            else:
                new_line = line          
            # print(len(new_line), new_line)
            # count += 1
            # break
            # if count == 1:
            #     break
            datas.append([new_line.strip(), line.strip()])
            #count += 1
            # if count == 10:
            #   break
            output.write(f'"{new_line.strip()}","{line.strip()}"\n')        
os.system(f"echo \"text,summary\" > {language}.train.csv")
num_lines = sum(1 for line in open(f"{language}.csv",'r'))
os.system(f"head -n {num_lines-2000} {language}.csv >> {language}.train.csv")
os.system(f"echo \"text,summary\" > {language}.test.csv")
os.system(f"tail -n 2000 {language}.csv >> {language}.test.csv")