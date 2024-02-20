import re

def extractStrings(line):
    pattern = re.compile(r'\s+|\t+')
    strings = re.split(pattern, line)
    strings = [s for s in strings if s]
    return strings

def correctFormat(string1, string2):
    return f"{string1} {string2}"

def rewriteSmilesFile(input_file_path, output_file_path):
    MAX_LINES = 1000000
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        input_lines = input_file.readlines()
        if len(input_lines) > MAX_LINES:
            print("Maximum lines exceeded.")
            for line in input_lines[:MAX_LINES]:
                strings = extractStrings(line)
                
                if len(strings) == 2:
                    corrected_line = correctFormat(strings[0], strings[1])
                    output_file.write(corrected_line + '\n')            
        else:
            for line in input_lines:
                strings = extractStrings(line)
                
                if len(strings) == 2:
                    corrected_line = correctFormat(strings[0], strings[1])
                    output_file.write(corrected_line + '\n')