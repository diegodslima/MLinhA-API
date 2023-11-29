import re

def extractStrings(line):
    pattern = re.compile(r'\s+|\t+')
    strings = re.split(pattern, line)
    strings = [s for s in strings if s]
    return strings

def correctFormat(string1, string2):
    return f"{string1} {string2}"

def rewriteSmilesFile(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            strings = extractStrings(line)
            
            if len(strings) == 2:
                corrected_line = correctFormat(strings[0], strings[1])
                output_file.write(corrected_line + '\n')