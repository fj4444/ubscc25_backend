import json

def roman_to_int(s):
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    n = len(s)
    for i in range(n):
        if i < n - 1 and roman_map[s[i]] < roman_map[s[i+1]]:
            total -= roman_map[s[i]]
        else:
            total += roman_map[s[i]]
    return total

def sort_numerals_1(nums):
    converted = []
    for num_str in nums:
        # Check if it is a Roman numeral: all characters in "IVXLCDM"
        if all(c in "IVXLCDM" for c in num_str):
            value = roman_to_int(num_str)
        else:
            value = int(num_str)
        converted.append(value)
    
    converted.sort()
    return converted

def parse_english(s):
    # Predefine a dictionary of English number words
    words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100, "thousand": 1000, "million": 1000000, "hundreds": 100, "thousands": 1000, "millions": 1000000
    }
    
    # Split the string into tokens by spaces? But if no spaces, we need to segment.
    # Since the problem might have no spaces, we will try to segment the string.
    # This is complex. Alternatively, we can require that the input has spaces.
    # For the sake of the challenge, we assume that the input is given with spaces.
    tokens = s.split()
    total = 0
    current = 0
    for token in tokens:
        if token not in words:
            return None  # invalid
        value = words[token]
        if value in [100, 1000, 1000000]:
            if current == 0:
                current = 1
            total += current * value
            current = 0
        else:
            current += value
    total += current
    return total

def parse_chinese(s):
    # Digits
    digits = {
        "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9
    }
    # Multipliers
    multipliers = {
        "十": 10, "百": 100, "千": 1000, "万": 10000, "萬": 10000, "億": 100000000, "亿": 100000000
    }
    
    total = 0
    current = 0
    for char in s:
        if char in digits:
            current = digits[char]
        elif char in multipliers:
            multiplier = multipliers[char]
            if current == 0:
                current = 1
            total += current * multiplier
            current = 0
        else:
            return None  # invalid
    total += current
    return total

def parse_german(s):
    # Dictionary of German number words
    german_words = {
        "null": 0,
        "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
        "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
        "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90,
        "hundert": 100, "tausend": 1000, "million": 1000000
    }
    
    # Remove any spaces and convert to lowercase for consistency
    s = s.replace(" ", "").lower()
    tokens = []
    i = 0
    n = len(s)
    # Segment the string into tokens by matching the longest words
    while i < n:
        found = False
        for j in range(n, i, -1):
            word = s[i:j]
            if word in german_words:
                tokens.append(word)
                i = j
                found = True
                break
        if not found:
            # If no word found, skip one character (might be 'und' or invalid)
            # Check for 'und' which is not in the dictionary but is used in compounds
            if s[i:i+3] == "und":
                tokens.append("und")
                i += 3
            else:
                i += 1
    # Now parse the tokens
    total = 0
    current = 0
    for token in tokens:
        if token == "und":
            continue  # 'und' is used for inversion, handled by the order
        value = german_words[token]
        if value == 100:
            if current == 0:
                current = 1
            current *= value
        elif value == 1000 or value == 1000000:
            if current == 0:
                current = 1
            total += current * value
            current = 0
        elif value >= 20 and value <= 90:  # tens
            # In German, units come before tens, e.g., "fünfundzwanzig" -> 5 + 20
            # So if there is a current value, it is the unit to be added to the ten
            current = value + current
        else:
            current += value
    total += current
    return total

def get_value_and_priority(s):
    # try Roman
    if all(c in "IVXLCDM" for c in s):
        return (roman_to_int(s), 0)
    if s.isdigit():
        return (int(s), 5)
    # try English
    value_eng = parse_english(s)
    if value_eng is not None:
        return (value_eng, 1)
    # try Traditional Chinese
    if not ("万" in s or "亿" in s):
        value_trad = parse_chinese(s)
        if value_trad is not None:
            return (value_trad, 2)
    # try Simplified Chinese
    value_simp = parse_chinese(s)
    if value_simp is not None:
        return (value_simp, 3)
    # try German
    value_ger = parse_german(s)
    if value_ger is not None:
        return (value_ger, 4)
    return (0,5)

def sort_numerals_2(nums):
    converted = []
    for num_str in nums:
        value, priority = get_value_and_priority(num_str)
        converted.append((value, priority, num_str))
    converted.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in converted]

def process(json_data):
    partName = json_data["part"]
    if partName == "ONE":
        nums = json_data["challengeInput"]["unsortedList"]
        py_results = [str(i) for i in sort_numerals_1(nums)]
        result = {"sortedList": py_results}
        json_output = json.dumps(result, indent=2)
        return json_output
    else:
        nums = json_data["challengeInput"]["unsortedList"]
        py_results = sort_numerals_2(nums)
        result = {"sortedList": py_results}
        # json_output = json.dumps(result, indent=2)
        # return json_output
        return result
    return None