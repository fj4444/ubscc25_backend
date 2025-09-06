import re

def mirror_words_r(x):
    return ' '.join(word[::-1] for word in x.split())
def encode_mirror_alphabet_r(x):
    ans = list()
    for c in x:
        if ord('a') <= ord(c) <= ord('z'):
            num = ord(c) - ord('a')
            ans.append(chr(ord('a') + 25 - num))
        elif ord('A') <= ord(c) <= ord('Z'):
            num = ord(c) - ord('A')
            ans.append(chr(ord('A') + 25 - num))
        else:
            ans.append(c)
    return ''.join(ans)
def toggle_case_r(x):
    ans = list()
    for c in x:
        if ord('a') <= ord(c) <= ord('z'):
            num = ord(c) - ord('a')
            ans.append(chr(ord('A') + num))
        elif ord('A') <= ord(c) <= ord('Z'):
            num = ord(c) - ord('A')
            ans.append(chr(ord('a') + num))
        else:
            ans.append(c)
    return ''.join(ans)
def swap_pairs_r(x):
    ans = list()
    for word in x.split():
        for i in range(len(word) // 2):
            ans.append(word[2 * i + 1])
            ans.append(word[2 * i])
        ans.append(' ')
    return ''.join(ans[:-1])
def encode_index_parity_r(x):
    ans = list()
    for word in x.split():
        mp = len(word) // 2
        word1 = word[:len(word) - mp]
        word2 = word[len(word) - mp:]
        temp = ['' for i in range(len(word))]
        temp[::2] = word1
        temp[1::2] = word2
        ans = ans + temp + [' ']
    return ''.join(ans[:-1])
def double_consonants_r(x):
    consonants = [
        'b', 'B', 'c', 'C', 'd', 'D', 'f', 'F', 'g', 'G', 
        'h', 'H', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 
        'n', 'N', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 
        't', 'T', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 
        'z', 'Z'
    ]
    prev = 'a'
    ans = list()
    for c in x:
        if c == prev and c in consonants:
            prev = 'a'
            continue
        else:
            prev = c
            ans.append(c)
    return ''.join(ans)

def reverse(redacted, transformations):
    function_names = re.findall(r'(\w+)\(', transformations)
    for f in reversed(function_names):
        if f == "mirror_words":
            redacted = mirror_words_r(redacted)
        if f == "encode_mirror_alphabet":
            redacted = encode_mirror_alphabet_r(redacted)
        if f == "toggle_case":
            redacted = toggle_case_r(redacted)
        if f == "swap_pairs":
            redacted = swap_pairs_r(redacted)
        if f == "encode_index_parity":
            redacted = encode_index_parity_r(redacted)
        if f == "double_consonants":
            redacted = double_consonants(redacted)
    return redacted

def final_firewall(json_data):
    redacted = json_data["challenge_one"]["transformed_encrypted_word"]
    transformations = json_data["challenge_one"]["transformations"]
    arr = reverse(redacted, transformations)
    ans = dict()
    ans["challenge_one"] = arr
    ans["challenge_two"] = 3
    ans["challenge_three"] = ""
    ans["challenge_four"] = ""
    return ans 