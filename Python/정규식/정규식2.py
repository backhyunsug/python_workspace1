import re

print(re.search(r"[a]", "a"))
print(re.search(r"[a]", "ba"))
print(re.search(r"[a]", "abb"))
print(re.search(r"[a]", "cb"))

print("-----------------------------------")
print(re.search(r"[abc]", "apple"))     # ✅ a → match
print(re.search(r"[abc]", "dog"))       # ❌ no match
print(re.search(r"[0-9]", "2024year"))  # ✅ 2 → match
print(re.search(r"[0-9]", "year"))      # ❌ no match
print(re.search(r"[A-Z]", "Hello"))     # ✅ H → match
print(re.search(r"[A-Z]", "hello"))     # ❌ no match
print(re.search(r"[a-z]", "Hello"))     # ✅ e → match
print(re.search(r"[A-Za-z]", "123Abc")) # ✅ A → match

print("-----------------------------------")
print(re.search(r"[^a]", "aaa"))    # ❌ 모두 a → no match a를 제외한 문자가 하나라도 있으면
print(re.search(r"[^a]", "aab"))    # ✅ b → match
print(re.search(r"[^0-9]", "1234a"))# ✅ a → match

print(re.search(r"[a-z]{3}", "abc"))    # ✅ abc → match
print(re.search(r"[a-z]{3}", "a1c"))    # ❌ 1 때문에 실패
print(re.search(r"[A-Za-z]{2,5}", "Hi"))# ✅ Hi → match

print(re.search(r"[0-9][a-z]", "3b"))     # ✅ 3b
print(re.search(r"[0-9][a-z]", "a3b"))    # ✅ 3b (3b에서 매치됨)
print(re.search(r"[0-9][a-z]", "ab"))     # ❌ no match, 숫자가 포함되어야 한다

print("===========================")
print( re.search(r"^abc", "abcde"))
print( re.search(r"^abc", "abc")) 
print(re.search(r"^abc", "abc12"))
print(re.search(r"^abc",  "aabc12")) #match안됨 

print(re.search(r"abc$",  "abc"))
print(re.search(r"abc$",  "12abc"))
print(re.search(r"abc$",  "ea abc"))

print("===========================")
print(re.search(r"[p|P]ython",  "python"))
print(re.search(r"[p|P]ython",  "Python"))
print(re.search(r"[p|P]ython",  "PYTHON"))

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(re.search(r"[^abc]",  "a")) #abc 제외한 문자가 있으면
print(re.search(r"[^abc]",  "b"))
print(re.search(r"[^abc]",  "c"))
print(re.search(r"[^abc]",  "aajdb")) #j
print(re.search(r"[^abc]",  "dfc"))  #dfc다


#숫자 0개 또는 1개
print(re.fullmatch(r"\d?", ""))     # ✅ '' (숫자 없음)
print(re.fullmatch(r"\d?", "5"))    # ✅ '5' (숫자 1개)
print(re.fullmatch(r"\d?", "57"))   # ❌ '57'은 숫자 2개 → 안 됨

#숫자한개이상
print(re.fullmatch(r"\d+", "5"))     # ✅ '5'
print(re.fullmatch(r"\d+", "123"))   # ✅ '123'
print(re.fullmatch(r"\d+", ""))      # ❌ 없음 → 안 됨

#숫자 0개이상 없어도 된다.
print(re.fullmatch(r"\d*", ""))       # ✅ '' (0개 허용)
print(re.fullmatch(r"\d*", "12345"))  # ✅ '12345'
print(re.fullmatch(r"\d*", "a"))      # ❌ 'a'는 숫자가 아님

pattern = r"\d{3}"

print(re.fullmatch(pattern, "123"))    # ✅ True (정확히 3자리 숫자)
print(re.fullmatch(pattern, "45"))     # ❌ False (2자리)
print(re.fullmatch(pattern, "7890"))   # ❌ False (4자리)
print(re.fullmatch(pattern, "abc"))    # ❌ False (숫자 아님)


pattern = r"\d{3,4}"

print(re.fullmatch(pattern, "123"))     # ✅ 3자리 → match
print(re.fullmatch(pattern, "1234"))    # ✅ 4자리 → match
print(re.fullmatch(pattern, "12"))      # ❌ 2자리 → no match
print(re.fullmatch(pattern, "12345"))   # ❌ 5자리 → no match

# \b 정규표현식에서 단어 경계(boundary) 를 의미합니다.
# 즉, 단어의 시작 또는 끝을 나타냅니다.
# 공백, 특수문자, 문장 끝 등과 연결된 위치가 경계입니


text = "I like apple and pineapple."
#단어 경계가 공백이나 문장의 끝일때만만
# 'apple'이라는 단어만 찾기 (단어 경계 \b 사용)
match = re.search(r"\bapple\b", text)
print(match.group())  # apple