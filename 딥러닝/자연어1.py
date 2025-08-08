#우리가 직접 벡터화를 해보자 
import string 
#print("구두점", string.punctuation)

class MyVectorize:
    def standardize(self, text): #표준화
        text = text.lower() #1.전부 소문자로 만든다 
        return "".join(c for c in text  if c not in string.punctuation)
        #구두점 제거한 문장을 만들어서 반환함 
    
    def tokenize(self, text): #토큰화 
        return text.split()  #잘라서 보낸다. 

    #어휘사전 만드는 함수 
    def make_vocabulary(self, dataset):
        #1.전체 데이터셋을 순회하녀 단어 사전을 만든다. 
        #2.기본저긍로 빈문장"" 과 UNK- 자주 사용하는 단어만, 자주 사용하는 단어가 아니면 UNK로 표현한다.(unknown)
        #3.새로운 단어가 발견되면 어휘사전에 추가하고 해당 단어에 고유한 숫자 인데스를 부여한다 
        self.vocabulary = {"":0, "[UNK]":1}  #0하고 1은 특수목적으로 사용한다 
        for text in dataset: #한문장씩 처리한다
            text = self.standardize(text) #표준화 
            tokens = self.tokenize(text)  #토큰화 
            for token in tokens:
                if token not in self.vocabulary: #아직 어휘사전에 없는 단어이면 추가한다 
                    self.vocabulary[token] = len(self.vocabulary)

        #역순서  단어:숫자 => 숫자:단어로 바꾼다
        self.inverse_vocabulary = dict( (v,k) for k,v in self.vocabulary.items())  

    def encode(self, text): #문장을 받아서 문장 벡터화를 한다 
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join(self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)  

mv = MyVectorize()
dataset=[
    "I write, erase, reqrite",
    "Erase again, and then",
    "A poppy blooms",
    "Dog is pretty"
]

test = mv.standardize(dataset[1])
print(test)
test = mv.tokenize(test)
print(test)

mv.make_vocabulary(dataset)
print(  mv.vocabulary)
print(mv.inverse_vocabulary)



print( mv.encode("I write erase"))
print( mv.decode([2,3,4,23]))










